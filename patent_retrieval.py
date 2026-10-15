from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the project .env file so OPENAI_API_KEY is available to this app.
load_dotenv()

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(
    page_title="RelevanceIQ | Patent Search",
    page_icon="🔎",
    layout="wide",
)

PUBLICATION_COL = "Publication Number"
TITLE_COL = "Title"
ABSTRACT_COL = "Abstract"
DESCRIPTION_COL = "Description"
EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_CHUNK_CHARS = 3_000
QUERY_MAX_CHARS = 6_000
EMBEDDING_BATCH_SIZE = 64

COLUMN_ALIASES = {
    PUBLICATION_COL: [
        "publication number", "publication_number", "publication no",
        "publication id", "patent number", "patent no", "publication",
        "patent_id", "patent id",
    ],
    TITLE_COL: ["title", "patent title", "invention title"],
    ABSTRACT_COL: ["abstract", "patent abstract"],
    DESCRIPTION_COL: [
        "description", "patent description", "detailed description",
        "full description", "specification",
    ],
}

def normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()

def prepare_patents_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the uploaded patent dataset and create searchable text."""
    if source_df is None or source_df.empty:
        raise ValueError("The uploaded patent dataset is empty.")

    df = source_df.copy()
    lookup = {normalize_header(column): column for column in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        source_column = next(
            (column for column in df.columns
             if normalize_header(column) == normalize_header(canonical)),
            None,
        )
        if source_column is None:
            for alias in aliases:
                matched = lookup.get(normalize_header(alias))
                if matched is not None:
                    source_column = matched
                    break

        if source_column is None:
            if canonical == ABSTRACT_COL:
                df[canonical] = ""
            else:
                raise ValueError(
                    f"Required column '{canonical}' was not found. "
                    f"Accepted alternatives: {', '.join(aliases)}"
                )
        elif source_column != canonical:
            df[canonical] = df[source_column]

    for column in [PUBLICATION_COL, TITLE_COL, ABSTRACT_COL, DESCRIPTION_COL]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["search_text"] = (
        df[TITLE_COL] + " " + df[ABSTRACT_COL] + " " + df[DESCRIPTION_COL]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["search_text"].str.len() > 0].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset contains no searchable patent records.")

    missing_ids = df[PUBLICATION_COL].eq("")
    if missing_ids.any():
        missing_count = int(missing_ids.sum())
        df.loc[missing_ids, PUBLICATION_COL] = [
            f"Unknown publication #{index + 1}" for index in range(missing_count)
        ]
    return df


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def get_ranks(scores: list[float] | np.ndarray) -> list[int]:
    """Convert scores to descending, one-based ranks."""
    values = np.asarray(scores, dtype=float)
    ordering = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[ordering] = np.arange(1, len(values) + 1)
    return ranks.tolist()

def tfidf_search(query: str, documents: list[str]) -> list[float]:
    corpus = [query] + documents
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=300_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(matrix[0], matrix[1:]).ravel().tolist()


def bm25_search(query: str, documents: list[str]) -> list[float] | None:
    if BM25Okapi is None:
        return None
    tokenized_documents = [tokenize(document) for document in documents]
    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0] * len(documents)
    bm25 = BM25Okapi(tokenized_documents)
    return [float(score) for score in bm25.get_scores(query_tokens)]

def get_openai_api_key() -> str | None:
    """Read the key from Streamlit secrets first, then environment variables."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("OPENAI_API_KEY")


def split_for_embeddings(text: str, max_chars: int = EMBEDDING_CHUNK_CHARS) -> list[str]:
    """Split long patent text into overlapping, API-safe embedding chunks."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # Small overlap helps avoid splitting a technical phrase at a boundary.
    overlap = 200
    step = max_chars - overlap
    chunks = [text[start:start + max_chars] for start in range(0, len(text), step)]
    return [chunk for chunk in chunks if chunk.strip()]

@st.cache_data(show_spinner=False, max_entries=4)
def get_document_embeddings(
    documents: tuple[str, ...], model_name: str, api_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Embed patent chunks once; return chunk vectors and their patent indexes."""
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing. Install it with: pip install openai")

    chunks: list[str] = []
    patent_indexes: list[int] = []
    for patent_index, document in enumerate(documents):
        for chunk in split_for_embeddings(document):
            chunks.append(chunk)
            patent_indexes.append(patent_index)

    client = OpenAI(api_key=api_key)
    vectors: list[list[float] | None] = [None] * len(chunks)

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=model_name, input=batch)
        for item in response.data:
            vectors[start + item.index] = item.embedding

    if any(vector is None for vector in vectors):
        raise RuntimeError("The embedding API did not return an embedding for every patent chunk.")
    return np.asarray(vectors, dtype=np.float32), np.asarray(patent_indexes, dtype=np.int32)

def semantic_search(
    query: str,
    documents: list[str],
    api_key: str,
    model_name: str = EMBEDDING_MODEL,
) -> list[float]:
    """Score each patent by its most semantically similar text chunk."""
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing. Install it with: pip install openai")

    client = OpenAI(api_key=api_key)
    safe_query = str(query).strip()[:QUERY_MAX_CHARS]
    query_response = client.embeddings.create(model=model_name, input=safe_query)
    query_vector = np.asarray(query_response.data[0].embedding, dtype=np.float32)

    chunk_vectors, patent_indexes = get_document_embeddings(
        tuple(documents), model_name, api_key
    )

    # Cosine similarity between the query and every chunk.
    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    denominator = chunk_norms * query_norm
    denominator[denominator == 0] = 1.0
    chunk_scores = (chunk_vectors @ query_vector) / denominator

    # material found deep in a long description while returning one score per patent.
    patent_scores = np.full(len(documents), -1.0, dtype=np.float32)
    for chunk_index, patent_index in enumerate(patent_indexes):
        patent_scores[patent_index] = max(
            patent_scores[patent_index], float(chunk_scores[chunk_index])
        )
    return patent_scores.astype(float).tolist()

def retrieve_patents(
    source_df: pd.DataFrame,
    product_description: str,
    similarity_threshold: float = 0.10,
    semantic_threshold: float = 0.30,
    bm25_rank_cutoff: int = 100,
    use_openai_embeddings: bool = True,
    rrf_k: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retrieve candidates if they qualify via TF-IDF, BM25 rank, or semantic score."""
    query = str(product_description or "").strip()
    if not query:
        raise ValueError("Please enter a product description.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("TF-IDF threshold must be between 0 and 1.")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("Semantic threshold must be between 0 and 1.")
    if bm25_rank_cutoff < 1:
        raise ValueError("BM25 candidate-rank cutoff must be at least 1.")

    df = prepare_patents_dataframe(source_df)
    documents = df["search_text"].tolist()

    tfidf_scores = tfidf_search(query, documents)
    df["TF-IDF Similarity"] = tfidf_scores
    rankings: dict[str, list[float]] = {"TF-IDF": tfidf_scores}

    bm25_scores = bm25_search(query, documents)
    bm25_ranks: list[int] | None = None
    if bm25_scores is not None:
        bm25_ranks = get_ranks(bm25_scores)
        df["BM25 Score"] = bm25_scores
        df["BM25 Rank"] = bm25_ranks
        rankings["BM25"] = bm25_scores

    semantic_scores: list[float] | None = None
    if use_openai_embeddings:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY as an environment variable "
                "or add OPENAI_API_KEY = \"your-key\" to .streamlit/secrets.toml."
            )
        semantic_scores = semantic_search(query, documents, api_key)
        df["Semantic Similarity"] = semantic_scores
        rankings["Semantic"] = semantic_scores

    # Combined retrieval rule: a patent qualifies through ANY enabled method.
    qualifying_mask = df["TF-IDF Similarity"] >= similarity_threshold
    if bm25_ranks is not None:
        qualifying_mask |= np.asarray(bm25_ranks) <= bm25_rank_cutoff
    if semantic_scores is not None:
        qualifying_mask |= df["Semantic Similarity"] >= semantic_threshold

    df["Search Score"] = 0.0
    for method_name, scores in rankings.items():
        ranks = get_ranks(scores)
        df[f"{method_name} Rank"] = ranks
        df["Search Score"] += [1.0 / (rrf_k + rank) for rank in ranks]

    candidates = df.loc[qualifying_mask].copy()
    candidates = candidates.sort_values(
        ["Search Score", "TF-IDF Similarity"], ascending=False
    ).reset_index(drop=True)
    candidates["Search Rank"] = range(1, len(candidates) + 1)

    metadata = {
        "methods_used": list(rankings.keys()),
        "threshold_method": "TF-IDF OR BM25 rank OR semantic similarity",
        "similarity_threshold": similarity_threshold,
        "semantic_threshold": semantic_threshold if semantic_scores is not None else None,
        "bm25_rank_cutoff": bm25_rank_cutoff if bm25_ranks is not None else None,
        "semantic_search_used": semantic_scores is not None,
        "embedding_model": EMBEDDING_MODEL if semantic_scores is not None else None,
        "cross_encoder_used": False,
        "corpus_size": len(df),
        "candidates_returned": len(candidates),
    }
    return candidates, metadata


def main() -> None:
    st.title("Patent Search")
    st.caption(
        "Search directly from the product description using TF-IDF, BM25, and "
        "OpenAI semantic embeddings. This retrieves patent candidates only; it does "
        "not run the separate relevance-analysis workflow."
    )
    st.divider()

    with st.form("patent_search_form"):
        product_description = st.text_area(
            "Product Description",
            height=220,
            placeholder=(
                "Describe the product, its technical operation, components, "
                "and intended functionality..."
            ),
        )
        uploaded_file = st.file_uploader(
            "Upload Patent Corpus (.xlsx)",
            type=["xlsx"],
            help=(
                "Required columns: Publication Number, Title and Description. "
                "Abstract is optional."
            ),
        )

        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.slider(
                "Minimum TF-IDF similarity threshold",
                min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                help="A patent qualifies through TF-IDF if its cosine similarity meets this threshold.",
            )
            semantic_threshold = st.slider(
                "Minimum semantic similarity threshold",
                min_value=0.0, max_value=1.0, value=0.30, step=0.01,
                help=(
                    "A patent also qualifies if its OpenAI embedding cosine similarity meets this threshold. "
                    "Tune this using representative search examples."
                ),
            )
        with col2:
            bm25_rank_cutoff = st.number_input(
                "BM25 candidate-rank cutoff",
                min_value=1, max_value=1_000_000, value=100, step=25,
                help="A patent qualifies if its BM25 rank is within this cutoff. This is configurable, not a BM25 limit.",
            )
            st.text_input(
                "Embedding model", value=EMBEDDING_MODEL, disabled=True,
                help="Hosted OpenAI model; no local model download is required.",
            )

        use_openai_embeddings = st.checkbox(
            "Use OpenAI semantic embeddings (requires API key and API billing)",
            value=True,
        )
        submitted = st.form_submit_button(
            "Search Patents", type="primary", use_container_width=True
        )

    if submitted:
        if not product_description.strip():
            st.error("Please enter a product description.")
        elif uploaded_file is None:
            st.error("Please upload the patent corpus Excel file.")
        else:
            try:
                source_df = pd.read_excel(uploaded_file, engine="openpyxl")
                with st.spinner(
                    "Searching patents... OpenAI may embed the corpus on the first run; cached embeddings are reused afterward."
                ):
                    candidates, metadata = retrieve_patents(
                        source_df=source_df,
                        product_description=product_description,
                        similarity_threshold=similarity_threshold,
                        semantic_threshold=semantic_threshold,
                        bm25_rank_cutoff=int(bm25_rank_cutoff),
                        use_openai_embeddings=use_openai_embeddings,
                    )
                st.session_state["patent_search_candidates"] = candidates
                st.session_state["patent_search_metadata"] = metadata
                st.session_state["patent_search_product_description"] = product_description
                st.session_state["patent_search_corpus_name"] = uploaded_file.name
                st.success(
                    f"Retrieved {len(candidates)} qualifying patents from "
                    f"{metadata['corpus_size']} searchable records."
                )
            except Exception as exc:
                st.error(f"Patent search failed: {exc}")

    candidates = st.session_state.get("patent_search_candidates")
    metadata = st.session_state.get("patent_search_metadata")
    if candidates is not None and metadata is not None:
        st.divider()
        st.subheader("Retrieved Patent Candidates")
        semantic_status = (
            f"OpenAI embeddings ({metadata['embedding_model']})"
            if metadata["semantic_search_used"] else "OpenAI embeddings disabled"
        )
        st.caption(
            f"Corpus: {st.session_state.get('patent_search_corpus_name', 'Uploaded file')} | "
            f"Records searched: {metadata['corpus_size']} | "
            f"Inclusion rule: {metadata['threshold_method']} | "
            f"TF-IDF threshold: {metadata['similarity_threshold']:.2f} | "
            f"Semantic: {semantic_status} | "
            f"Ranking methods: {', '.join(metadata['methods_used'])}"
        )
        st.metric("Qualifying patents", len(candidates))

        if candidates.empty:
            st.warning(
                "No patents met the selected retrieval rule. Lower the thresholds or increase the BM25 candidate-rank cutoff."
            )
        else:
            preferred_columns = [
                PUBLICATION_COL, TITLE_COL, ABSTRACT_COL, DESCRIPTION_COL,
                "TF-IDF Similarity", "TF-IDF Rank", "BM25 Score", "BM25 Rank",
                "Semantic Similarity", "Semantic Rank", "Search Rank", "Search Score",
            ]
            display_columns = [
                column for column in preferred_columns if column in candidates.columns
            ]
            st.dataframe(
                candidates[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Search Rank": st.column_config.NumberColumn("Search Rank", format="%d"),
                    "Search Score": st.column_config.NumberColumn("RRF Search Score", format="%.6f"),
                    "TF-IDF Similarity": st.column_config.NumberColumn("TF-IDF Similarity", format="%.4f"),
                    "BM25 Score": st.column_config.NumberColumn("BM25 Score", format="%.4f"),
                    "Semantic Similarity": st.column_config.NumberColumn("Semantic Similarity", format="%.4f"),
                },
            )
            csv_data = candidates[display_columns].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download All Retrieved Patents (CSV)",
                data=csv_data,
                file_name="RelevanceIQ_Patent_Search_Results.csv",
                mime="text/csv",
            )

            with st.expander("Inspect Patent Details"):
                for _, row in candidates.iterrows():
                    publication_number = row.get(PUBLICATION_COL, "Unknown publication")
                    title = row.get(TITLE_COL, "Untitled patent")
                    with st.expander(f"{publication_number} — {title}"):
                        st.markdown("**Abstract**")
                        st.write(row.get(ABSTRACT_COL, "") or "No abstract provided.")
                        st.markdown("**Description**")
                        st.write(row.get(DESCRIPTION_COL, "") or "No description provided.")

        st.divider()
        st.subheader("Next Step")
        st.info(
            "Patent search is complete. The separate relevance-analysis workflow has not been started. "
            "To perform relevance analysis, use its usual Excel file containing All Claims and Independent Claim."
        )


if __name__ == "__main__":
    main()
