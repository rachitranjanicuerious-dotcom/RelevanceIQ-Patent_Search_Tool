import streamlit as st
import sqlite3
import uuid
import os
import textwrap
import pandas as pd
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from node import llm, feature_parser
from graph import workflow
from database import create_table, save_result
import templates as ui

# NEW: Import corpus preparation and patent retrieval functions.
from patent_retrieval import retrieve_patents, cross_encoder_rerank 
from sentence_transformers import SentenceTransformer

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="RelevanceIQ | iCuerious",
    page_icon="Icon.png",
    layout="wide",
    initial_sidebar_state="expanded"

)

# NEW: Cache the embedding model so Streamlit reruns do not reload it repeatedly.
@st.cache_resource

def get_retrieval_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_cross_encoder_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ============================================================
# HTML HELPER
# ============================================================

def render_html(content):

    """
    Safely render custom HTML.
    textwrap.dedent() prevents Streamlit Markdown
    from interpreting indented HTML as a code block.
    """

    content = textwrap.dedent(content)

    if hasattr(st, "html"):
        st.html(content)

    else:
        st.markdown(
            content,
            unsafe_allow_html=True
        )

def count_rating(df, column, value):
    return len(df[df[column] == value])

def render_cards(items):

    """Render a row of info cards from (title, value) pairs."""
    columns = st.columns(len(items))
    for column, (title, value) in zip(columns, items):

        with column:

            render_html(ui.info_card(title, value))

# ============================================================
# CUSTOM CSS  (styles.css)
# ============================================================

render_html(ui.load_css())

# ============================================================
# LOGIN
# ============================================================

def login():

    render_html(ui.login_card())
    username = st.text_input(

        "Username",
        placeholder="Enter username",
        key="login_username"

    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        key="login_password"

    )

    if st.button(
        "Login",
        use_container_width=True,
        key="login_button"
    ):

        users = st.secrets["users"]

        if username in users and password == users[username]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()

        else:

            st.error(
                "Invalid username or password"
            )

# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "df" not in st.session_state:
    st.session_state.df = None

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "product_description" not in st.session_state:
    st.session_state.product_description = ""

if "relevance_framework" not in st.session_state:
    st.session_state.relevance_framework = ""

if "primary_product_features" not in st.session_state:
    st.session_state.primary_product_features = []

if "secondary_product_features" not in st.session_state:
    st.session_state.secondary_product_features = []

if "product_features_approved" not in st.session_state:
    st.session_state.product_features_approved = False

# NEW: Track whether corpus search has already run in this analysis session.
# This prevents retrieval from repeating on every Streamlit rerun.

if "patent_retrieval_done" not in st.session_state:
    st.session_state.patent_retrieval_done = False

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = False

# ============================================================
# DATABASE
# ============================================================

create_table()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(ui.sidebar_brand())
    render_html(ui.sidebar_section("Application"))
    page = st.radio(
        "Navigation",
        [
            "Analysis",
            "Results",
            "Database"
        ],

        label_visibility="collapsed"
    )

    render_html(ui.sidebar_tool())
    render_html(ui.sidebar_user(st.session_state.username))
    st.markdown(

        "<br>",
        unsafe_allow_html=True
    )

    if st.button(
        "Logout",
        use_container_width=True,
        key="logout_button"

    ):

        st.session_state.logged_in = False
        st.session_state.pop(
            "username",
            None
        )
        st.rerun()

# ============================================================
# TOP HEADER
# ============================================================

render_html(ui.top_header())

# ============================================================
# DATABASE PAGE
# ============================================================

if page == "Database":

    render_html(

        ui.hero(
            "RelevanceIQ",
            "Analysis Database",
            "Review previously generated patent relevance "
            "analyses stored by the application."
        )

    )

    conn = sqlite3.connect(
        "RelevanceIQ.db"
    )

    try:

        database_df = pd.read_sql_query(

            """
            SELECT *
            FROM FTO_results
            ORDER BY id DESC
            """,

            conn

        )

    except Exception as e:

        database_df = pd.DataFrame()

        st.error(
            f"Unable to load database records: {e}"
        )

    finally:

        conn.close()

    if not database_df.empty:

        # Number of records
        
        st.write(

            f"**{len(database_df)} records found**"

        )

        st.dataframe(

            database_df,
            use_container_width=True,
            hide_index=True

        )

        st.download_button(

            "⬇ Download Database",
            database_df.to_csv(index=False),
            "RelevanceIQ_Database.csv",
            "text/csv",
            key="download_database"

        )

    else:

        st.info(

            "No database records are available."

        )

    st.stop()

# ============================================================
# RESULTS PAGE
# ============================================================

if page == "Results":

    render_html(

        ui.hero(

            "RelevanceIQ",
            "Analysis Results",
            "Patent relevance results generated from the "
            "current analysis session."

        )

    )

    if st.session_state.results:

        result_df = pd.DataFrame(
            st.session_state.results
        )

        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        render_cards([

            ("Patents Analysed", len(result_df)),
            ("Highly Relevant", count_rating(result_df, "Relevance", "H")),
            ("M+", count_rating(result_df, "Relevance", "M+")),
            ("L", count_rating(result_df, "Relevance", "L")),
            ("NR", count_rating(result_df, "Relevance", "NR")),

        ])

        # ----------------------------------------------------

        # FRAMEWORK-ONLY SUMMARY

        # ----------------------------------------------------

        render_html(

            ui.section(

                "F",

                "Framework-Only Classification",

                "Classification based specifically on the "

                "user-provided relevance framework."

            )

        )

        render_cards([

            ("H", count_rating(result_df, "Relevance_Framework_Only", "H")),

            ("M+", count_rating(result_df, "Relevance_Framework_Only", "M+")),

            ("L", count_rating(result_df, "Relevance_Framework_Only", "L")),

            ("NR", count_rating(result_df, "Relevance_Framework_Only", "NR")),

        ])

        # ====================================================

        # RESULTS TABLE

        # ====================================================

        render_html(

            ui.section(

                "R",

                "Common Relevance Framework Results"

            )

        )

        st.dataframe(

            result_df,

            use_container_width=True,

            hide_index=True

        )

        # ====================================================

        # DOWNLOAD

        # ====================================================

        st.download_button(

            "Download Results",

            result_df.to_csv(index=False),

            "RelevanceIQ_Results.csv",

            "text/csv",

            key="download_results"

        )

    else:

        st.info(

            "No analysis results are available yet. "

            "Run an analysis first."

        )

    st.stop()

# ============================================================
# ANALYSIS PAGE
# ============================================================

render_html(

    ui.hero(

        "Patent Intelligence • FTO",
        "Patent Relevance Analysis",
        "Evaluate patents against a target product using "
        "a defined relevance framework and human-reviewed "
        "technical feature extraction."

    )

)

# ============================================================
# INPUT SECTION
# ============================================================

render_html(
    ui.section(
        1,
        "Target Product",
        "Describe the technical product that you want to analyze."
    )

)

product_description = st.text_area(

    "Product Description",
    value=st.session_state.product_description,
    height=180,
    placeholder=(

        "Enter the technical description of the target product..."

    ),

    label_visibility="collapsed",
    key="product_description_input"

)

# ============================================================
# PATENT FILE
# ============================================================

render_html(

    ui.section(

        2,

        "Patent Dataset",

        "Upload the patent corpus. RelevanceIQ will retrieve candidate patents from it before rating them."

    )

)

uploaded_file = st.file_uploader(

    "Upload Patent Corpus Excel File",

    type=["xlsx"],

    label_visibility="collapsed",

    key="patent_file"

)

col1, col2 = st.columns([3, 1])



with col1:

    render_html(ui.input_info())



with col2:

    template_path = (

        "RelevanceIQ_Input_Data_Schema.xlsx"

    )

    if os.path.exists(template_path):

        with open(

            template_path,

            "rb"

        ) as f:

            st.download_button(

                "Download Template",

                f.read(),

                "RelevanceIQ_Input_Data_Schema.xlsx",

                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                use_container_width=True,

                key="download_template"

            )



# ============================================================

# FRAMEWORK



# ============================================================

render_html(

    ui.section(

        3,

        "Relevance Framework",

        "Define the criteria that will be used to determine "

        "patent relevance."

    )

)

relevance_framework = st.text_area(

    "Relevance Framework",

    value=st.session_state.relevance_framework,

    height=220,

    placeholder=(

        "Enter the relevance framework used to classify "

        "patents as H, M+, L or NR..."

    ),

    label_visibility="collapsed",

    key="relevance_framework_input"

)



# ============================================================

# RUN ANALYSIS



# ============================================================

st.markdown(

    "<br>",

    unsafe_allow_html=True

)

run_col1, run_col2, run_col3 = st.columns(

    [1, 2, 1]

)



with run_col2:

    run = st.button(

        "Run Relevance Analysis",

        use_container_width=True,

        key="run_analysis_button"

    )



if run:

    # --------------------------------------------------------

    # VALIDATION

    # --------------------------------------------------------

    if uploaded_file is None:

        st.warning(

            "Please upload an Excel file containing the patents."

        )

        st.stop()

    if not product_description.strip():

        st.warning(

            "Please enter the product description."

        )

        st.stop()

    if not relevance_framework.strip():

        st.warning(

            "Please provide the relevance framework."

        )

        st.stop()

    # --------------------------------------------------------

    # READ EXCEL

    # --------------------------------------------------------

    try:

        uploaded_df = pd.read_excel(

            uploaded_file

        )

    except Exception as e:

        st.error(

            f"Unable to read the Excel file: {e}"

        )

        st.stop()

    if uploaded_df.empty:

        st.warning(

            "The uploaded Excel file does not contain "

            "any patent rows."

        )

        st.stop()



    # --------------------------------------------------------

    # VALIDATE / NORMALIZE CORPUS COLUMNS

    # --------------------------------------------------------

    # Normalize uploaded corpus headers so retrieval accepts either the



    column_lookup = {str(column).strip().lower(): column for column in uploaded_df.columns}

    column_aliases = {

        "Publication Number": ["publication_number", "publication number", "Publication_Number" , "Publication Number"],

        "Title": ["title" , "Title"],

        "Abstract": ["abstract" , "Abstract"],

        "Independent Claim": ["independent_claim", "independent claim", "independent claims", "independent_claims"],

        "All Claims": ["all_claims", "all claims", "claims"],

        "Relevance Human": ["relevance_human" , "Relevance_Human" , "Relevance Human"],

    }



    for expected_column, aliases in column_aliases.items():

        if expected_column not in uploaded_df.columns:

            source_column = next(

                (column_lookup[alias.strip().lower()] for alias in aliases if alias.strip().lower() in column_lookup),

                None,

            )

            if source_column is not None:

                uploaded_df[expected_column] = uploaded_df[source_column]



    required_columns = [

        "Publication Number",

        "Title",

        "Abstract",

        "Independent Claim",

        "All Claims",

    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in uploaded_df.columns

    ]



    if missing_columns:

        st.error(

            "The uploaded Excel file is missing the following "

            f"required columns: {', '.join(missing_columns)}"

        )

        st.stop()



    # --------------------------------------------------------

    # SAVE SESSION DATA

    # --------------------------------------------------------

    st.session_state.df = uploaded_df

    st.session_state.product_description = (

        product_description

    )

    st.session_state.relevance_framework = (

        relevance_framework

    )

    st.session_state.results = []

    st.session_state.current_index = 0

    st.session_state.primary_product_features = []

    st.session_state.secondary_product_features = []

    st.session_state.product_features_approved = False

    # Reset retrieval for every new product/corpus analysis.

    st.session_state.patent_retrieval_done = False

    st.session_state.analysis_started = True

    st.session_state.session_id = str(

        uuid.uuid4()

    )

    st.rerun()



# ============================================================

# PRODUCT FEATURE EXTRACTION



# ============================================================



if (

    st.session_state.analysis_started

    and not st.session_state.product_features_approved

):

    # --------------------------------------------------------

    # EXTRACT FEATURES

    # --------------------------------------------------------

    if (

        not st.session_state.primary_product_features

        and not st.session_state.secondary_product_features

    ):

        render_html(

            ui.status_card(

                "Extracting Product Features",

                "RelevanceIQ is identifying the primary "

                "and secondary technical features of the "

                "target product for human review."

            )

        )

        product_prompt = f"""

You are an experienced patent analyst performing an FTO

(Freedom-to-Operate) technical analysis.

Extract the technical features explicitly disclosed in the

product description.

PRIMARY FEATURES:

- Core technical features and mechanisms.

- Features important for satisfying potential patent claim

  limitations.

SECONDARY FEATURES:

- Supporting, auxiliary, optional, control, communication,

  interface, or peripheral features.

Rules:

- Extract only explicitly disclosed features.

- Do not infer or invent features.

- Keep features concise and technically specific.

{feature_parser.get_format_instructions()}

Product Description:

{st.session_state.product_description}

"""

        response = llm.invoke(

            [

                HumanMessage(

                    content=product_prompt

                )

            ]

        )

        result = feature_parser.parse(

            response.content

        )

        st.session_state.primary_product_features = (

            result.primary_features

        )

        st.session_state.secondary_product_features = (

            result.secondary_features

        )

        st.rerun()

    # --------------------------------------------------------

    # HUMAN REVIEW

    # --------------------------------------------------------

    render_html(

        ui.section(

            "✓",

            "Review Product Features",

            "Review and edit the extracted features before "

            "continuing with patent analysis."

        )

    )

    col1, col2 = st.columns(2)

    with col1:

        edited_primary = st.text_area(

            "Primary Product Features",

            value="\n".join(

                st.session_state.primary_product_features

            ),

            height=250,

            key="product_primary_review"

        )

    with col2:

        edited_secondary = st.text_area(

            "Secondary Product Features",

            value="\n".join(

                st.session_state.secondary_product_features

            ),

            height=250,

            key="product_secondary_review"

        )

    approve_col1, approve_col2, approve_col3 = st.columns(

        [1, 2, 1]

    )

    with approve_col2:

        if st.button(

            "Approve Product Features",

            use_container_width=True,

            key="approve_product_features"

        ):

            st.session_state.primary_product_features = [

                x.strip()

                for x in edited_primary.split("\n")

                if x.strip()

            ]

            st.session_state.secondary_product_features = [

                x.strip()

                for x in edited_secondary.split("\n")

                if x.strip()

            ]

            st.session_state.product_features_approved = True

            st.rerun()

    st.stop()



# ============================================================



# NEW: PATENT RETRIEVAL FROM CORPUS

# Runs after the user approves product features and before the existing per-patent LangGraph relevance analysis.



# ============================================================



if (

    st.session_state.analysis_started

    and st.session_state.product_features_approved

    and not st.session_state.patent_retrieval_done

):

    try:

        with st.spinner(

            "Searching the patent corpus using TF-IDF, BM25 and semantic retrieval..."

        ):

            # NEW: Preserve the uploaded corpus as the source for patent search.

            source_corpus = st.session_state.df.copy()



            # NEW: Search with the original description plus user-approved features.

            approved_features = (

                st.session_state.primary_product_features

                + st.session_state.secondary_product_features

            )



            # Reuse the cached embedding model for semantic retrieval.
            embedding_model = get_retrieval_embedding_model()

            # TF-IDF, BM25 and semantic rankings are combined with RRF.
            # retrieve_patents normalizes the uploaded corpus internally.
            candidates, _ = retrieve_patents(
                source_corpus,
                st.session_state.product_description,
                approved_features,
                model=embedding_model,
            )

            if candidates.empty:
                raise ValueError("No patent candidates were retrieved from the corpus.")

            # Rerank the RRF shortlist before the existing LLM analysis.
            reranker = get_cross_encoder_reranker()
            candidates = cross_encoder_rerank(
                candidates,
                st.session_state.product_description,
                approved_features,
                top_k=30,
                reranker=reranker,
            )

            if candidates.empty:
                raise ValueError("Cross-encoder reranking returned no patent candidates.")

            # NEW: Map retrieved candidates to the column names expected by the existing LangGraph workflow. The rating workflow remains unchanged.



            workflow_df = candidates.copy()

            workflow_df["Publication Number"] = workflow_df["publication_number"]

            workflow_df["Title"] = workflow_df["title"]

            workflow_df["Abstract"] = workflow_df["abstract"]

            workflow_df["Independent Claim"] = workflow_df["independent_claim"]

            workflow_df["All Claims"] = workflow_df["claims"]



            if "Relevance Human" not in workflow_df.columns:

                workflow_df["Relevance Human"] = ""



            # NEW: Replace the full corpus with the ranked candidates, so the existing workflow rates these candidates rather than every corpus row.



            st.session_state.df = workflow_df.reset_index(drop=True)

            st.session_state.current_index = 0

            st.session_state.results = []



            # NEW: Prevent search from rerunning when Streamlit reruns the app.

            st.session_state.patent_retrieval_done = True

        st.success(

            f"Retrieved {len(st.session_state.df)} candidate patents. "

            "Starting the existing relevance workflow."

        )

        st.rerun()



    # NEW: Handle corpus-search errors before starting patent relevance rating.

    except Exception as e:

        st.error(f"Patent retrieval failed: {e}")

        st.info(

            "Check that the uploaded corpus contains the required patent columns "

            "and that retrieval dependencies are installed."

        )

        st.stop()



# ============================================================

# PATENT PROCESSING

# ============================================================



if st.session_state.analysis_started:

    df = st.session_state.df

    index = st.session_state.current_index

    # ========================================================

    # COMPLETEDS

    # ========================================================

    if index >= len(df):

        render_html(

            ui.hero(

                "Analysis Complete",

                "Patent Analysis Completed",

                "The selected patent dataset has been "

                "processed successfully."

            )

        )

        result_df = pd.DataFrame(

            st.session_state.results

        )

        # ====================================================

        # SUMMARY METRICS

        # ====================================================

        render_cards([

            ("Common Relevance Framework Results", len(result_df)),

            ("H", count_rating(result_df, "Relevance", "H")),

            ("M+", count_rating(result_df, "Relevance", "M+")),

            ("L", count_rating(result_df, "Relevance", "L")),

            ("NR", count_rating(result_df, "Relevance", "NR")),

        ])

        # ====================================================

        # FRAMEWORK ONLY SUMMARY

        # ====================================================

        render_html(

            ui.section(

                "F",

                "Framework-Only Results",

                "Classification based only on the "

                "user-provided relevance framework."

            )

        )

        render_cards([

            ("H", count_rating(result_df, "Relevance_Framework_Only", "H")),

            ("M+", count_rating(result_df, "Relevance_Framework_Only", "M+")),

            ("L", count_rating(result_df, "Relevance_Framework_Only", "L")),

            ("NR", count_rating(result_df, "Relevance_Framework_Only", "NR")),

        ])

        # ====================================================

        # RESULTS

        # ====================================================

        render_html(

            ui.section(

                "R",

                "Results"

            )

        )

        st.dataframe(

            result_df,

            use_container_width=True,

            hide_index=True

        )

        # ====================================================

        # DOWNLOAD

        # ====================================================

        st.download_button(

            "Download Results",

            result_df.to_csv(index=False),

            "RelevanceIQ_Results.csv",

            "text/csv",

            key="download_final_results"

        )

    # ========================================================

    # PROCESS CURRENT PATENT

    # ========================================================

    else:

        try:

            progress_value = (

                index / len(df)

            )

            st.progress(

                progress_value

            )

            render_html(

                ui.status_card(

                    f"Patent {index + 1} of {len(df)}",

                    "RelevanceIQ is processing the current patent."

                )

            )

            row = df.iloc[index]

            config = {

                "configurable": {

                    "thread_id":

                    f"{st.session_state.session_id}_patent_{index}"

                }

            }

            snapshot = workflow.get_state(

                config

            )

            # ====================================================

            # FIRST EXECUTION

            # ====================================================

            if not snapshot or not snapshot.values:

                state = {

                    "product_description":

                        st.session_state.product_description,

                    "publication_number":

                        str(row["Publication Number"]),

                    "title":

                        row["Title"],  ## changed

                    "abstract":

                        row["Abstract"],

                    "independent_claim":

                        row["Independent Claim"],

                    "all_claims":

                        row["All Claims"],

                    "relevance_human":

                        row["Relevance Human"] ,

                    "relevance_framework":

                        st.session_state.relevance_framework,

                    "primary_product_features":

                        st.session_state.primary_product_features,

                    "secondary_product_features":

                        st.session_state.secondary_product_features,

                    "primary_patent_features":

                        [],

                    "secondary_patent_features":

                        [],

                    "comparison":

                        [],

                    "relevance":

                        "",

                    "confidence":

                        0,

                    "rationale":

                        "",

                    "reasoning":

                        "",

                    "relevance_framework_only":

                        "",

                    "confidence_framework_only":

                        0,

                    "rationale_framework_only":

                        "",

                    "reasoning_framework_only":

                        "",

                    "review_type":

                        ""

                }

                workflow.invoke(

                    state,

                    config=config

                )

                snapshot = workflow.get_state(

                    config

                )

            # ====================================================

            # HUMAN INTERRUPT

            # ====================================================

            if snapshot.interrupts:

                review = snapshot.interrupts[0].value

                if review["review_type"] == "product":

                    render_html(

                        ui.section(

                            "!",

                            "Human Review Required",

                            "Review the extracted product features "

                            "before patent relevance analysis continues."

                        )

                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        edited_primary = st.text_area(

                            "Primary Product Features",

                            value="\n".join(

                                review["primary_features"]

                            ),

                            height=230,

                            key=f"primary_{index}_product"

                        )

                    with col2:

                        edited_secondary = st.text_area(

                            "Secondary Product Features",

                            value="\n".join(

                                review["secondary_features"]

                            ),

                            height=230,

                            key=f"secondary_{index}_product"

                        )

                    if st.button(

                        "Approve & Continue",

                        key=f"approve_{index}_product",

                        use_container_width=True

                    ):

                        approved_features = {

                            "primary_features": [

                                x.strip()

                                for x in edited_primary.split("\n")

                                if x.strip()

                            ],

                            "secondary_features": [

                                x.strip()

                                for x in edited_secondary.split("\n")

                                if x.strip()

                            ]

                        }

                        workflow.invoke(

                            Command(

                                resume=approved_features

                            ),

                            config=config

                        )

                        st.rerun()

            # ====================================================

            # CONTINUE WORKFLOW

            # ====================================================

            elif snapshot.next:

                st.info(

                    "Continuing patent analysis..."

                )

                st.rerun()

            # ====================================================

            # WORKFLOW FINISHED

            # ====================================================

            else:

                result = snapshot.values

                save_result(

                    result

                )

                st.session_state.results.append({

                    "Publication Number":

                        result["publication_number"],

                    "Title":

                        result["title"],

                    "Relevance Human":

                        result["relevance_human"],

                    "Patent Primary Features":

                        result["primary_patent_features"],

                    "Patent Secondary Features":

                        result["secondary_patent_features"],

                    "Relevance":

                        result["relevance"],

                    "Confidence":

                        result["confidence"],

                    "Rationale":

                        result["rationale"],

                    "Reasoning":

                        result["reasoning"],

                    "Relevance_Framework_Only":

                        result["relevance_framework_only"],

                    "Confidence_Framework_Only":

                        result["confidence_framework_only"],

                    "Rationale_Framework_Only":

                        result["rationale_framework_only"],

                    "Reasoning_Framework_Only":

                        result["reasoning_framework_only"]

                })

                st.session_state.current_index += 1

                st.rerun()

        except Exception as e:

            st.session_state.analysis_started = False

            st.session_state.analysis_error = True

            st.error(

                f"Analysis stopped at Patent {index + 1}: "

                f"{row['Publication Number']}"

            )

            st.warning(

                f"Error: {str(e)}"

            )

            st.info(

                "Correct the issue and click Resume Analysis "

                "to continue from this patent."

            )



if st.session_state.get("analysis_error", False):

    if st.button(

        "▶ Resume Analysis",

        use_container_width=True,

        key="resume_analysis"

    ):

        st.session_state.analysis_error = False

        st.session_state.analysis_started = True

        st.rerun()



# ============================================================

# FOOTER



# ============================================================

render_html(ui.footer())
