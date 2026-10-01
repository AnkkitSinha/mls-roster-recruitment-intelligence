import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = (
    BASE_DIR
    / "vector_store"
    / "mls_roster_intelligence.faiss"
)

METADATA_FILE = (
    BASE_DIR
    / "vector_store"
    / "mls_roster_intelligence_metadata.json"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "all-MiniLM-L6-v2"
)

GEMINI_MODEL = "gemini-3.5-flash-lite"

VECTOR_CANDIDATES = 20
LEXICAL_CANDIDATES = 20
FINAL_TOP_K = 5
RRF_K = 60


# PAGE CONFIGURATION

st.set_page_config(
    page_title="MLS Roster & Recruitment Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# STYLES

st.html(
    """
    <style>

        :root {
            --bg: #0E1117;
            --panel: #161A20;
            --border: #2A313B;
            --text: #FAFAFA;
            --muted: #AEB7C2;
            --accent: #E65B54;
            --accent-soft: rgba(230, 91, 84, 0.14);
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 80% 0%,
                    rgba(230, 91, 84, 0.09),
                    transparent 32%
                ),
                var(--bg);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.1rem;
            padding-bottom: 3.5rem;
        }

        .hero {
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 30px 32px;
            background:
                linear-gradient(
                    135deg,
                    rgba(230, 91, 84, 0.13),
                    rgba(22, 26, 32, 0.96) 42%,
                    rgba(22, 26, 32, 0.98)
                );
            box-shadow:
                0 16px 45px rgba(0, 0, 0, 0.22);
            margin-bottom: 18px;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.77rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            margin-bottom: 10px;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(
                2rem,
                4vw,
                3.45rem
            );
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -0.045em;
            margin: 0;
        }

        .hero-title span {
            color: var(--accent);
        }

        .hero-copy {
            color: var(--muted);
            max-width: 850px;
            font-size: 1.03rem;
            line-height: 1.65;
            margin-top: 16px;
            margin-bottom: 0;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 18px;
        }

        .pill {
            border:
                1px solid
                rgba(230, 91, 84, 0.34);
            background: var(--accent-soft);
            color: #F5C0BC;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.77rem;
            font-weight: 650;
        }

        .section-kicker {
            color: var(--accent);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            margin-bottom: 4px;
        }

        .section-title {
            color: var(--text);
            font-size: 1.34rem;
            font-weight: 780;
            margin: 0 0 4px 0;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            margin-bottom: 14px;
        }

        div[data-testid="stMetric"] {
            border:
                1px solid
                var(--border);
            background: var(--panel);
            padding: 13px 15px;
            border-radius: 15px;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border:
                1px solid
                var(--border);
            min-height: 2.75rem;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        div[data-testid="stFormSubmitButton"]
        > button {
            border-radius: 12px;
            min-height: 2.9rem;
            font-weight: 750;
            background: var(--accent);
            color: white;
            border: none;
        }

        div[data-testid="stFormSubmitButton"]
        > button:hover {
            background: #F06B64;
            color: white;
            border: none;
        }

        div[data-testid="stExpander"] {
            border:
                1px solid
                var(--border);
            border-radius: 14px;
            background: var(--panel);
        }

    </style>
    """
)


# GET GEMINI API KEY

def get_api_key():

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if key:

        return key.strip()

    try:

        key = st.secrets[
            "GEMINI_API_KEY"
        ]

        if key:

            return str(
                key
            ).strip()

    except Exception:

        pass

    return None


# METADATA HELPERS

def get_chunks(
    metadata
):

    if isinstance(
        metadata,
        list
    ):

        return metadata

    if not isinstance(
        metadata,
        dict
    ):

        raise ValueError(
            "Metadata must contain "
            "a dictionary or list."
        )

    possible_keys = (
        "chunks",
        "documents",
        "records",
        "items",
    )

    for key in possible_keys:

        value = metadata.get(
            key
        )

        if isinstance(
            value,
            list
        ):

            return value

    raise KeyError(
        "Could not find chunks "
        "inside the metadata file."
    )


def get_chunk_text(
    chunk
):

    possible_keys = (
        "text",
        "chunk_text",
        "content",
        "page_content",
    )

    for key in possible_keys:

        value = chunk.get(
            key
        )

        if value:

            return str(
                value
            ).strip()

    return ""


def first_value(
    chunk,
    keys,
    default=""
):

    for key in keys:

        value = chunk.get(
            key
        )

        if (
            value is not None
            and str(value).strip()
        ):

            return str(
                value
            ).strip()

    return default


def get_source_title(
    chunk
):

    source = first_value(
        chunk,
        (
            "source_name",
            "source",
            "document_title",
            "document",
        ),
    )

    source_type = first_value(
        chunk,
        (
            "source_type",
            "document_type",
        ),
    )

    section = first_value(
        chunk,
        (
            "section_title",
            "section",
            "title",
            "heading",
        ),
    )

    if not source:

        source_type_lower = (
            source_type.lower()
        )

        if (
            "cba"
            in source_type_lower
        ):

            source = (
                "MLS-MLSPA Collective "
                "Bargaining Agreement"
            )

        elif (
            "roster"
            in source_type_lower
        ):

            source = (
                "MLS Roster Rules"
            )

        else:

            source = (
                "Official MLS Source"
            )

    if (
        section
        and section.lower()
        not in source.lower()
    ):

        return (
            f"{source} — "
            f"{section}"
        )

    return source


def get_source_details(
    chunk
):

    details = []

    section_number = first_value(
        chunk,
        (
            "section_number",
            "article_number",
        ),
    )

    page = first_value(
        chunk,
        (
            "page_number",
            "page",
            "pdf_page",
        ),
    )

    if section_number:

        details.append(
            f"Section "
            f"{section_number}"
        )

    if page:

        details.append(
            f"Page {page}"
        )

    return " · ".join(
        details
    )


def get_source_url(
    chunk
):

    return first_value(
        chunk,
        (
            "source_url",
            "url",
            "document_url",
        ),
    )


# LOAD SYSTEM

@st.cache_resource(
    show_spinner=False
)
def load_system():

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            "FAISS index not found: "
            f"{INDEX_FILE}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "Metadata file not found: "
            f"{METADATA_FILE}"
        )

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            "Local embedding model "
            "not found: "
            f"{MODEL_DIR}"
        )

    with METADATA_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    chunks = get_chunks(
        metadata
    )

    texts = [
        get_chunk_text(
            chunk
        )
        for chunk
        in chunks
    ]

    if (
        not texts
        or not any(texts)
    ):

        raise ValueError(
            "No chunk text was found "
            "in the metadata."
        )

    index = faiss.read_index(
        str(
            INDEX_FILE
        )
    )

    if (
        index.ntotal
        != len(chunks)
    ):

        raise ValueError(
            "FAISS index and metadata "
            "do not match. "
            f"Index vectors: "
            f"{index.ntotal}; "
            f"metadata chunks: "
            f"{len(chunks)}."
        )

    # LOAD LOCAL MODEL
    # NO HUGGING FACE CONNECTION

    embedding_model = (
        SentenceTransformer(
            str(
                MODEL_DIR
            ),
            local_files_only=True,
        )
    )

    vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(
                1,
                2
            ),
        )
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    return {
        "metadata":
            metadata,
        "chunks":
            chunks,
        "texts":
            texts,
        "index":
            index,
        "embedding_model":
            embedding_model,
        "vectorizer":
            vectorizer,
        "tfidf_matrix":
            tfidf_matrix,
    }


# VECTOR SEARCH

def vector_search(
    question,
    system,
    top_n=VECTOR_CANDIDATES,
):

    model = system[
        "embedding_model"
    ]

    index = system[
        "index"
    ]

    query_vector = model.encode(
        [
            question
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32",
    )

    top_n = min(
        top_n,
        index.ntotal,
    )

    scores, ids = (
        index.search(
            query_vector,
            top_n,
        )
    )

    results = []

    for rank, (
        chunk_id,
        score,
    ) in enumerate(
        zip(
            ids[0],
            scores[0]
        ),
        start=1,
    ):

        chunk_id = int(
            chunk_id
        )

        if chunk_id < 0:

            continue

        results.append(
            {
                "chunk_id":
                    chunk_id,
                "rank":
                    rank,
                "score":
                    float(
                        score
                    ),
            }
        )

    return results


# TF-IDF SEARCH

def lexical_search(
    question,
    system,
    top_n=LEXICAL_CANDIDATES,
):

    vectorizer = system[
        "vectorizer"
    ]

    matrix = system[
        "tfidf_matrix"
    ]

    query_vector = (
        vectorizer.transform(
            [
                question
            ]
        )
    )

    scores = (
        matrix
        @ query_vector.T
    ).toarray().ravel()

    top_n = min(
        top_n,
        len(scores),
    )

    ranked_ids = (
        np.argsort(
            scores
        )[::-1][
            :top_n
        ]
    )

    results = []

    for rank, chunk_id in (
        enumerate(
            ranked_ids,
            start=1,
        )
    ):

        results.append(
            {
                "chunk_id":
                    int(
                        chunk_id
                    ),
                "rank":
                    rank,
                "score":
                    float(
                        scores[
                            chunk_id
                        ]
                    ),
            }
        )

    return results


# RECIPROCAL RANK FUSION

def reciprocal_rank_fusion(
    vector_results,
    lexical_results,
    rrf_k=RRF_K,
):

    fused = {}

    for result in (
        vector_results
    ):

        chunk_id = result[
            "chunk_id"
        ]

        fused.setdefault(
            chunk_id,
            {
                "rrf_score":
                    0.0,
                "vector_rank":
                    None,
                "vector_score":
                    None,
                "lexical_rank":
                    None,
                "lexical_score":
                    None,
            },
        )

        fused[
            chunk_id
        ][
            "rrf_score"
        ] += (
            1.0
            / (
                rrf_k
                + result[
                    "rank"
                ]
            )
        )

        fused[
            chunk_id
        ][
            "vector_rank"
        ] = result[
            "rank"
        ]

        fused[
            chunk_id
        ][
            "vector_score"
        ] = result[
            "score"
        ]

    for result in (
        lexical_results
    ):

        chunk_id = result[
            "chunk_id"
        ]

        fused.setdefault(
            chunk_id,
            {
                "rrf_score":
                    0.0,
                "vector_rank":
                    None,
                "vector_score":
                    None,
                "lexical_rank":
                    None,
                "lexical_score":
                    None,
            },
        )

        fused[
            chunk_id
        ][
            "rrf_score"
        ] += (
            1.0
            / (
                rrf_k
                + result[
                    "rank"
                ]
            )
        )

        fused[
            chunk_id
        ][
            "lexical_rank"
        ] = result[
            "rank"
        ]

        fused[
            chunk_id
        ][
            "lexical_score"
        ] = result[
            "score"
        ]

    ranked = sorted(
        fused.items(),
        key=lambda item:
            item[
                1
            ][
                "rrf_score"
            ],
        reverse=True,
    )

    return ranked


# HYBRID SEARCH

def hybrid_search(
    question,
    system,
    top_k=FINAL_TOP_K,
):

    vector_results = (
        vector_search(
            question,
            system,
        )
    )

    lexical_results = (
        lexical_search(
            question,
            system,
        )
    )

    fused_results = (
        reciprocal_rank_fusion(
            vector_results,
            lexical_results,
        )
    )

    results = []

    for final_rank, (
        chunk_id,
        scores,
    ) in enumerate(
        fused_results[
            :top_k
        ],
        start=1,
    ):

        chunk = (
            system[
                "chunks"
            ][
                chunk_id
            ]
        )

        results.append(
            {
                "rank":
                    final_rank,
                "chunk_id":
                    chunk_id,
                "chunk":
                    chunk,
                **scores,
            }
        )

    return results


# BUILD GEMINI CONTEXT

def build_context(
    retrieved_results
):

    context_parts = []

    for result in (
        retrieved_results
    ):

        number = result[
            "rank"
        ]

        chunk = result[
            "chunk"
        ]

        title = (
            get_source_title(
                chunk
            )
        )

        details = (
            get_source_details(
                chunk
            )
        )

        text = (
            get_chunk_text(
                chunk
            )
        )

        header = (
            f"[Source {number}]\n"
            f"Title: {title}"
        )

        if details:

            header += (
                "\nLocation: "
                f"{details}"
            )

        context_parts.append(
            f"{header}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


# GEMINI ANSWER GENERATION

def generate_answer(
    question,
    retrieved_results,
    api_key,
):

    context = build_context(
        retrieved_results
    )

    prompt = f"""
You are an MLS roster and recruitment
rules research assistant.

Answer the user's question using ONLY
the retrieved official-source context
below.

Rules:

- Do not invent facts that are not
  supported by the retrieved context.

- If the retrieved material is
  insufficient, clearly say what cannot
  be established from the available
  sources.

- Cite factual statements using
  [Source 1], [Source 2], and so on.

- When a detailed provision from the
  MLS-MLSPA Collective Bargaining
  Agreement and a more general MLS
  roster-rules passage overlap, prefer
  the more specific provision for
  detailed eligibility, timing, or
  contractual mechanics.

- If sources describe different
  situations, explain the distinction
  instead of treating them as a
  contradiction.

- Keep the answer practical and easy
  to read.

- Do not present the answer as legal,
  contractual, or compliance advice.

USER QUESTION:

{question}

RETRIEVED OFFICIAL-SOURCE CONTEXT:

{context}

Write the grounded answer now.
""".strip()

    client = genai.Client(
        api_key=api_key
    )

    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    )

    answer = getattr(
        response,
        "text",
        None,
    )

    if not answer:

        raise RuntimeError(
            "Gemini returned "
            "an empty response."
        )

    return answer.strip()


# HERO

st.html(
    """
    <div class="hero">

        <div class="eyebrow">
            MLS ROSTER &amp;
            RECRUITMENT INTELLIGENCE
        </div>

        <h1 class="hero-title">
            Understand MLS roster rules.
            <br>
            <span>
                Ask better recruitment
                questions.
            </span>
        </h1>

        <p class="hero-copy">
            A retrieval-augmented
            intelligence system that
            searches official MLS roster
            regulations and the MLS-MLSPA
            Collective Bargaining Agreement
            before generating a grounded
            answer.
        </p>

        <div class="pill-row">

            <span class="pill">
                Official MLS Rules
            </span>

            <span class="pill">
                MLS-MLSPA CBA
            </span>

            <span class="pill">
                Hybrid Retrieval
            </span>

            <span class="pill">
                Grounded Gemini Generation
            </span>

        </div>

    </div>
    """
)


# LOAD APPLICATION

try:

    system = load_system()

except Exception as error:

    st.error(
        "The MLS intelligence system "
        "could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# CHECK GEMINI KEY

api_key = get_api_key()

if not api_key:

    st.error(
        "Gemini API key was not found. "
        "Add GEMINI_API_KEY to your "
        "environment or Streamlit "
        "Community Cloud Secrets."
    )

    st.stop()


# SIDEBAR

with st.sidebar:

    st.markdown(
        "## System Overview"
    )

    st.caption(
        "A portfolio RAG application "
        "for MLS roster-rule research."
    )

    st.markdown(
        "### Knowledge Base"
    )

    st.write(
        f"**{len(system['chunks'])}** "
        "indexed chunks"
    )

    st.write(
        "Official MLS roster rules"
    )

    st.write(
        "MLS-MLSPA CBA Article 29"
    )

    st.markdown(
        "### Retrieval Stack"
    )

    st.write(
        "Semantic search — FAISS"
    )

    st.write(
        "Lexical search — TF-IDF"
    )

    st.write(
        "Fusion — Reciprocal Rank Fusion"
    )

    st.write(
        f"Final evidence — "
        f"Top {FINAL_TOP_K}"
    )

    st.markdown(
        "### Evaluation"
    )

    st.write(
        "**Rank 1:** 7 / 8"
    )

    st.write(
        "**Top 3:** 8 / 8"
    )

    st.write(
        "**MRR:** 0.938"
    )

    st.caption(
        "Evaluation results are from "
        "a small 8-question retrieval "
        "benchmark and should not be "
        "interpreted as a broad "
        "system-wide accuracy estimate."
    )

    st.markdown(
        "### Generator"
    )

    st.write(
        f"`{GEMINI_MODEL}`"
    )

    st.caption(
        "Answers are generated after "
        "retrieval and are constrained "
        "to the supplied official-source "
        "evidence."
    )


# METRICS

metric_1, metric_2, metric_3, metric_4 = (
    st.columns(
        4
    )
)

with metric_1:

    st.metric(
        "Indexed Chunks",
        len(
            system[
                "chunks"
            ]
        ),
    )


with metric_2:

    st.metric(
        "Official Sources",
        "2",
    )


with metric_3:

    st.metric(
        "Top-3 Benchmark",
        "100%",
    )


with metric_4:

    st.metric(
        "Benchmark MRR",
        "0.938",
    )


# QUICK QUESTIONS

st.html(
    """
    <div style="margin-top: 24px;">

        <div class="section-kicker">
            QUICK START
        </div>

        <div class="section-title">
            Explore common MLS
            roster questions
        </div>

        <div class="section-copy">
            Choose a prompt or write
            your own question below.
        </div>

    </div>
    """
)


quick_1, quick_2, quick_3, quick_4 = (
    st.columns(
        4
    )
)


with quick_1:

    if st.button(
        "Free Agency Eligibility",
        use_container_width=True,
    ):

        st.session_state[
            "question"
        ] = (
            "Who is eligible for MLS "
            "free agency in 2026?"
        )


with quick_2:

    if st.button(
        "U22 Initiative",
        use_container_width=True,
    ):

        st.session_state[
            "question"
        ] = (
            "How many U22 Initiative "
            "roster slots can an MLS "
            "club have?"
        )


with quick_3:

    if st.button(
        "GAM Buy-Down",
        use_container_width=True,
    ):

        st.session_state[
            "question"
        ] = (
            "How can General Allocation "
            "Money be used to buy down "
            "a player's salary budget "
            "charge?"
        )


with quick_4:

    if st.button(
        "Out-of-Contract Player",
        use_container_width=True,
    ):

        st.session_state[
            "question"
        ] = (
            "What happens to an MLS "
            "player who is out of "
            "contract?"
        )


# QUESTION STATE

if (
    "question"
    not in st.session_state
):

    st.session_state[
        "question"
    ] = ""


# QUESTION FORM

with st.form(
    "mls_question_form",
    clear_on_submit=False,
):

    question = st.text_area(
        "Ask an MLS roster question",
        key="question",
        height=105,
        placeholder=(
            "Example: Who is eligible "
            "for MLS free agency "
            "in 2026?"
        ),
    )

    submitted = (
        st.form_submit_button(
            "Ask MLS Rules",
            use_container_width=True,
        )
    )


# PROCESS QUESTION

if submitted:

    clean_question = (
        question.strip()
    )

    if not clean_question:

        st.warning(
            "Enter a question first."
        )

    else:

        try:

            with st.spinner(
                "Searching official "
                "MLS sources..."
            ):

                retrieved_results = (
                    hybrid_search(
                        clean_question,
                        system,
                        top_k=FINAL_TOP_K,
                    )
                )

            if not retrieved_results:

                st.warning(
                    "No relevant source "
                    "passages were retrieved."
                )

            else:

                with st.spinner(
                    "Generating a grounded "
                    "answer..."
                ):

                    answer = (
                        generate_answer(
                            clean_question,
                            retrieved_results,
                            api_key,
                        )
                    )

                st.html(
                    """
                    <div
                        style="
                            margin-top: 26px;
                        "
                    >

                        <div
                            class="
                                section-kicker
                            "
                        >
                            GROUNDED ANSWER
                        </div>

                        <div
                            class="
                                section-title
                            "
                        >
                            MLS Rules Intelligence
                        </div>

                    </div>
                    """
                )

                st.markdown(
                    answer
                )


                # RETRIEVED SOURCES

                st.html(
                    """
                    <div
                        style="
                            margin-top: 24px;
                        "
                    >

                        <div
                            class="
                                section-kicker
                            "
                        >
                            RETRIEVED EVIDENCE
                        </div>

                        <div
                            class="
                                section-title
                            "
                        >
                            Official-source
                            passages
                        </div>

                        <div
                            class="
                                section-copy
                            "
                        >
                            These are the passages
                            retrieved before Gemini
                            generated the answer.
                        </div>

                    </div>
                    """
                )


                for result in (
                    retrieved_results
                ):

                    rank = result[
                        "rank"
                    ]

                    chunk = result[
                        "chunk"
                    ]

                    title = (
                        get_source_title(
                            chunk
                        )
                    )

                    details = (
                        get_source_details(
                            chunk
                        )
                    )

                    text_value = (
                        get_chunk_text(
                            chunk
                        )
                    )

                    url = (
                        get_source_url(
                            chunk
                        )
                    )

                    expander_title = (
                        f"Source {rank} — "
                        f"{title}"
                    )

                    with st.expander(
                        expander_title,
                        expanded=(
                            rank == 1
                        ),
                    ):

                        if details:

                            st.caption(
                                details
                            )

                        st.write(
                            text_value
                        )

                        retrieval_bits = [
                            (
                                "RRF "
                                f"{result['rrf_score']:.5f}"
                            )
                        ]

                        if (
                            result[
                                "vector_rank"
                            ]
                            is not None
                        ):

                            retrieval_bits.append(
                                "Vector rank "
                                f"{result['vector_rank']}"
                            )

                        if (
                            result[
                                "lexical_rank"
                            ]
                            is not None
                        ):

                            retrieval_bits.append(
                                "TF-IDF rank "
                                f"{result['lexical_rank']}"
                            )

                        st.caption(
                            " · ".join(
                                retrieval_bits
                            )
                        )

                        if url:

                            st.link_button(
                                "Open official source",
                                url,
                            )

        except Exception as error:

            st.error(
                "The question could not "
                "be processed."
            )

            st.exception(
                error
            )


# FOOTER

st.divider()

st.caption(
    "This project is an analytical "
    "research tool built from publicly "
    "available MLS roster rules and "
    "MLS-MLSPA CBA material. It is not "
    "legal, contractual, or compliance "
    "advice. Always verify important "
    "roster decisions against the "
    "current official source documents."
)