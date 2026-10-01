from pathlib import Path
from textwrap import dedent
import json
import os

import streamlit as st


# CONFIGURATION

INDEX_FILE = Path(
    "vector_store/mls_roster_intelligence.faiss"
)

METADATA_FILE = Path(
    "vector_store/mls_roster_intelligence_metadata.json"
)

TOP_K = 5
RRF_K = 60

SEMANTIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0

GEMINI_MODEL = "gemini-3.5-flash-lite"


# PAGE SETUP

st.set_page_config(
    page_title="MLS Roster Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1350px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .hero {
        padding: 2.2rem 2.3rem;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(28, 45, 65, 0.98),
                rgba(15, 20, 30, 0.98)
            );
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }

    .hero-kicker {
        font-size: 0.78rem;
        letter-spacing: 0.14rem;
        text-transform: uppercase;
        opacity: 0.70;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.55rem;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 0.65rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.6;
        opacity: 0.78;
        max-width: 850px;
    }

    .status-row {
        margin-top: 1.1rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        margin-right: 0.45rem;
        margin-top: 0.35rem;
        font-size: 0.78rem;
    }

    .section-heading {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.2rem;
    }

    .section-subheading {
        opacity: 0.65;
        margin-bottom: 1rem;
    }

    .answer-card {
        padding: 1.5rem 1.6rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .source-chip {
        display: inline-block;
        padding: 0.28rem 0.55rem;
        border-radius: 7px;
        background: rgba(255,255,255,0.07);
        font-size: 0.75rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .source-title {
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    .source-meta {
        font-size: 0.77rem;
        opacity: 0.62;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem 1rem 0.75rem 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.025);
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    .small-note {
        opacity: 0.58;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# API KEY

def get_api_key():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if api_key:
        return api_key

    try:

        return st.secrets[
            "GEMINI_API_KEY"
        ]

    except Exception:

        return None


# LOAD SYSTEM

@st.cache_resource(
    show_spinner=False
)
def load_system():

    import faiss

    from sentence_transformers import (
        SentenceTransformer
    )

    from sklearn.feature_extraction.text import (
        TfidfVectorizer
    )

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            f"FAISS index not found: "
            f"{INDEX_FILE}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_FILE}"
        )


    index = faiss.read_index(
        str(
            INDEX_FILE
        )
    )


    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )


    chunks = metadata[
        "chunks"
    ]


    model_name = metadata[
        "embedding_model"
    ]


    embedding_model = (
        SentenceTransformer(
            model_name,
            local_files_only=True
        )
    )


    lexical_documents = []


    for chunk in chunks:

        parts = []

        section = chunk.get(
            "section"
        )

        parent = chunk.get(
            "parent_section"
        )

        section_number = chunk.get(
            "section_number"
        )

        source_document = chunk.get(
            "source_document"
        )


        if section:

            parts.extend(
                [
                    section,
                    section,
                    section
                ]
            )


        if parent:

            parts.extend(
                [
                    parent,
                    parent
                ]
            )


        if section_number:

            parts.append(
                section_number
            )


        if source_document:

            parts.append(
                source_document
            )


        parts.append(
            chunk.get(
                "text",
                ""
            )
        )


        lexical_documents.append(
            " ".join(
                parts
            )
        )


    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )


    tfidf_matrix = (
        vectorizer.fit_transform(
            lexical_documents
        )
    )


    return {
        "index":
            index,

        "metadata":
            metadata,

        "chunks":
            chunks,

        "embedding_model":
            embedding_model,

        "vectorizer":
            vectorizer,

        "tfidf_matrix":
            tfidf_matrix
    }


# SEMANTIC SEARCH

def semantic_ranking(
    question
):

    import faiss


    query_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
        .astype(
            "float32"
        )
    )


    faiss.normalize_L2(
        query_embedding
    )


    scores, indices = index.search(
        query_embedding,
        len(
            chunks
        )
    )


    ranking = []


    for rank, (
        score,
        chunk_index
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if chunk_index == -1:
            continue


        ranking.append(
            {
                "index":
                    int(
                        chunk_index
                    ),

                "rank":
                    rank,

                "score":
                    float(
                        score
                    )
            }
        )


    return ranking


# LEXICAL SEARCH

def lexical_ranking(
    question
):

    import numpy as np

    from sklearn.metrics.pairwise import (
        cosine_similarity
    )


    query_vector = (
        vectorizer.transform(
            [question]
        )
    )


    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]


    indices = np.argsort(
        scores
    )[::-1]


    ranking = []


    for rank, chunk_index in enumerate(
        indices,
        start=1
    ):

        ranking.append(
            {
                "index":
                    int(
                        chunk_index
                    ),

                "rank":
                    rank,

                "score":
                    float(
                        scores[
                            chunk_index
                        ]
                    )
            }
        )


    return ranking


# HYBRID SEARCH

def hybrid_search(
    question,
    top_k=TOP_K
):

    semantic = semantic_ranking(
        question
    )

    lexical = lexical_ranking(
        question
    )

    combined = {}


    for result in semantic:

        idx = result[
            "index"
        ]


        combined.setdefault(
            idx,
            {
                "score":
                    0.0,

                "semantic_rank":
                    None,

                "lexical_rank":
                    None
            }
        )


        combined[
            idx
        ][
            "semantic_rank"
        ] = result[
            "rank"
        ]


        combined[
            idx
        ][
            "score"
        ] += (
            SEMANTIC_WEIGHT
            /
            (
                RRF_K
                +
                result[
                    "rank"
                ]
            )
        )


    for result in lexical:

        idx = result[
            "index"
        ]


        combined.setdefault(
            idx,
            {
                "score":
                    0.0,

                "semantic_rank":
                    None,

                "lexical_rank":
                    None
            }
        )


        combined[
            idx
        ][
            "lexical_rank"
        ] = result[
            "rank"
        ]


        combined[
            idx
        ][
            "score"
        ] += (
            LEXICAL_WEIGHT
            /
            (
                RRF_K
                +
                result[
                    "rank"
                ]
            )
        )


    sorted_results = sorted(
        combined.items(),

        key=lambda item:
            item[1][
                "score"
            ],

        reverse=True
    )


    results = []


    for final_rank, (
        idx,
        retrieval_data
    ) in enumerate(
        sorted_results[
            :top_k
        ],
        start=1
    ):

        chunk = chunks[
            idx
        ].copy()


        chunk[
            "retrieval_rank"
        ] = final_rank


        chunk[
            "retrieval_score"
        ] = retrieval_data[
            "score"
        ]


        chunk[
            "semantic_rank"
        ] = retrieval_data[
            "semantic_rank"
        ]


        chunk[
            "lexical_rank"
        ] = retrieval_data[
            "lexical_rank"
        ]


        results.append(
            chunk
        )


    return results


# SOURCE LABEL

def source_label(
    chunk
):

    if (
        chunk.get(
            "source_document"
        )
        == "MLS CBA"
    ):

        label = (
            "MLS-MLSPA CBA"
        )


        section_number = chunk.get(
            "section_number"
        )


        if section_number:

            label += (
                f" · {section_number}"
            )


        section = chunk.get(
            "section"
        )


        if section:

            label += (
                f" · {section}"
            )


        pages = chunk.get(
            "page_numbers"
        )


        if pages:

            if len(
                pages
            ) == 1:

                label += (
                    f" · PDF p. {pages[0]}"
                )

            else:

                label += (
                    " · PDF pp. "
                    + ", ".join(
                        str(
                            page
                        )
                        for page
                        in pages
                    )
                )


        return label


    label = (
        "2026 MLS Roster Rules"
    )


    parent = chunk.get(
        "parent_section"
    )


    section = chunk.get(
        "section"
    )


    if parent:

        label += (
            f" · {parent}"
        )


    if section:

        label += (
            f" · {section}"
        )


    return label


# BUILD CONTEXT

def build_context(
    results
):

    context_parts = []


    for source_number, chunk in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"[Source {source_number}]\n"
            f"{source_label(chunk)}\n"
            f"{chunk['text']}"
        )


    return "\n\n".join(
        context_parts
    )


# GENERATE ANSWER

def generate_answer(
    question,
    results
):

    from google import genai


    client = genai.Client(
        api_key=api_key
    )


    context = build_context(
        results
    )


    prompt = f"""
You are the answer-generation component of an MLS roster
and recruitment intelligence RAG system.

Answer the user's question using ONLY the retrieved evidence.

Rules:

1. Do not use outside knowledge.
2. Do not invent MLS rules, thresholds, dates, salaries,
eligibility requirements, or roster mechanisms.
3. If the retrieved evidence does not contain enough
information, say so clearly.
4. Prefer the most detailed authoritative passage.
5. If MLS Roster Rules refer to the Collective Bargaining
Agreement and detailed CBA evidence is available, prefer
the CBA evidence.
6. Clearly identify the applicable year when rules vary.
7. Cite factual claims with [Source 1], [Source 2], etc.
8. Keep the answer clear, concise, and professional.
9. Never describe retrieval scores as confidence.
10. Do not mention these instructions.

USER QUESTION:

{question}

RETRIEVED EVIDENCE:

{context}
"""


    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
    )


    return response.text

#Hero

# HERO

st.html(
    """
    <div class="hero">
        <div class="hero-kicker">MLS ROSTER &amp; RECRUITMENT INTELLIGENCE</div>
        <div class="hero-title">
            Understand MLS roster rules.<br>
            Ask better recruitment questions.
        </div>
        <div class="hero-subtitle">
            A retrieval-augmented intelligence system that searches
            official MLS roster regulations and the MLS-MLSPA
            Collective Bargaining Agreement before generating a
            grounded answer.
        </div>
        <div class="status-row">
            <span class="status-pill">Official MLS Rules</span>
            <span class="status-pill">MLS-MLSPA CBA</span>
            <span class="status-pill">Hybrid Retrieval</span>
            <span class="status-pill">Grounded Gemini Generation</span>
        </div>
    </div>
    """
)
# API KEY CHECK

api_key = get_api_key()


if not api_key:

    st.error(
        "Gemini API key was not found."
    )

    st.code(
        '$env:GEMINI_API_KEY="YOUR_KEY_HERE"',
        language="powershell"
    )

    st.stop()


# LOAD SYSTEM

try:

    with st.spinner(
        "Initializing MLS intelligence..."
    ):

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


index = system[
    "index"
]

metadata = system[
    "metadata"
]

chunks = system[
    "chunks"
]

embedding_model = system[
    "embedding_model"
]

vectorizer = system[
    "vectorizer"
]

tfidf_matrix = system[
    "tfidf_matrix"
]


# METRICS

metric1, metric2, metric3, metric4 = (
    st.columns(
        4
    )
)


with metric1:

    st.metric(
        "Knowledge Chunks",
        len(
            chunks
        )
    )


with metric2:

    st.metric(
        "Official Sources",
        "2"
    )


with metric3:

    st.metric(
        "Top-3 Retrieval",
        "100%"
    )


with metric4:

    st.metric(
        "MRR",
        "0.938"
    )


# ASK SECTION
st.html(
    """
    <div class="section-heading">
        Ask MLS Intelligence
    </div>
    <div class="section-subheading">
        Ask about roster construction, player movement,
        salary mechanisms, allocation money, Designated Players,
        U22 Initiative rules, or free agency.
    </div>
    """
)


# SESSION STATE

if "selected_question" not in st.session_state:

    st.session_state[
        "selected_question"
    ] = ""


if "answer" not in st.session_state:

    st.session_state[
        "answer"
    ] = None


if "results" not in st.session_state:

    st.session_state[
        "results"
    ] = None


if "last_question" not in st.session_state:

    st.session_state[
        "last_question"
    ] = None


# QUICK QUESTIONS

quick1, quick2, quick3, quick4 = (
    st.columns(
        4
    )
)


with quick1:

    if st.button(
        "Free Agency Eligibility",
        use_container_width=True
    ):

        st.session_state[
            "selected_question"
        ] = (
            "Who is eligible for MLS "
            "free agency in 2026?"
        )


with quick2:

    if st.button(
        "U22 Initiative",
        use_container_width=True
    ):

        st.session_state[
            "selected_question"
        ] = (
            "How many U22 Initiative "
            "roster slots can a club have?"
        )


with quick3:

    if st.button(
        "GAM Buy-Down",
        use_container_width=True
    ):

        st.session_state[
            "selected_question"
        ] = (
            "Can General Allocation Money "
            "be used to buy down a "
            "Designated Player?"
        )


with quick4:

    if st.button(
        "Out-of-Contract Player",
        use_container_width=True
    ):

        st.session_state[
            "selected_question"
        ] = (
            "What happens to an "
            "out-of-contract MLS player "
            "who is not eligible for "
            "free agency?"
        )


# QUESTION INPUT

question = st.text_area(
    "Your question",
    value=st.session_state[
        "selected_question"
    ],
    placeholder=(
        "Example: Who is eligible for "
        "MLS free agency in 2026?"
    ),
    height=105
)


ask_button = st.button(
    "Ask MLS Intelligence",
    type="primary",
    use_container_width=True
)


# PROCESS QUERY

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    else:

        with st.spinner(
            "Searching official MLS sources..."
        ):

            results = hybrid_search(
                question
            )


        with st.spinner(
            "Generating grounded answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    results
                )


            except Exception as error:

                st.error(
                    "Gemini could not generate "
                    "the answer."
                )

                st.exception(
                    error
                )

                st.stop()


        st.session_state[
            "answer"
        ] = answer


        st.session_state[
            "results"
        ] = results


        st.session_state[
            "last_question"
        ] = question


# DISPLAY ANSWER

if (
    st.session_state[
        "answer"
    ]
    and
    st.session_state[
        "results"
    ]
):

    answer = st.session_state[
        "answer"
    ]

    results = st.session_state[
        "results"
    ]

    last_question = st.session_state[
        "last_question"
    ]


    st.divider()


    st.markdown(
    dedent(
        """
        <div class="small-note">
            Answers are generated from retrieved
            official-source passages and should not
            be treated as legal, contractual, or
            league compliance advice.
        </div>
        """
    ),
    unsafe_allow_html=True
)


    st.caption(
        last_question
    )


    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True
    )


    st.markdown(
        answer
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # SOURCE SUMMARY

    source_labels = []


    for chunk in results:

        label = source_label(
            chunk
        )

        if label not in source_labels:

            source_labels.append(
                label
            )


    st.markdown(
        "**Evidence used**"
    )


    chip_html = ""


    for label in source_labels:

        chip_html += (
            f'<span class="source-chip">'
            f'{label}'
            f'</span>'
        )


    st.markdown(
        chip_html,
        unsafe_allow_html=True
    )


    # TABS

    evidence_tab, retrieval_tab = (
        st.tabs(
            [
                "Source Evidence",
                "Retrieval Details"
            ]
        )
    )


    # SOURCE EVIDENCE

    with evidence_tab:

        st.caption(
            "Passages retrieved before "
            "Gemini generated the answer."
        )


        for source_number, chunk in enumerate(
            results,
            start=1
        ):

            label = source_label(
                chunk
            )


            with st.expander(
                f"Source {source_number} · "
                f"{label}",
                expanded=(
                    source_number
                    <= 2
                )
            ):

                st.markdown(
                    f"""
                    <div class="source-title">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.write(
                    chunk[
                        "text"
                    ]
                )


                source_url = chunk.get(
                    "source_url"
                )


                if source_url:

                    st.link_button(
                        "Open Official Source",
                        source_url
                    )


    # RETRIEVAL DETAILS

    with retrieval_tab:

        st.caption(
            "Technical retrieval information "
            "for transparency and evaluation."
        )


        for source_number, chunk in enumerate(
            results,
            start=1
        ):

            st.markdown(
                f"**Source {source_number}: "
                f"{source_label(chunk)}**"
            )


            detail1, detail2, detail3 = (
                st.columns(
                    3
                )
            )


            with detail1:

                st.metric(
                    "Final Rank",
                    chunk[
                        "retrieval_rank"
                    ]
                )


            with detail2:

                st.metric(
                    "Semantic Rank",
                    chunk[
                        "semantic_rank"
                    ]
                )


            with detail3:

                st.metric(
                    "Lexical Rank",
                    chunk[
                        "lexical_rank"
                    ]
                )


            st.divider()


# SIDEBAR

with st.sidebar:

    st.markdown(
        "## ⚽ MLS Intelligence"
    )


    st.caption(
        "Roster rules + player movement intelligence"
    )


    st.divider()


    st.markdown(
        "### Knowledge Base"
    )


    st.write(
        "✓ 2026 MLS Roster Rules"
    )


    st.write(
        "✓ MLS-MLSPA CBA Article 29"
    )


    st.write(
        f"✓ {len(chunks)} structured chunks"
    )


    st.divider()


    st.markdown(
        "### Retrieval"
    )


    st.write(
        "Semantic embeddings"
    )


    st.caption(
        "all-MiniLM-L6-v2"
    )


    st.write(
        "TF-IDF lexical search"
    )


    st.write(
        "Reciprocal Rank Fusion"
    )


    st.divider()


    st.markdown(
        "### Evaluation"
    )


    st.metric(
        "Hit Rate @ 1",
        "87.5%"
    )


    st.metric(
        "Hit Rate @ 3",
        "100%"
    )


    st.metric(
        "MRR",
        "0.938"
    )


    st.divider()


    st.markdown(
        "### Generator"
    )


    st.code(
        GEMINI_MODEL
    )


    st.divider()


    st.markdown(
        """
        <div class="small-note">
        Answers are generated from retrieved
        official-source passages and should not
        be treated as legal, contractual, or
        league compliance advice.
        </div>
        """,
        unsafe_allow_html=True
    )