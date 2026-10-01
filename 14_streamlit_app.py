from pathlib import Path
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


# PAGE HEADER

st.title(
    "⚽ MLS Roster & Recruitment Intelligence"
)

st.caption(
    "AI-powered MLS roster intelligence grounded in "
    "official MLS roster rules and the MLS-MLSPA "
    "Collective Bargaining Agreement."
)

st.info(
    "Loading the MLS intelligence system. "
    "The first startup may take a little longer."
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


# LOAD API KEY

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


# LOAD KNOWLEDGE BASE

try:

    with st.spinner(
        "Loading FAISS index and embedding model..."
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


# SYSTEM STATUS

st.success(
    "MLS intelligence system loaded successfully."
)


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
        "Knowledge Sources",
        2
    )


with metric3:

    st.metric(
        "Retrieval",
        "Hybrid"
    )


with metric4:

    st.metric(
        "Generator",
        "Gemini"
    )


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


    scores, indices = (
        index.search(
            query_embedding,
            len(
                chunks
            )
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

    semantic = (
        semantic_ranking(
            question
        )
    )


    lexical = (
        lexical_ranking(
            question
        )
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

    source_document = chunk.get(
        "source_document"
    )


    if source_document == "MLS CBA":

        section_number = (
            chunk.get(
                "section_number"
            )
        )


        section = chunk.get(
            "section",
            ""
        )


        pages = chunk.get(
            "page_numbers"
        )


        label = (
            "MLS-MLSPA CBA"
        )


        if section_number:

            label += (
                f" | {section_number}"
            )


        if section:

            label += (
                f" | {section}"
            )


        if pages:

            if len(
                pages
            ) == 1:

                label += (
                    f" | PDF Page "
                    f"{pages[0]}"
                )

            else:

                label += (
                    " | PDF Pages "
                    + ", ".join(
                        str(
                            page
                        )
                        for page
                        in pages
                    )
                )


        return label


    section = chunk.get(
        "section",
        ""
    )


    parent = chunk.get(
        "parent_section"
    )


    if parent:

        return (
            "2026 MLS Roster Rules"
            f" | {parent}"
            f" > {section}"
        )


    return (
        "2026 MLS Roster Rules"
        f" | {section}"
    )


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

Answer the user's question using ONLY the retrieved evidence
provided below.

Rules:

1. Do not use outside knowledge.

2. Do not invent MLS rules, thresholds, dates, salaries,
eligibility requirements, or roster mechanisms.

3. If the retrieved evidence does not contain enough
information, clearly say that the available sources do not
contain enough information.

4. Prefer the most detailed authoritative passage.

5. If the MLS Roster Rules refer to the Collective Bargaining
Agreement and a detailed CBA passage is also available,
use the detailed CBA passage.

6. If a rule differs by year, clearly identify the
applicable year.

7. Cite factual claims using [Source 1], [Source 2], etc.

8. Keep the answer concise but complete.

9. Do not describe retrieval scores as confidence.

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


# QUESTION SECTION

st.divider()

st.subheader(
    "Ask MLS Rules"
)


st.write(
    "Ask about roster construction, salary rules, "
    "U22 Initiative slots, Designated Players, "
    "allocation money, free agency, or player movement."
)


# EXAMPLE QUESTIONS

st.caption(
    "Example questions"
)


example_col1, example_col2 = (
    st.columns(
        2
    )
)


with example_col1:

    st.write(
        "• Who is eligible for MLS free agency in 2026?"
    )

    st.write(
        "• How many U22 Initiative slots can a club have?"
    )


with example_col2:

    st.write(
        "• Can GAM be used to buy down a Designated Player?"
    )

    st.write(
        "• What happens to an out-of-contract player?"
    )


# SESSION STATE

if "question_input" not in st.session_state:

    st.session_state[
        "question_input"
    ] = ""


# QUESTION FORM

with st.form(
    "question_form"
):

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: Who is eligible for "
            "MLS free agency in 2026?"
        ),
        height=100
    )


    ask_button = st.form_submit_button(
        "Ask MLS Intelligence",
        type="primary",
        use_container_width=True
    )


# PROCESS QUESTION

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    else:

        st.divider()


        st.subheader(
            "Answer"
        )


        with st.spinner(
            "Searching MLS rules and CBA..."
        ):

            results = hybrid_search(
                question
            )


        with st.spinner(
            "Generating grounded answer with Gemini..."
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


        st.markdown(
            answer
        )


        st.divider()


        st.subheader(
            "Retrieved Sources"
        )


        st.caption(
            "These are the passages retrieved "
            "before Gemini generated the answer."
        )


        for source_number, chunk in enumerate(
            results,
            start=1
        ):

            label = source_label(
                chunk
            )


            with st.expander(
                f"Source {source_number} — "
                f"{label}",
                expanded=(
                    source_number
                    == 1
                )
            ):

                st.write(
                    chunk[
                        "text"
                    ]
                )


                col1, col2, col3 = (
                    st.columns(
                        3
                    )
                )


                with col1:

                    st.caption(
                        f"Final Rank: "
                        f"{chunk['retrieval_rank']}"
                    )


                with col2:

                    st.caption(
                        f"Semantic Rank: "
                        f"{chunk['semantic_rank']}"
                    )


                with col3:

                    st.caption(
                        f"Lexical Rank: "
                        f"{chunk['lexical_rank']}"
                    )


                source_url = chunk.get(
                    "source_url"
                )


                if source_url:

                    st.markdown(
                        f"[Open Official Source]"
                        f"({source_url})"
                    )


# SIDEBAR

with st.sidebar:

    st.header(
        "MLS Intelligence"
    )


    st.write(
        "A retrieval-augmented generation "
        "system for MLS roster and player "
        "movement rules."
    )


    st.divider()


    st.subheader(
        "Knowledge Base"
    )


    st.write(
        "✓ 2026 MLS Roster Rules"
    )


    st.write(
        "✓ MLS-MLSPA CBA Article 29"
    )


    st.write(
        f"✓ {len(chunks)} RAG chunks"
    )


    st.divider()


    st.subheader(
        "Architecture"
    )


    st.write(
        "Semantic Search"
    )

    st.write(
        "+"
    )

    st.write(
        "TF-IDF Lexical Search"
    )

    st.write(
        "↓"
    )

    st.write(
        "Reciprocal Rank Fusion"
    )

    st.write(
        "↓"
    )

    st.write(
        "Gemini Grounded Generation"
    )


    st.divider()


    st.subheader(
        "Model"
    )


    st.code(
        GEMINI_MODEL
    )


    st.divider()


    st.caption(
        "The system answers questions using "
        "retrieved official-source passages. "
        "It should not be treated as legal or "
        "contractual advice."
    )