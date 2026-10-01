from pathlib import Path
import json
import os

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


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


# CHECK API KEY

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:

    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Set it in PowerShell before running this script."
    )


# LOAD KNOWLEDGE BASE

print(
    "MLS ROSTER INTELLIGENCE - "
    "GROUNDED RAG"
)

index = faiss.read_index(
    str(INDEX_FILE)
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

print(
    f"\nKnowledge base chunks: "
    f"{len(chunks)}"
)


# LOAD EMBEDDING MODEL

print(
    "\nLoading embedding model..."
)

embedding_model = SentenceTransformer(
    model_name,
    local_files_only=True
)

print(
    "Embedding model loaded."
)


# BUILD LEXICAL INDEX

def build_lexical_text(chunk):

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

    return " ".join(
        parts
    )


lexical_documents = [
    build_lexical_text(
        chunk
    )
    for chunk
    in chunks
]


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

print(
    "Hybrid retrieval index ready."
)


# SEMANTIC SEARCH

def semantic_ranking(
    question
):

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
        len(chunks)
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
                "score": 0.0
            }
        )

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
                "score": 0.0
            }
        )

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

    for rank, (
        idx,
        score_data
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
        ] = rank

        chunk[
            "retrieval_score"
        ] = score_data[
            "score"
        ]

        results.append(
            chunk
        )

    return results


# FORMAT SOURCE NAME

def source_label(
    chunk
):

    if (
        chunk[
            "source_document"
        ]
        == "MLS CBA"
    ):

        pages = chunk.get(
            "page_numbers"
        )

        page_text = ""

        if pages:

            page_text = (
                " | PDF Page"
                + (
                    "s "
                    if len(
                        pages
                    ) > 1
                    else " "
                )
                + ", ".join(
                    str(page)
                    for page
                    in pages
                )
            )

        return (
            f"MLS-MLSPA CBA | "
            f"{chunk.get('section_number')} | "
            f"{chunk['section']}"
            f"{page_text}"
        )


    parent = chunk.get(
        "parent_section"
    )

    if parent:

        return (
            f"2026 MLS Roster Rules | "
            f"{parent} > "
            f"{chunk['section']}"
        )

    return (
        f"2026 MLS Roster Rules | "
        f"{chunk['section']}"
    )


# BUILD RAG CONTEXT

def build_context(
    results
):

    context_parts = []

    for source_number, chunk in enumerate(
        results,
        start=1
    ):

        label = source_label(
            chunk
        )

        context_parts.append(
            f"[Source {source_number}]\n"
            f"{label}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# CREATE GEMINI CLIENT

client = genai.Client(
    api_key=api_key
)


# GENERATE GROUNDED ANSWER

def generate_answer(
    question,
    results
):

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
3. If the retrieved evidence is insufficient, clearly say
   that the available sources do not contain enough information.
4. Prefer the most detailed authoritative passage.
5. If one MLS Roster Rules passage merely refers the reader
   to the Collective Bargaining Agreement and a detailed CBA
   passage is also provided, use the detailed CBA passage.
6. When a rule differs by year, clearly state the applicable
   year.
7. Cite factual claims using [Source 1], [Source 2], etc.
8. Keep the answer concise but complete.
9. Do not describe a similarity or retrieval score as confidence.
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


# DISPLAY SOURCES

def display_sources(
    results
):

    print(
        "\nSources used by the RAG system:"
    )

    for source_number, chunk in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[Source {source_number}] "
            f"{source_label(chunk)}"
        )

        print(
            f"Retrieval rank: "
            f"{chunk['retrieval_rank']}"
        )


# INTERACTIVE RAG

print(
    f"\nGenerator: "
    f"{GEMINI_MODEL}"
)

print(
    "\nAsk an MLS roster question."
)

print(
    "Type 'exit' to stop."
)


while True:

    question = input(
        "\nYour question: "
    ).strip()

    if question.lower() in [
        "exit",
        "quit"
    ]:

        break

    if not question:

        continue


    print(
        "\nRetrieving evidence..."
    )

    results = hybrid_search(
        question
    )


    print(
        "Generating grounded answer..."
    )

    try:

        answer = generate_answer(
            question,
            results
        )

    except Exception as error:

        print(
            f"\nGemini API error: "
            f"{error}"
        )

        continue


    print(
        "\nAnswer:"
    )

    print(
        answer
    )


    display_sources(
        results
    )


print(
    "\nSTEP 13 COMPLETE"
)