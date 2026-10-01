from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# CONFIGURATION

VECTOR_DIR = Path("vector_store")

INDEX_FILE = VECTOR_DIR / "mls_roster_rules.faiss"

METADATA_FILE = VECTOR_DIR / "mls_roster_rules_metadata.json"

TOP_K = 5


# LOAD VECTOR INDEX
print("MLS ROSTER INTELLIGENCE - RETRIEVAL TEST")

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {INDEX_FILE}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {METADATA_FILE}"
    )

index = faiss.read_index(
    str(INDEX_FILE)
)

metadata = json.loads(
    METADATA_FILE.read_text(
        encoding="utf-8"
    )
)

chunks = metadata["chunks"]

model_name = metadata[
    "embedding_model"
]

print(
    f"\nVectors loaded : "
    f"{index.ntotal}"
)

print(
    f"Chunks loaded  : "
    f"{len(chunks)}"
)

print(
    f"Model          : "
    f"{model_name}"
)


# LOAD EMBEDDING MODEL

print(
    "\nLoading embedding model..."
)

model = SentenceTransformer(
    model_name,
    local_files_only=True
)

print(
    "Embedding model loaded."
)


# SEARCH FUNCTION

def search_documents(
    question,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = (
        query_embedding.astype(
            "float32"
        )
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (
        score,
        vector_index
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if vector_index == -1:
            continue

        chunk = chunks[
            vector_index
        ]

        result = {
            "rank": rank,
            "score": float(
                score
            ),
            "chunk_id": chunk[
                "chunk_id"
            ],
            "parent_section": chunk[
                "parent_section"
            ],
            "section": chunk[
                "section"
            ],
            "chunk_index": chunk[
                "chunk_index"
            ],
            "section_chunk_count": chunk[
                "section_chunk_count"
            ],
            "text": chunk[
                "text"
            ],
            "source_url": chunk[
                "source_url"
            ]
        }

        results.append(
            result
        )

    return results


# DISPLAY RESULTS

def print_results(
    question,
    results
):

    print(
        "\n"
    )

    print(
        f"QUESTION: {question}"
    )

    for result in results:

        print(
            f"\nRank: "
            f"{result['rank']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Section: "
            f"{result['section']}"
        )

        if result[
            "parent_section"
        ]:

            print(
                f"Parent: "
                f"{result['parent_section']}"
            )

        print(
            f"Chunk: "
            f"{result['chunk_index']}/"
            f"{result['section_chunk_count']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print("\nRetrieved text:")

        print(
            result["text"][:700]
        )

        print(
            "\n"
        )


# TEST QUESTIONS

test_questions = [

    (
        "How many U22 Initiative roster "
        "slots can an MLS club have?"
    ),

    (
        "What is the maximum salary "
        "budget charge for a player?"
    ),

    (
        "Can General Allocation Money "
        "be used to buy down a "
        "Designated Player?"
    ),

    (
        "How does free agency work "
        "in MLS?"
    )

]


# RUN TEST QUESTIONS

for question in test_questions:

    results = search_documents(
        question
    )

    print_results(
        question,
        results
    )


# INTERACTIVE SEARCH

print(
    "\n"
)

print(
    "INTERACTIVE SEARCH"
)


print(
    "\nEnter your own MLS roster "
    "question."
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

    results = search_documents(
        question
    )

    print_results(
        question,
        results
    )


print(
    "\nSTEP 6 COMPLETE"
)