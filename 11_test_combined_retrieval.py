from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# CONFIGURATION

INDEX_FILE = Path(
    "vector_store/mls_roster_intelligence.faiss"
)

METADATA_FILE = Path(
    "vector_store/mls_roster_intelligence_metadata.json"
)

TOP_K = 5


# LOAD INDEX

print(
    "MLS ROSTER INTELLIGENCE - "
    "COMBINED RETRIEVAL TEST"
)

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

chunks = metadata[
    "chunks"
]

model_name = metadata[
    "embedding_model"
]

print(
    f"\nVectors loaded: "
    f"{index.ntotal}"
)

print(
    f"Chunks loaded: "
    f"{len(chunks)}"
)


# LOAD MODEL

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

def search(
    question,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(
        "float32"
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

        results.append(
            {
                "rank":
                    rank,

                "score":
                    float(score),

                "source_document":
                    chunk[
                        "source_document"
                    ],

                "section":
                    chunk[
                        "section"
                    ],

                "section_number":
                    chunk.get(
                        "section_number"
                    ),

                "parent_section":
                    chunk.get(
                        "parent_section"
                    ),

                "page_numbers":
                    chunk.get(
                        "page_numbers"
                    ),

                "text":
                    chunk[
                        "text"
                    ]
            }
        )

    return results


# TEST QUESTIONS

tests = [
    {
        "question":
            (
                "How many U22 Initiative roster "
                "slots can an MLS club have?"
            ),

        "expected_source":
            "MLS Roster Rules",

        "expected_sections":
            [
                "U22 Initiative Roster Slots"
            ]
    },

    {
        "question":
            (
                "What is the maximum salary "
                "budget charge for a player "
                "in 2026?"
            ),

        "expected_source":
            "MLS Roster Rules",

        "expected_sections":
            [
                "Senior Roster",
                "2026 Salary Budget Information",
                "Salary Parameters"
            ]
    },

    {
        "question":
            (
                "Can General Allocation Money "
                "be used to buy down a player's "
                "salary budget charge?"
            ),

        "expected_source":
            "MLS Roster Rules",

        "expected_sections":
            [
                "Buy-Down",
                "Use against a Salary Budget Charge"
            ]
    },

    {
        "question":
            (
                "Who is eligible for MLS "
                "free agency in 2026?"
            ),

        "expected_source":
            "MLS CBA",

        "expected_sections":
            [
                "Free Agency"
            ]
    },

    {
        "question":
            (
                "What happens to an "
                "out-of-contract MLS player "
                "who is not eligible for "
                "free agency?"
            ),

        "expected_source":
            "MLS CBA",

        "expected_sections":
            [
                "Re-Entry Draft – "
                "Out-of-Contract Players"
            ]
    },

    {
        "question":
            (
                "What happens when an MLS "
                "player's contract option "
                "is not exercised?"
            ),

        "expected_source":
            "MLS CBA",

        "expected_sections":
            [
                "Re-Entry Draft – Players "
                "Whose Options Are Not Exercised"
            ]
    },

    {
        "question":
            (
                "When does the Re-Entry Draft "
                "and MLS free agency take place?"
            ),

        "expected_source":
            "MLS CBA",

        "expected_sections":
            [
                "Mechanics and Timing"
            ]
    },

    {
        "question":
            (
                "Does a team receive allocation "
                "money when it has a net loss "
                "of players through free agency?"
            ),

        "expected_source":
            "MLS CBA",

        "expected_sections":
            [
                "Free Agency: Miscellaneous"
            ]
    }
]


# EVALUATION FUNCTION

def section_matches(
    result_section,
    expected_sections
):

    result_lower = (
        result_section.lower()
    )

    for expected in expected_sections:

        if (
            expected.lower()
            in result_lower
        ):
            return True

    return False


def is_correct_result(
    result,
    test
):

    source_correct = (
        result[
            "source_document"
        ]
        == test[
            "expected_source"
        ]
    )

    section_correct = (
        section_matches(
            result[
                "section"
            ],
            test[
                "expected_sections"
            ]
        )
    )

    return (
        source_correct
        and section_correct
    )


# RUN BENCHMARK

top1_hits = 0
top3_hits = 0

print(
    "\nRunning retrieval benchmark..."
)

for test_number, test in enumerate(
    tests,
    start=1
):

    question = test[
        "question"
    ]

    results = search(
        question
    )

    top1_correct = (
        len(results) > 0
        and
        is_correct_result(
            results[0],
            test
        )
    )

    top3_correct = any(
        is_correct_result(
            result,
            test
        )
        for result
        in results[:3]
    )

    if top1_correct:
        top1_hits += 1

    if top3_correct:
        top3_hits += 1

    print(
        f"\nTest {test_number}"
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Expected source: "
        f"{test['expected_source']}"
    )

    print(
        "Expected section: "
        + " OR ".join(
            test[
                "expected_sections"
            ]
        )
    )

    print(
        f"Hit@1: "
        f"{'YES' if top1_correct else 'NO'}"
    )

    print(
        f"Hit@3: "
        f"{'YES' if top3_correct else 'NO'}"
    )

    print(
        "\nTop 3 retrieved:"
    )

    for result in results[:3]:

        source = result[
            "source_document"
        ]

        section = result[
            "section"
        ]

        section_number = result.get(
            "section_number"
        )

        pages = result.get(
            "page_numbers"
        )

        location = ""

        if section_number:

            location += (
                f" | {section_number}"
            )

        if pages:

            location += (
                f" | PDF pages {pages}"
            )

        print(
            f"{result['rank']}. "
            f"{result['score']:.4f} | "
            f"{source} | "
            f"{section}"
            f"{location}"
        )

    if results:

        print(
            "\nTop result preview:"
        )

        print(
            results[0][
                "text"
            ][:350]
        )

        print("...")


# BENCHMARK SUMMARY

test_count = len(
    tests
)

recall_at_1 = (
    top1_hits
    / test_count
)

recall_at_3 = (
    top3_hits
    / test_count
)

print(
    "\nRetrieval benchmark summary"
)

print(
    f"Questions tested: "
    f"{test_count}"
)

print(
    f"Correct at Rank 1: "
    f"{top1_hits}/{test_count}"
)

print(
    f"Correct within Top 3: "
    f"{top3_hits}/{test_count}"
)

print(
    f"Recall@1: "
    f"{recall_at_1:.3f}"
)

print(
    f"Recall@3: "
    f"{recall_at_3:.3f}"
)

print(
    "\nSTEP 11 COMPLETE"
)