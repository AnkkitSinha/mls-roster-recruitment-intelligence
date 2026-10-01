from pathlib import Path
import json
import re

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


# LOAD DATA

print(
    "MLS ROSTER INTELLIGENCE - "
    "HYBRID RETRIEVAL TEST"
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


# PREPARE LEXICAL SEARCH TEXT

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

    text = chunk.get(
        "text",
        ""
    )

    # Repeat important metadata so headings
    # receive more lexical weight.

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
        text
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


# BUILD TF-IDF INDEX

print(
    "\nBuilding lexical search index..."
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

print(
    f"Lexical matrix shape: "
    f"{tfidf_matrix.shape}"
)


# SEMANTIC RANKING

def semantic_ranking(
    question
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
        len(chunks)
    )

    ranking = []

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

        ranking.append(
            {
                "index":
                    int(vector_index),

                "rank":
                    rank,

                "score":
                    float(score)
            }
        )

    return ranking


# LEXICAL RANKING

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

    for rank, vector_index in enumerate(
        indices,
        start=1
    ):

        ranking.append(
            {
                "index":
                    int(vector_index),

                "rank":
                    rank,

                "score":
                    float(
                        scores[
                            vector_index
                        ]
                    )
            }
        )

    return ranking


# RECIPROCAL RANK FUSION

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

        if idx not in combined:

            combined[idx] = {
                "rrf_score":
                    0.0,

                "semantic_rank":
                    None,

                "semantic_score":
                    None,

                "lexical_rank":
                    None,

                "lexical_score":
                    None
            }

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
            "semantic_score"
        ] = result[
            "score"
        ]

        combined[
            idx
        ][
            "rrf_score"
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

        if idx not in combined:

            combined[idx] = {
                "rrf_score":
                    0.0,

                "semantic_rank":
                    None,

                "semantic_score":
                    None,

                "lexical_rank":
                    None,

                "lexical_score":
                    None
            }

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
            "lexical_score"
        ] = result[
            "score"
        ]

        combined[
            idx
        ][
            "rrf_score"
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
                "rrf_score"
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
        ]

        results.append(
            {
                "rank":
                    final_rank,

                "rrf_score":
                    retrieval_data[
                        "rrf_score"
                    ],

                "semantic_rank":
                    retrieval_data[
                        "semantic_rank"
                    ],

                "semantic_score":
                    retrieval_data[
                        "semantic_score"
                    ],

                "lexical_rank":
                    retrieval_data[
                        "lexical_rank"
                    ],

                "lexical_score":
                    retrieval_data[
                        "lexical_score"
                    ],

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
                "What happens to an out-of-contract "
                "MLS player who is not eligible "
                "for free agency?"
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
                "What happens when an MLS player's "
                "contract option is not exercised?"
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


# MATCH FUNCTIONS

def section_matches(
    result_section,
    expected_sections
):

    result_lower = (
        result_section.lower()
    )

    return any(
        expected.lower()
        in result_lower
        for expected
        in expected_sections
    )


def is_correct(
    result,
    test
):

    return (
        result[
            "source_document"
        ]
        == test[
            "expected_source"
        ]
        and
        section_matches(
            result[
                "section"
            ],
            test[
                "expected_sections"
            ]
        )
    )


# RUN BENCHMARK

top1_hits = 0
top3_hits = 0

reciprocal_rank_total = 0.0


print(
    "\nRunning hybrid retrieval benchmark..."
)


for test_number, test in enumerate(
    tests,
    start=1
):

    results = hybrid_search(
        test[
            "question"
        ]
    )

    first_correct_rank = None

    for result in results:

        if is_correct(
            result,
            test
        ):

            first_correct_rank = (
                result[
                    "rank"
                ]
            )

            break

    top1_correct = (
        first_correct_rank == 1
    )

    top3_correct = (
        first_correct_rank is not None
        and
        first_correct_rank <= 3
    )

    if top1_correct:

        top1_hits += 1

    if top3_correct:

        top3_hits += 1

    if first_correct_rank:

        reciprocal_rank_total += (
            1
            / first_correct_rank
        )


    print(
        f"\nTest {test_number}"
    )

    print(
        f"Question: "
        f"{test['question']}"
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


    for result in results[
        :3
    ]:

        location = ""

        if result[
            "section_number"
        ]:

            location += (
                f" | "
                f"{result['section_number']}"
            )

        if result[
            "page_numbers"
        ]:

            location += (
                f" | PDF pages "
                f"{result['page_numbers']}"
            )


        print(
            f"{result['rank']}. "
            f"RRF "
            f"{result['rrf_score']:.5f}"
            f" | "
            f"{result['source_document']}"
            f" | "
            f"{result['section']}"
            f"{location}"
        )

        print(
            f"   Semantic rank: "
            f"{result['semantic_rank']}"
            f" | "
            f"Lexical rank: "
            f"{result['lexical_rank']}"
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


# SUMMARY

test_count = len(
    tests
)

hit_rate_1 = (
    top1_hits
    / test_count
)

hit_rate_3 = (
    top3_hits
    / test_count
)

mrr = (
    reciprocal_rank_total
    / test_count
)


print(
    "\nHybrid retrieval benchmark summary"
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
    f"Hit Rate@1: "
    f"{hit_rate_1:.3f}"
)

print(
    f"Hit Rate@3: "
    f"{hit_rate_3:.3f}"
)

print(
    f"MRR: "
    f"{mrr:.3f}"
)

print(
    "\nSTEP 12 COMPLETE"
)