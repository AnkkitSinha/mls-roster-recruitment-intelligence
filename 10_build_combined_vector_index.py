from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# CONFIGURATION

ROSTER_CHUNKS_FILE = Path(
    "data/processed/mls_roster_rules_2026_chunks.json"
)

CBA_CHUNKS_FILE = Path(
    "data/processed/mls_cba_article_29_chunks.json"
)

VECTOR_DIR = Path(
    "vector_store"
)

INDEX_FILE = (
    VECTOR_DIR
    / "mls_roster_intelligence.faiss"
)

METADATA_FILE = (
    VECTOR_DIR
    / "mls_roster_intelligence_metadata.json"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# LOAD DATA

print(
    "MLS ROSTER INTELLIGENCE - "
    "COMBINED VECTOR INDEX"
)

if not ROSTER_CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: "
        f"{ROSTER_CHUNKS_FILE}"
    )

if not CBA_CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: "
        f"{CBA_CHUNKS_FILE}"
    )

VECTOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


roster_chunks = json.loads(
    ROSTER_CHUNKS_FILE.read_text(
        encoding="utf-8"
    )
)

cba_chunks = json.loads(
    CBA_CHUNKS_FILE.read_text(
        encoding="utf-8"
    )
)


print(
    f"\nMLS roster-rule chunks: "
    f"{len(roster_chunks)}"
)

print(
    f"CBA Article 29 chunks: "
    f"{len(cba_chunks)}"
)


# NORMALIZE ROSTER RULE CHUNKS

combined_chunks = []

global_chunk_id = 1


for chunk in roster_chunks:

    record = {
        "global_chunk_id":
            global_chunk_id,

        "source_document":
            "MLS Roster Rules",

        "document":
            chunk["document"],

        "season":
            chunk.get(
                "season"
            ),

        "article":
            None,

        "section_number":
            None,

        "parent_section":
            chunk.get(
                "parent_section"
            ),

        "section":
            chunk["section"],

        "chunk_index":
            chunk[
                "chunk_index"
            ],

        "section_chunk_count":
            chunk[
                "section_chunk_count"
            ],

        "page_numbers":
            None,

        "text":
            chunk["text"],

        "embedding_text":
            chunk[
                "embedding_text"
            ],

        "word_count":
            chunk[
                "word_count"
            ],

        "source_type":
            chunk[
                "source_type"
            ],

        "source_url":
            chunk[
                "source_url"
            ]
    }

    combined_chunks.append(
        record
    )

    global_chunk_id += 1


# NORMALIZE CBA CHUNKS

for chunk in cba_chunks:

    record = {
        "global_chunk_id":
            global_chunk_id,

        "source_document":
            "MLS CBA",

        "document":
            chunk["document"],

        "season":
            None,

        "article":
            chunk.get(
                "article"
            ),

        "section_number":
            chunk.get(
                "section_number"
            ),

        "parent_section":
            None,

        "section":
            chunk["section"],

        "chunk_index":
            chunk[
                "chunk_index"
            ],

        "section_chunk_count":
            chunk[
                "section_chunk_count"
            ],

        "page_numbers":
            chunk.get(
                "page_numbers"
            ),

        "text":
            chunk["text"],

        "embedding_text":
            chunk[
                "embedding_text"
            ],

        "word_count":
            chunk[
                "word_count"
            ],

        "source_type":
            chunk[
                "source_type"
            ],

        "source_url":
            chunk[
                "source_page_url"
            ]
    }

    combined_chunks.append(
        record
    )

    global_chunk_id += 1


print(
    f"\nTotal combined chunks: "
    f"{len(combined_chunks)}"
)


# LOAD EMBEDDING MODEL

print(
    "\nLoading embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True
)

print(
    "Embedding model loaded."
)


# CREATE EMBEDDINGS

texts = [
    chunk[
        "embedding_text"
    ]
    for chunk
    in combined_chunks
]

print(
    f"\nCreating embeddings for "
    f"{len(texts)} chunks..."
)

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype(
    "float32"
)

faiss.normalize_L2(
    embeddings
)


print(
    f"\nEmbedding matrix shape: "
    f"{embeddings.shape}"
)


# BUILD FAISS INDEX

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"Vectors stored: "
    f"{index.ntotal}"
)

print(
    f"Vector dimension: "
    f"{dimension}"
)


# SAVE FAISS INDEX

faiss.write_index(
    index,
    str(INDEX_FILE)
)


# SAVE METADATA

metadata = {
    "embedding_model":
        MODEL_NAME,

    "embedding_dimension":
        dimension,

    "vector_count":
        int(
            index.ntotal
        ),

    "source_summary": {
        "MLS Roster Rules":
            len(
                roster_chunks
            ),

        "MLS CBA Article 29":
            len(
                cba_chunks
            )
    },

    "chunks":
        combined_chunks
}


METADATA_FILE.write_text(
    json.dumps(
        metadata,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# QUALITY CHECK

norms = np.linalg.norm(
    embeddings,
    axis=1
)

print(
    f"\nMinimum vector norm: "
    f"{norms.min():.4f}"
)

print(
    f"Maximum vector norm: "
    f"{norms.max():.4f}"
)

print(
    f"Average vector norm: "
    f"{norms.mean():.4f}"
)


# SOURCE DISTRIBUTION

roster_count = sum(
    1
    for chunk
    in combined_chunks
    if chunk[
        "source_document"
    ] == "MLS Roster Rules"
)

cba_count = sum(
    1
    for chunk
    in combined_chunks
    if chunk[
        "source_document"
    ] == "MLS CBA"
)

print(
    "\nSource distribution:"
)

print(
    f"MLS Roster Rules: "
    f"{roster_count}"
)

print(
    f"MLS CBA: "
    f"{cba_count}"
)


print(
    "\nSaved:"
)

print(
    INDEX_FILE
)

print(
    METADATA_FILE
)

print(
    "\nSTEP 10 COMPLETE"
)