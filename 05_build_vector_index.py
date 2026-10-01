from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# CONFIGURATION

INPUT_FILE = Path(
    "data/processed/mls_roster_rules_2026_chunks.json"
)

VECTOR_DIR = Path(
    "vector_store"
)

INDEX_FILE = VECTOR_DIR / "mls_roster_rules.faiss"

METADATA_FILE = VECTOR_DIR / "mls_roster_rules_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# LOAD CHUNKS

print("MLS ROSTER INTELLIGENCE - EMBEDDING BUILDER")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {INPUT_FILE}"
    )

VECTOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)

chunks = json.loads(
    INPUT_FILE.read_text(
        encoding="utf-8"
    )
)

print(
    f"\nLoaded {len(chunks)} RAG chunks."
)


# LOAD EMBEDDING MODEL

print(
    f"\nLoading embedding model:"
    f"\n{MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME
)

print("\nEmbedding model loaded.")


# PREPARE TEXT

texts = [
    chunk["embedding_text"]
    for chunk in chunks
]

print(
    f"\nCreating embeddings for "
    f"{len(texts)} chunks..."
)


# CREATE EMBEDDINGS

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype(
    "float32"
)

print(
    f"\nEmbedding matrix shape: "
    f"{embeddings.shape}"
)


# NORMALIZE EMBEDDINGS

faiss.normalize_L2(
    embeddings
)


# CREATE FAISS INDEX

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"\nFAISS index created."
)

print(
    f"Vector dimension : {dimension}"
)

print(
    f"Vectors stored   : {index.ntotal}"
)


# SAVE FAISS INDEX

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print(
    f"\nFAISS index saved:"
    f"\n{INDEX_FILE}"
)


# SAVE METADATA

metadata = {
    "embedding_model": MODEL_NAME,
    "embedding_dimension": dimension,
    "vector_count": int(
        index.ntotal
    ),
    "chunks": chunks
}

METADATA_FILE.write_text(
    json.dumps(
        metadata,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print(
    f"\nMetadata saved:"
    f"\n{METADATA_FILE}"
)


# VECTOR QUALITY CHECK

vector_norms = np.linalg.norm(
    embeddings,
    axis=1
)

print("\n")
print("VECTOR QUALITY CHECK")

print(
    f"Minimum vector norm : "
    f"{vector_norms.min():.4f}"
)

print(
    f"Maximum vector norm : "
    f"{vector_norms.max():.4f}"
)

print(
    f"Average vector norm : "
    f"{vector_norms.mean():.4f}"
)


# PREVIEW

print("\n")
print("SAMPLE VECTOR")

print(
    f"Chunk ID: "
    f"{chunks[0]['chunk_id']}"
)

print(
    f"Section: "
    f"{chunks[0]['section']}"
)

print(
    "\nFirst 10 embedding values:"
)

print(
    embeddings[0][:10]
)


print("\n")
print("STEP 5 COMPLETE")