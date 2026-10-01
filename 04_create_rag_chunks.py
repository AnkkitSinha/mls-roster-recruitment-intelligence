from pathlib import Path
import json
import re


# CONFIGURATION

INPUT_FILE = Path(
    "data/processed/mls_roster_rules_2026_sections.json"
)

OUTPUT_JSON = Path(
    "data/processed/mls_roster_rules_2026_chunks.json"
)

OUTPUT_JSONL = Path(
    "data/processed/mls_roster_rules_2026_chunks.jsonl"
)

TARGET_WORDS = 220
MAX_WORDS = 280
OVERLAP_WORDS = 50


# LOAD STRUCTURED SECTIONS

print("MLS ROSTER INTELLIGENCE - RAG CHUNK BUILDER")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {INPUT_FILE}"
    )

sections = json.loads(
    INPUT_FILE.read_text(
        encoding="utf-8"
    )
)

print(
    f"\nLoaded {len(sections)} structured sections."
)


# SENTENCE SPLITTING

def split_into_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# WORD COUNT

def word_count(text):

    return len(
        text.split()
    )


# CREATE OVERLAP

def get_overlap_sentences(sentences):

    overlap = []
    total_words = 0

    for sentence in reversed(sentences):

        sentence_words = word_count(
            sentence
        )

        if (
            total_words + sentence_words
            > OVERLAP_WORDS
            and overlap
        ):
            break

        overlap.insert(
            0,
            sentence
        )

        total_words += sentence_words

    return overlap


# CHUNK ONE SECTION

def chunk_section(text):

    sentences = split_into_sentences(
        text
    )

    if not sentences:
        return []

    chunks = []

    current_sentences = []
    current_words = 0

    for sentence in sentences:

        sentence_words = word_count(
            sentence
        )

        if (
            current_sentences
            and current_words + sentence_words
            > MAX_WORDS
        ):

            chunk_text = " ".join(
                current_sentences
            )

            chunks.append(
                chunk_text
            )

            overlap = get_overlap_sentences(
                current_sentences
            )

            current_sentences = overlap.copy()

            current_words = sum(
                word_count(s)
                for s in current_sentences
            )

        current_sentences.append(
            sentence
        )

        current_words += sentence_words

        if current_words >= TARGET_WORDS:

            chunk_text = " ".join(
                current_sentences
            )

            chunks.append(
                chunk_text
            )

            overlap = get_overlap_sentences(
                current_sentences
            )

            current_sentences = overlap.copy()

            current_words = sum(
                word_count(s)
                for s in current_sentences
            )

    if current_sentences:

        final_chunk = " ".join(
            current_sentences
        )

        if (
            not chunks
            or final_chunk != chunks[-1]
        ):
            chunks.append(
                final_chunk
            )

    return chunks


# BUILD ALL CHUNKS

all_chunks = []

global_chunk_id = 1

for section in sections:

    section_chunks = chunk_section(
        section["text"]
    )

    total_section_chunks = len(
        section_chunks
    )

    for chunk_index, chunk_text in enumerate(
        section_chunks,
        start=1
    ):

        parent_section = section.get(
            "parent_section"
        )

        section_name = section[
            "section"
        ]

        if parent_section:

            heading_context = (
                f"{parent_section} > "
                f"{section_name}"
            )

        else:

            heading_context = section_name

        embedding_text = (
            f"Document: "
            f"{section['document']}\n"
            f"Season: "
            f"{section['season']}\n"
            f"Section: "
            f"{heading_context}\n\n"
            f"{chunk_text}"
        )

        record = {
            "chunk_id": global_chunk_id,
            "document": section[
                "document"
            ],
            "season": section[
                "season"
            ],
            "parent_section": parent_section,
            "section": section_name,
            "section_id": section[
                "section_id"
            ],
            "chunk_index": chunk_index,
            "section_chunk_count": (
                total_section_chunks
            ),
            "text": chunk_text,
            "embedding_text": (
                embedding_text
            ),
            "word_count": word_count(
                chunk_text
            ),
            "source_type": section[
                "source_type"
            ],
            "source_url": section[
                "source_url"
            ]
        }

        all_chunks.append(
            record
        )

        global_chunk_id += 1


# SAVE JSON

OUTPUT_JSON.write_text(
    json.dumps(
        all_chunks,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# SAVE JSONL

with OUTPUT_JSONL.open(
    "w",
    encoding="utf-8"
) as file:

    for chunk in all_chunks:

        file.write(
            json.dumps(
                chunk,
                ensure_ascii=False
            )
            + "\n"
        )


# SUMMARY

print("\n")
print("CHUNK SUMMARY")


print(
    f"Sections processed : "
    f"{len(sections)}"
)

print(
    f"Chunks created     : "
    f"{len(all_chunks)}"
)

chunk_sizes = [
    chunk["word_count"]
    for chunk in all_chunks
]

print(
    f"Minimum chunk size : "
    f"{min(chunk_sizes)} words"
)

print(
    f"Maximum chunk size : "
    f"{max(chunk_sizes)} words"
)

average_size = (
    sum(chunk_sizes)
    / len(chunk_sizes)
)

print(
    f"Average chunk size : "
    f"{average_size:.1f} words"
)


# CHUNKS PER SECTION

print("\n")
print("CHUNKS PER SECTION")

for section in sections:

    matching_chunks = [
        chunk
        for chunk in all_chunks
        if chunk["section_id"]
        == section["section_id"]
    ]

    print(
        f"{section['section']:<55} "
        f"{len(matching_chunks):>2} chunks"
    )


# IMPORTANT CHUNK PREVIEW

important_sections = [
    "Designated Player",
    "U22 Initiative Roster Slots",
    "General Allocation Money"
]

print("\n")
print("CHUNK PREVIEW")

for section_name in important_sections:

    matches = [
        chunk
        for chunk in all_chunks
        if chunk["section"]
        == section_name
    ]

    if not matches:
        continue

    print(
        f"\n[{section_name}]"
    )

    for chunk in matches:

        print(
            f"\nChunk "
            f"{chunk['chunk_index']}/"
            f"{chunk['section_chunk_count']}"
            f" | "
            f"{chunk['word_count']} words"
        )

        print(
            chunk["text"][:500]
        )

        print("...")


# OUTPUT FILES

print("\n")
print("OUTPUT FILES")


print(OUTPUT_JSON)
print(OUTPUT_JSONL)


print("\n")
print("STEP 4 COMPLETE")
