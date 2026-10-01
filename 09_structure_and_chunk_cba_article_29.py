from pathlib import Path
import json
import re


# CONFIGURATION

INPUT_FILE = Path(
    "data/processed/mls_cba_article_29_pages.json"
)

SECTIONS_JSON = Path(
    "data/processed/mls_cba_article_29_sections.json"
)

CHUNKS_JSON = Path(
    "data/processed/mls_cba_article_29_chunks.json"
)

CHUNKS_JSONL = Path(
    "data/processed/mls_cba_article_29_chunks.jsonl"
)

TARGET_WORDS = 220
MAX_WORDS = 280
OVERLAP_WORDS = 50


# SECTION DEFINITIONS

SECTION_DEFINITIONS = [
    {
        "number": "29.1",
        "title": "Players Whose Team Wants to Waive the Player",
        "pattern": (
            r"Section\s+29\.1\s+"
            r"Players\s+Whose\s+Team\s+Wants\s+to\s+"
            r"Waive\s+the\s+Player\s*:"
        )
    },
    {
        "number": "29.2",
        "title": (
            "Re-Entry Draft – Players Whose Options "
            "Are Not Exercised (22 + 1)"
        ),
        "pattern": (
            r"Section\s+29\.2\s+"
            r"Re[- ]?Entry\s+Draft\s*[–—-]?\s*"
            r"Players\s+Whose\s+Options\s+Are\s+"
            r"Not\s+Exercised\s*"
            r"\(\s*22\s*\+\s*1\s*\)\s*:"
        )
    },
    {
        "number": "29.3",
        "title": (
            "Re-Entry Draft – Out-of-Contract "
            "Players (22 + 1)"
        ),
        "pattern": (
            r"Section\s+29\.3\s+"
            r"Re[- ]?Entry\s+Draft\s*[–—-]?\s*"
            r"Out[- ]of[- ]Contract\s+Players\s*"
            r"\(\s*22\s*\+\s*1\s*\)\s*:"
        )
    },
    {
        "number": "29.4",
        "title": "Free Agency",
        "pattern": (
            r"Section\s+29\.4\s+"
            r"Free\s+Agency\s*:"
        )
    },
    {
        "number": "29.5",
        "title": "Free Agency: Miscellaneous",
        "pattern": (
            r"Section\s+29\.5\s+"
            r"Free\s+Agency\s*:\s*"
            r"Miscellaneous\s*:"
        )
    },
    {
        "number": "29.6",
        "title": (
            "Mechanics and Timing of Re-Entry Draft "
            "and Free Agency"
        ),
        "pattern": (
            r"Section\s+29\.6\s+"
            r"Mechanics\s+and\s+Timing\s+of\s+"
            r"Re[- ]?Entry\s+Draft\s+and\s+"
            r"Free\s+Agency\s*:"
        )
    }
]


# LOAD ARTICLE 29

print(
    "MLS ROSTER INTELLIGENCE - "
    "CBA ARTICLE 29 FINAL STRUCTURE"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {INPUT_FILE}"
    )

pages = json.loads(
    INPUT_FILE.read_text(
        encoding="utf-8"
    )
)

print(
    f"\nLoaded {len(pages)} Article 29 pages."
)


# CLEAN PAGE TEXT

def clean_page_text(text):

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if re.fullmatch(
            r"\d{1,3}",
            line
        ):
            continue

        lines.append(
            line
        )

    text = " ".join(
        lines
    )

    text = text.replace(
        "\u00ad",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


clean_pages = []

for page in pages:

    clean_pages.append(
        {
            "page_number":
                page["page_number"],
            "text":
                clean_page_text(
                    page["text"]
                )
        }
    )


# FIND SECTION HEADINGS

heading_locations = []

for page in clean_pages:

    for section in SECTION_DEFINITIONS:

        match = re.search(
            section["pattern"],
            page["text"],
            flags=re.IGNORECASE
        )

        if match:

            heading_locations.append(
                {
                    "section_number":
                        section["number"],
                    "section":
                        section["title"],
                    "page_number":
                        page["page_number"],
                    "start":
                        match.start(),
                    "end":
                        match.end()
                }
            )


heading_locations.sort(
    key=lambda item: (
        item["page_number"],
        item["start"]
    )
)


print("\nDetected headings:")

for heading in heading_locations:

    print(
        f"{heading['section_number']} | "
        f"{heading['section']} | "
        f"PDF page "
        f"{heading['page_number']}"
    )


# VALIDATE HEADINGS

detected_numbers = [
    heading["section_number"]
    for heading
    in heading_locations
]

expected_numbers = [
    section["number"]
    for section
    in SECTION_DEFINITIONS
]

missing = [
    number
    for number
    in expected_numbers
    if number
    not in detected_numbers
]

duplicates = [
    number
    for number
    in detected_numbers
    if detected_numbers.count(
        number
    ) > 1
]


if missing:

    raise RuntimeError(
        f"Missing Article 29 sections: {missing}"
    )


if duplicates:

    raise RuntimeError(
        "Duplicate Article 29 sections "
        f"detected: {sorted(set(duplicates))}"
    )


# BUILD STRUCTURED SECTIONS

sections = []

for heading_index, heading in enumerate(
    heading_locations
):

    current_page = heading[
        "page_number"
    ]

    if (
        heading_index + 1
        < len(heading_locations)
    ):

        next_heading = heading_locations[
            heading_index + 1
        ]

    else:

        next_heading = None

    segments = []

    for page in clean_pages:

        page_number = page[
            "page_number"
        ]

        text = page[
            "text"
        ]

        if page_number < current_page:
            continue

        if next_heading:

            if (
                page_number
                > next_heading[
                    "page_number"
                ]
            ):
                break

        start = 0
        end = len(
            text
        )

        if (
            page_number
            == current_page
        ):

            start = heading[
                "end"
            ]

        if (
            next_heading
            and
            page_number
            == next_heading[
                "page_number"
            ]
        ):

            end = next_heading[
                "start"
            ]

        segment_text = text[
            start:end
        ].strip()

        if segment_text:

            segments.append(
                {
                    "page_number":
                        page_number,
                    "text":
                        segment_text
                }
            )

        if (
            next_heading
            and
            page_number
            == next_heading[
                "page_number"
            ]
        ):
            break

    full_text = " ".join(
        segment["text"]
        for segment
        in segments
    )

    full_text = re.sub(
        r"\s+",
        " ",
        full_text
    ).strip()

    page_numbers = sorted(
        {
            segment["page_number"]
            for segment
            in segments
        }
    )

    sections.append(
        {
            "section_id":
                heading_index + 1,
            "document":
                "MLS-MLSPA Collective "
                "Bargaining Agreement",
            "document_short_name":
                "MLS CBA",
            "article":
                "Article 29 - "
                "Player Movement Rules",
            "section_number":
                heading[
                    "section_number"
                ],
            "section":
                heading[
                    "section"
                ],
            "page_numbers":
                page_numbers,
            "start_page":
                min(
                    page_numbers
                ),
            "end_page":
                max(
                    page_numbers
                ),
            "word_count":
                len(
                    full_text.split()
                ),
            "text":
                full_text,
            "segments":
                segments
        }
    )


# SAVE SECTIONS

sections_output = []

for section in sections:

    sections_output.append(
        {
            key: value
            for key, value
            in section.items()
            if key != "segments"
        }
    )


SECTIONS_JSON.write_text(
    json.dumps(
        sections_output,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# SENTENCE SPLITTER

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence
        in sentences
        if sentence.strip()
    ]


def count_words(text):

    return len(
        text.split()
    )


# CREATE SENTENCE RECORDS

def build_sentence_records(
    section
):

    records = []

    for segment in section[
        "segments"
    ]:

        sentences = split_sentences(
            segment["text"]
        )

        for sentence in sentences:

            records.append(
                {
                    "page_number":
                        segment[
                            "page_number"
                        ],
                    "text":
                        sentence
                }
            )

    return records


# BUILD OVERLAP

def get_overlap(records):

    overlap = []

    total_words = 0

    for record in reversed(
        records
    ):

        words = count_words(
            record["text"]
        )

        if (
            total_words + words
            > OVERLAP_WORDS
            and overlap
        ):
            break

        overlap.insert(
            0,
            record
        )

        total_words += words

    return overlap


# CHUNK SECTION

def chunk_section(section):

    records = (
        build_sentence_records(
            section
        )
    )

    chunks = []

    current = []

    current_words = 0

    for record in records:

        words = count_words(
            record["text"]
        )

        if (
            current
            and
            current_words + words
            > MAX_WORDS
        ):

            chunks.append(
                current.copy()
            )

            current = (
                get_overlap(
                    current
                )
            )

            current_words = sum(
                count_words(
                    item["text"]
                )
                for item
                in current
            )

        current.append(
            record
        )

        current_words += words

        if (
            current_words
            >= TARGET_WORDS
        ):

            chunks.append(
                current.copy()
            )

            current = (
                get_overlap(
                    current
                )
            )

            current_words = sum(
                count_words(
                    item["text"]
                )
                for item
                in current
            )

    if current:

        current_text = " ".join(
            item["text"]
            for item
            in current
        )

        previous_text = ""

        if chunks:

            previous_text = " ".join(
                item["text"]
                for item
                in chunks[-1]
            )

        if (
            current_text
            != previous_text
        ):

            chunks.append(
                current.copy()
            )

    return chunks


# CREATE RAG CHUNKS

all_chunks = []

chunk_id = 1

for section in sections:

    section_chunks = (
        chunk_section(
            section
        )
    )

    total_chunks = len(
        section_chunks
    )

    for chunk_index, records in enumerate(
        section_chunks,
        start=1
    ):

        chunk_text = " ".join(
            record["text"]
            for record
            in records
        )

        pages_in_chunk = sorted(
            {
                record[
                    "page_number"
                ]
                for record
                in records
            }
        )

        embedding_text = (
            f"Document: MLS-MLSPA "
            f"Collective Bargaining Agreement\n"
            f"Article: Article 29 - "
            f"Player Movement Rules\n"
            f"Section: "
            f"{section['section_number']} "
            f"{section['section']}\n"
            f"PDF Pages: "
            f"{', '.join(map(str, pages_in_chunk))}"
            f"\n\n"
            f"{chunk_text}"
        )

        chunk_record = {
            "chunk_id":
                chunk_id,
            "document":
                section[
                    "document"
                ],
            "document_short_name":
                "MLS CBA",
            "article":
                section[
                    "article"
                ],
            "section_number":
                section[
                    "section_number"
                ],
            "section":
                section[
                    "section"
                ],
            "section_id":
                section[
                    "section_id"
                ],
            "chunk_index":
                chunk_index,
            "section_chunk_count":
                total_chunks,
            "page_numbers":
                pages_in_chunk,
            "start_page":
                min(
                    pages_in_chunk
                ),
            "end_page":
                max(
                    pages_in_chunk
                ),
            "text":
                chunk_text,
            "embedding_text":
                embedding_text,
            "word_count":
                count_words(
                    chunk_text
                ),
            "source_type":
                (
                    "Official MLS Players "
                    "Association document"
                ),
            "source_page_url":
                (
                    "https://mlsplayers.org/"
                    "resources/cba"
                )
        }

        all_chunks.append(
            chunk_record
        )

        chunk_id += 1


# SAVE CHUNKS

CHUNKS_JSON.write_text(
    json.dumps(
        all_chunks,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


with CHUNKS_JSONL.open(
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


# PRINT SUMMARY

print(
    f"\nClean sections detected: "
    f"{len(sections)}"
)

for section in sections:

    print(
        f"{section['section_number']} | "
        f"{section['section']} | "
        f"Pages "
        f"{section['start_page']}-"
        f"{section['end_page']} | "
        f"{section['word_count']} words"
    )


print(
    f"\nRAG chunks created: "
    f"{len(all_chunks)}"
)

chunk_sizes = [
    chunk["word_count"]
    for chunk
    in all_chunks
]

print(
    f"Minimum chunk size: "
    f"{min(chunk_sizes)} words"
)

print(
    f"Maximum chunk size: "
    f"{max(chunk_sizes)} words"
)

print(
    f"Average chunk size: "
    f"{sum(chunk_sizes) / len(chunk_sizes):.1f} words"
)


print("\nSaved:")

print(SECTIONS_JSON)
print(CHUNKS_JSON)
print(CHUNKS_JSONL)

print(
    "\nSTEP 9 COMPLETE"
)