from pathlib import Path
import json
import re

from bs4 import BeautifulSoup

# CONFIGURATION

RAW_HTML_FILE = Path("data/raw/mls_roster_rules_2026.html")

OUTPUT_JSONL = Path(
    "data/processed/mls_roster_rules_2026_sections.jsonl"
)

OUTPUT_JSON = Path(
    "data/processed/mls_roster_rules_2026_sections.json"
)

SOURCE_URL = (
    "https://www.mlssoccer.com/about/"
    "roster-rules-and-regulations"
)

# LOAD HTML
print("MLS ROSTER INTELLIGENCE - STRUCTURED SECTION BUILDER")


if not RAW_HTML_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {RAW_HTML_FILE}"
    )

html = RAW_HTML_FILE.read_text(
    encoding="utf-8"
)

soup = BeautifulSoup(
    html,
    "lxml"
)

# FIND MAIN CONTENT

main_content = soup.find("main")

if main_content is None:
    main_content = soup.body

if main_content is None:
    raise RuntimeError(
        "Could not locate main page content."
    )

# REMOVE NON-CONTENT ELEMENTS\

for tag in main_content.find_all(
    [
        "script",
        "style",
        "noscript",
        "svg",
        "button",
        "form"
    ]
):
    tag.decompose()

# CLEAN TEXT HELPER\

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
# STRUCTURE STORAGE

sections = []

current_h2 = None
current_h3 = None

current_title = None
current_parent = None
current_level = None

current_text = []

# SAVE CURRENT SECTION

def save_current_section():

    global current_text

    if current_title is None:
        return

    text = clean_text(
        " ".join(current_text)
    )

    if not text:
        current_text = []
        return

    record = {
        "section_id": len(sections) + 1,
        "document": "MLS Roster Rules and Regulations",
        "season": 2026,
        "parent_section": current_parent,
        "section": current_title,
        "heading_level": current_level,
        "text": text,
        "word_count": len(text.split()),
        "source_type": "Official MLS website",
        "source_url": SOURCE_URL
    }

    sections.append(record)

    current_text = []

# WALK THROUGH DOCUMENT

print("\nParsing structured sections...")

for element in main_content.find_all(
    [
        "h2",
        "h3",
        "p",
        "li",
        "table"
    ]
):

    tag_name = element.name

    element_text = clean_text(
        element.get_text(
            " ",
            strip=True
        )
    )

    if not element_text:
        continue

    # STOP BEFORE OLD ROSTER RULES

    if (
        tag_name == "h2"
        and (
            "2021 MLS Roster Rules" in element_text
            or
            "2020 MLS Roster Rules" in element_text
        )
    ):
        save_current_section()
        break

    # H2 = MAIN CATEGORY\

    if tag_name == "h2":

        save_current_section()

        current_h2 = element_text
        current_h3 = None

        current_title = current_h2
        current_parent = None
        current_level = "H2"

        continue

    # H3 = SUBSECTION
    if tag_name == "h3":

        # Ignore date heading
        if element_text.lower().startswith(
            "as of "
        ):
            continue

        save_current_section()

        current_h3 = element_text

        current_title = current_h3
        current_parent = current_h2
        current_level = "H3"

        continue

    # CONTENT

    if current_title is not None:

        current_text.append(
            element_text
        )

# SAVE FINAL SECTION

save_current_section()

# REMOVE DUPLICATED TEXT INSIDE EACH SECTION

for section in sections:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        section["text"]
    )

    unique_sentences = []

    seen = set()

    for sentence in sentences:

        key = sentence.strip().lower()

        if not key:
            continue

        if key not in seen:

            seen.add(key)

            unique_sentences.append(
                sentence.strip()
            )

    cleaned = " ".join(
        unique_sentences
    )

    section["text"] = cleaned

    section["word_count"] = len(
        cleaned.split()
    )

# SAVE JSON

OUTPUT_JSON.write_text(
    json.dumps(
        sections,
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

    for section in sections:

        file.write(
            json.dumps(
                section,
                ensure_ascii=False
            )
            + "\n"
        )

# DISPLAY RESULTs

print("\n")
print("STRUCTURED SECTION SUMMARY")

print(
    f"Sections created : "
    f"{len(sections)}"
)

total_words = sum(
    section["word_count"]
    for section in sections
)

print(
    f"Total words      : "
    f"{total_words:,}"
)

# PRINT SECTION LIST
print("\n")
print("SECTIONS")


for section in sections:

    parent = (
        section["parent_section"]
        if section["parent_section"]
        else "-"
    )

    print(
        f"{section['section_id']:>2} | "
        f"{section['heading_level']} | "
        f"{section['section']} | "
        f"Parent: {parent} | "
        f"{section['word_count']} words"
    )

# PREVIEW IMPORTANT SECTIONS
important_sections = [
    "Designated Player",
    "U22 Initiative Roster Slots",
    "General Allocation Money",
    "Free Agency"
]

print("\n")
print("IMPORTANT SECTION PREVIEW")

for target in important_sections:

    matches = [
        section
        for section in sections
        if section["section"] == target
    ]

    if not matches:
        continue

    section = matches[0]

    print(
        f"\n[{section['section']}]"
    )

    print(
        section["text"][:700]
    )

    print("...")

# OUTPUT FILES
print("\n")
print("OUTPUT FILES")

print(OUTPUT_JSON)
print(OUTPUT_JSONL)


print("\n")
print("STEP 3 COMPLETE")
