from pathlib import Path
import json
import re

from bs4 import BeautifulSoup

# CONFIGURATION
RAW_HTML_FILE = Path("data/raw/mls_roster_rules_2026.html")
CLEAN_TEXT_FILE = Path("data/processed/mls_roster_rules_2026.txt")

OUTPUT_FILE = Path(
    "data/processed/mls_roster_rules_2026_structure.json"
)

# CHECK FILES

if not RAW_HTML_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {RAW_HTML_FILE}"
    )

if not CLEAN_TEXT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {CLEAN_TEXT_FILE}"
    )

print("MLS ROSTER INTELLIGENCE - DOCUMENT STRUCTURE")

# LOAD HTML
html = RAW_HTML_FILE.read_text(
    encoding="utf-8"
)

soup = BeautifulSoup(
    html,
    "lxml"
)

# FIND MAIN PAGE CONTENT
main_content = soup.find("main")

if main_content is None:
    main_content = soup.body

if main_content is None:
    raise RuntimeError(
        "Could not locate main document content."
    )


# EXTRACT HTML HEADINGS

html_headings = []

for tag in main_content.find_all(
    ["h1", "h2", "h3", "h4", "h5", "h6"]
):

    text = tag.get_text(
        " ",
        strip=True
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if not text:
        continue

    html_headings.append(
        {
            "tag": tag.name,
            "text": text
        }
    )

# REMOVE DUPLICATE HTML HEADINGS
unique_html_headings = []

seen = set()

for heading in html_headings:

    key = heading["text"].lower()

    if key not in seen:

        seen.add(key)

        unique_html_headings.append(
            heading
        )

# LOAD CLEAN TEXT
clean_text = CLEAN_TEXT_FILE.read_text(
    encoding="utf-8"
)

lines = [
    line.strip()
    for line in clean_text.splitlines()
    if line.strip()
]

# FIND POSSIBLE TEXT-BASED SUBHEADINGS

candidate_headings = []

for i, line in enumerate(lines):

    # Ignore very long sentences
    if len(line) > 100:
        continue

    # Ignore lines that look like normal sentences
    if line.endswith(
        (
            ".",
            "?",
            "!"
        )
    ):
        continue

    # Ignore lines with too many words
    word_count = len(
        line.split()
    )

    if word_count > 14:
        continue

    # Ignore obvious date/source line
    if line.lower().startswith(
        "as of "
    ):
        continue

    # Candidate heading
    candidate_headings.append(
        {
            "line_number": i + 1,
            "text": line
        }
    )

# REMOVE DUPLICATE CANDIDATES
unique_candidates = []

seen_candidates = set()

for heading in candidate_headings:

    key = heading["text"].lower()

    if key not in seen_candidates:

        seen_candidates.add(key)

        unique_candidates.append(
            heading
        )

# SAVE STRUCTURE
structure = {
    "document": "MLS Roster Rules and Regulations",
    "season": 2026,
    "html_headings": unique_html_headings,
    "candidate_text_headings": unique_candidates
}


OUTPUT_FILE.write_text(
    json.dumps(
        structure,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

# PRINT HTML HEADINGS
print("\n" )
print("HTML HEADINGS")

if unique_html_headings:

    for heading in unique_html_headings:

        print(
            f"{heading['tag'].upper():4} | "
            f"{heading['text']}"
        )

else:

    print(
        "No HTML heading tags were detected."
    )

# PRINT TEXT-BASED CANDIDATE HEADINGS

print("\n")
print("POSSIBLE TEXT SUBHEADINGS")

for heading in unique_candidates:

    print(
        f"Line "
        f"{heading['line_number']:>3} | "
        f"{heading['text']}"
    )

# SUMMARY

print("\n")
print("STRUCTURE SUMMARY")

print(
    f"HTML headings detected      : "
    f"{len(unique_html_headings)}"
)

print(
    f"Text heading candidates     : "
    f"{len(unique_candidates)}"
)

print("\nStructure saved:")
print(OUTPUT_FILE)


print("\n")
print("STEP 2 COMPLETE")
