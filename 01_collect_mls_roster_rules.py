from pathlib import Path
from datetime import datetime, timezone
import json
import re

import requests
from bs4 import BeautifulSoup
# Configuration

URL = "https://www.mlssoccer.com/about/roster-rules-and-regulations"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_HTML_FILE = RAW_DIR / "mls_roster_rules_2026.html"
CLEAN_TEXT_FILE = PROCESSED_DIR / "mls_roster_rules_2026.txt"
METADATA_FILE = PROCESSED_DIR / "mls_roster_rules_2026_metadata.json"

# CREATE DIRECTORIES
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#Download Pages

print("MLS ROSTER INTELLIGENCE - DATA COLLECTION")

print("\nDownloading MLS roster rules...")

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

response = requests.get(
    URL,
    headers=headers,
    timeout=30
)

response.raise_for_status()

# Force UTF-8 decoding to preserve punctuation correctly
response.encoding = "utf-8"

print(f"HTTP Status: {response.status_code}")
print(f"Downloaded: {len(response.content):,} bytes")

#Save Raw HTML
RAW_HTML_FILE.write_text(
    response.text,
    encoding="utf-8"
)

print(f"\nRaw HTML saved:")
print(RAW_HTML_FILE)

#Parse HTML
print("\nParsing page...")

soup = BeautifulSoup(response.text, "lxml")


# Prefer the main page content if available
main_content = soup.find("main")

if main_content is None:
    main_content = soup.body

if main_content is None:
    raise RuntimeError("Could not locate page content.")

# Remove unwanted content
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

# Extract Text
text = main_content.get_text(
    separator="\n",
    strip=True
)

#Basic Text cleaning

lines = []

for line in text.splitlines():

    line = line.strip()

    # Skip empty lines
    if not line:
        continue

    # Normalize internal whitespace
    line = re.sub(r"\s+", " ", line)

    lines.append(line)

# TRY TO REMOVE WEBSITE CONTENT BEFORE THE ACTUAL RULES

possible_starts = [
    "As of February 3, 2026",
    "2026 MLS Roster Composition"
]

start_index = None

for i, line in enumerate(lines):

    for start_text in possible_starts:

        if start_text.lower() in line.lower():
            start_index = i
            break

    if start_index is not None:
        break


if start_index is not None:

    lines = lines[start_index:]

    print(
        f"\nDetected beginning of roster rules at line "
        f"{start_index + 1}"
    )

else:

    print(
        "\nWARNING: Could not automatically detect "
        "the start of the roster rules."
    )


clean_text = "\n".join(lines)

# SAVE CLEANED TEXT

CLEAN_TEXT_FILE.write_text(
    clean_text,
    encoding="utf-8"
)

print("\nClean text saved:")
print(CLEAN_TEXT_FILE)

# SAVE SOURCE METADATA

metadata = {
    "document_name": "MLS Roster Rules and Regulations",
    "season": 2026,
    "source_type": "Official MLS website",
    "source_url": URL,
    "retrieved_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "raw_html_file": str(RAW_HTML_FILE),
    "clean_text_file": str(CLEAN_TEXT_FILE),
    "character_count": len(clean_text),
    "line_count": len(lines)
}


METADATA_FILE.write_text(
    json.dumps(
        metadata,
        indent=4
    ),
    encoding="utf-8"
)


print("\nMetadata saved:")
print(METADATA_FILE)

# DATA QUALITY SUMMARY

word_count = len(clean_text.split())

print("\n")
print("DOCUMENT SUMMARY")

print(f"Characters : {len(clean_text):,}")
print(f"Words      : {word_count:,}")
print(f"Lines      : {len(lines):,}")

# PREVIEW FIRST 30 LINES

print("\n")
print("DOCUMENT PREVIEW")

for line in lines[:30]:
    print(line)


print("\n" )
print("STEP 1 COMPLETE")
