from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime, timezone
import json
import re

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


# CONFIGURATION

CBA_PAGE_URL = "https://mlsplayers.org/resources/cba"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PDF_FILE = RAW_DIR / "mls_cba_2020_2028.pdf"

PAGES_JSON = (
    PROCESSED_DIR
    / "mls_cba_2020_2028_pages.json"
)

PAGES_JSONL = (
    PROCESSED_DIR
    / "mls_cba_2020_2028_pages.jsonl"
)

TEXT_FILE = (
    PROCESSED_DIR
    / "mls_cba_2020_2028.txt"
)

METADATA_FILE = (
    PROCESSED_DIR
    / "mls_cba_2020_2028_metadata.json"
)


# CREATE DIRECTORIES

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("MLS ROSTER INTELLIGENCE - CBA COLLECTION")

# FIND OFFICIAL PDF LINK

print("\nFinding official English CBA PDF...")

headers = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

page_response = requests.get(
    CBA_PAGE_URL,
    headers=headers,
    timeout=30
)

page_response.raise_for_status()

page_response.encoding = "utf-8"

soup = BeautifulSoup(
    page_response.text,
    "lxml"
)

pdf_url = None

for link in soup.find_all(
    "a",
    href=True
):

    link_text = link.get_text(
        " ",
        strip=True
    ).lower()

    if (
        "2020-2028" in link_text
        and "english" in link_text
    ):

        pdf_url = urljoin(
            CBA_PAGE_URL,
            link["href"]
        )

        break


if pdf_url is None:

    raise RuntimeError(
        "Could not locate the English "
        "2020-2028 CBA PDF."
    )


print("\nCBA PDF found:")
print(pdf_url)


# DOWNLOAD PDF

print("\nDownloading CBA PDF...")

pdf_response = requests.get(
    pdf_url,
    headers=headers,
    timeout=60
)

pdf_response.raise_for_status()

PDF_FILE.write_bytes(
    pdf_response.content
)

print(
    f"Downloaded: "
    f"{len(pdf_response.content):,} bytes"
)

print("\nRaw PDF saved:")
print(PDF_FILE)


# LOAD PDF

print("\nReading PDF...")

reader = PdfReader(
    str(PDF_FILE)
)

page_count = len(
    reader.pages
)

print(
    f"PDF pages: {page_count}"
)


# CLEAN TEXT FUNCTION

def clean_page_text(text):

    if not text:
        return ""

    text = text.replace(
        "\u00ad",
        ""
    )

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# EXTRACT PAGES

print("\nExtracting page text...")

pages = []

total_words = 0

for page_number, page in enumerate(
    reader.pages,
    start=1
):

    raw_text = page.extract_text()

    clean_text = clean_page_text(
        raw_text
    )

    page_words = len(
        clean_text.split()
    )

    total_words += page_words

    page_record = {
        "document": (
            "MLS-MLSPA Collective "
            "Bargaining Agreement"
        ),
        "document_short_name": "MLS CBA",
        "agreement_period": "2020-2028",
        "page_number": page_number,
        "text": clean_text,
        "word_count": page_words,
        "source_type": (
            "Official MLS Players "
            "Association document"
        ),
        "source_page_url": CBA_PAGE_URL,
        "source_pdf_url": pdf_url
    }

    pages.append(
        page_record
    )


# SAVE JSON

PAGES_JSON.write_text(
    json.dumps(
        pages,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# SAVE JSONL

with PAGES_JSONL.open(
    "w",
    encoding="utf-8"
) as file:

    for page in pages:

        file.write(
            json.dumps(
                page,
                ensure_ascii=False
            )
            + "\n"
        )


# SAVE READABLE TEXT FILE

with TEXT_FILE.open(
    "w",
    encoding="utf-8"
) as file:

    for page in pages:

        file.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        file.write(
            f"PAGE {page['page_number']}\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            page["text"]
        )

        file.write(
            "\n"
        )


# SAVE METADATA

metadata = {
    "document": (
        "MLS-MLSPA Collective "
        "Bargaining Agreement"
    ),
    "agreement_period": "2020-2028",
    "source_page_url": CBA_PAGE_URL,
    "source_pdf_url": pdf_url,
    "retrieved_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "pdf_pages": page_count,
    "total_words": total_words,
    "raw_pdf_file": str(
        PDF_FILE
    ),
    "pages_json_file": str(
        PAGES_JSON
    )
}

METADATA_FILE.write_text(
    json.dumps(
        metadata,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# FIND IMPORTANT CONTENT

search_terms = [
    "ARTICLE 29",
    "Free Agency",
    "PLAYER MOVEMENT RULES"
]

print("\n")
print("IMPORTANT CONTENT CHECK")

for term in search_terms:

    matching_pages = []

    for page in pages:

        if term.lower() in (
            page["text"].lower()
        ):

            matching_pages.append(
                page["page_number"]
            )

    print(
        f"{term:<25} "
        f"Pages: {matching_pages}"
    )


# SUMMARY

print("\n")
print("CBA SUMMARY")

print(
    f"Pages extracted : "
    f"{page_count}"
)

print(
    f"Total words     : "
    f"{total_words:,}"
)

print("\nOutput files:")

print(PDF_FILE)
print(PAGES_JSON)
print(PAGES_JSONL)
print(TEXT_FILE)
print(METADATA_FILE)


# PREVIEW FIRST PAGE WITH ARTICLE 29

article_29_pages = [
    page
    for page in pages
    if "article 29"
    in page["text"].lower()
]

if article_29_pages:

    page = article_29_pages[0]

    print("\n")
    print(
        f"ARTICLE 29 PREVIEW - "
        f"PAGE {page['page_number']}"
    )

    print(
        page["text"][:1500]
    )


print("\n")
print("STEP 7 COMPLETE")