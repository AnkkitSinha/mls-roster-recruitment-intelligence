from pathlib import Path
import json
import re


# CONFIGURATION

INPUT_FILE = Path(
    "data/processed/mls_cba_2020_2028_pages.json"
)

OUTPUT_JSON = Path(
    "data/processed/mls_cba_article_29_pages.json"
)

OUTPUT_TEXT = Path(
    "data/processed/mls_cba_article_29.txt"
)

OUTPUT_STRUCTURE = Path(
    "data/processed/mls_cba_article_29_structure.json"
)

# LOAD CBA PAGES
print("MLS ROSTER INTELLIGENCE - CBA ARTICLE 29 EXTRACTOR")

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
    f"\nLoaded {len(pages)} CBA pages."
)


# FIND REAL ARTICLE 29 START

article_29_candidates = []

for page in pages:

    text = page["text"].lower()

    if (
        "article 29"
        in text
        and
        "player movement rules"
        in text
    ):

        article_29_candidates.append(
            page["page_number"]
        )


print(
    f"\nARTICLE 29 occurrences: "
    f"{article_29_candidates}"
)


# Ignore table of contents occurrences.
# The actual Article 29 is expected near the end of the CBA.

real_candidates = [
    page_number
    for page_number
    in article_29_candidates
    if page_number >= 90
]

if not real_candidates:

    raise RuntimeError(
        "Could not identify the real "
        "Article 29 start page."
    )

article_start_page = min(
    real_candidates
)

print(
    f"Detected actual Article 29 start: "
    f"PDF page {article_start_page}"
)


# COLLECT ARTICLE 29 PAGES

article_pages = []

for page in pages:

    if page["page_number"] < article_start_page:
        continue

    text = page["text"]

    # Stop when an exhibit begins
    exhibit_match = re.search(
        r"\bEXHIBIT\s+1\b",
        text,
        flags=re.IGNORECASE
    )

    if exhibit_match:

        text_before_exhibit = text[
            :exhibit_match.start()
        ].strip()

        if text_before_exhibit:

            new_page = page.copy()

            new_page["text"] = (
                text_before_exhibit
            )

            new_page["word_count"] = len(
                text_before_exhibit.split()
            )

            article_pages.append(
                new_page
            )

        break

    article_pages.append(
        page
    )


if not article_pages:

    raise RuntimeError(
        "No Article 29 pages were extracted."
    )


# SAVE ARTICLE PAGES

OUTPUT_JSON.write_text(
    json.dumps(
        article_pages,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# SAVE READABLE ARTICLE TEXT

with OUTPUT_TEXT.open(
    "w",
    encoding="utf-8"
) as file:

    for page in article_pages:

        file.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        file.write(
            f"PDF PAGE "
            f"{page['page_number']}\n"
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


# FIND POSSIBLE SECTION HEADINGS

candidate_headings = []

for page in article_pages:

    lines = [
        line.strip()
        for line in page["text"].splitlines()
        if line.strip()
    ]

    for line in lines:

        normalized = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if len(normalized) > 100:
            continue

        word_count = len(
            normalized.split()
        )

        if word_count > 14:
            continue

        # Numbered CBA subsection
        if re.match(
            r"^\d+\.\d+",
            normalized
        ):

            candidate_headings.append(
                {
                    "page_number":
                        page["page_number"],
                    "text":
                        normalized,
                    "type":
                        "numbered_section"
                }
            )

            continue

        # Known player movement concepts
        keywords = [
            "free agency",
            "waiver",
            "re-entry",
            "out of contract",
            "option",
            "player movement",
            "right of first refusal"
        ]

        if any(
            keyword
            in normalized.lower()
            for keyword in keywords
        ):

            candidate_headings.append(
                {
                    "page_number":
                        page["page_number"],
                    "text":
                        normalized,
                    "type":
                        "keyword_candidate"
                }
            )


# REMOVE DUPLICATES

unique_headings = []

seen = set()

for heading in candidate_headings:

    key = (
        heading["page_number"],
        heading["text"].lower()
    )

    if key not in seen:

        seen.add(key)

        unique_headings.append(
            heading
        )


# SAVE STRUCTURE INSPECTION

structure = {
    "document":
        "MLS-MLSPA Collective Bargaining Agreement",
    "article":
        "Article 29 - Player Movement Rules",
    "start_pdf_page":
        article_pages[0]["page_number"],
    "end_pdf_page":
        article_pages[-1]["page_number"],
    "pages":
        len(article_pages),
    "candidate_headings":
        unique_headings
}

OUTPUT_STRUCTURE.write_text(
    json.dumps(
        structure,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# SUMMARY

total_words = sum(
    page["word_count"]
    for page in article_pages
)

print("\n")
print("ARTICLE 29 SUMMARY")

print(
    f"Start PDF page : "
    f"{article_pages[0]['page_number']}"
)

print(
    f"End PDF page   : "
    f"{article_pages[-1]['page_number']}"
)

print(
    f"Pages extracted: "
    f"{len(article_pages)}"
)

print(
    f"Words extracted: "
    f"{total_words:,}"
)


# SECTION CANDIDATES

print("\n")
print("POSSIBLE ARTICLE 29 SECTIONS")

for heading in unique_headings:

    print(
        f"Page "
        f"{heading['page_number']:>3} | "
        f"{heading['text']}"
    )


# FREE AGENCY SEARCH

print("\n")
print("FREE AGENCY CONTENT CHECK")

for page in article_pages:

    if (
        "free agency"
        in page["text"].lower()
    ):

        print(
            f"\nPDF Page "
            f"{page['page_number']}"
        )

        position = (
            page["text"]
            .lower()
            .find("free agency")
        )

        start = max(
            0,
            position - 300
        )

        end = min(
            len(page["text"]),
            position + 1200
        )

        print(
            page["text"][
                start:end
            ]
        )



# OUTPUT FILES

print("\n")
print("OUTPUT FILES")

print(OUTPUT_JSON)
print(OUTPUT_TEXT)
print(OUTPUT_STRUCTURE)

print("\n")
print("STEP 8 COMPLETE")