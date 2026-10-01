# MLS Roster & Recruitment Intelligence

An AI-powered Retrieval-Augmented Generation (RAG) application for exploring Major League Soccer roster rules, player movement mechanisms, and Collective Bargaining Agreement provisions.

The system retrieves evidence from official MLS sources before using Gemini to generate grounded answers with source references.

## Current Knowledge Base

- 2026 MLS Roster Rules and Regulations
- MLS-MLSPA 2020–2028 Collective Bargaining Agreement
- Article 29: Player Movement Rules
- 93 structured retrieval chunks

## RAG Architecture

User Question

→ Semantic Retrieval using Sentence Transformers

→ TF-IDF Lexical Retrieval

→ Reciprocal Rank Fusion

→ Top-K Official Source Passages

→ Gemini Grounded Generation

→ Answer + Source Evidence

## Retrieval Stack

- Sentence Transformers
- all-MiniLM-L6-v2 embeddings
- FAISS vector search
- TF-IDF lexical retrieval
- Reciprocal Rank Fusion
- Gemini 3.5 Flash Lite
- Streamlit

## Retrieval Evaluation

Current benchmark:

| Metric | Result |
|---|---:|
| Hit Rate @ 1 | 87.5% |
| Hit Rate @ 3 | 100% |
| Mean Reciprocal Rank | 0.938 |

The current results are based on an initial 8-question retrieval benchmark and should be interpreted as an early evaluation rather than a large-scale benchmark.

## Example Questions

- Who is eligible for MLS free agency in 2026?
- How many U22 Initiative roster slots can a club have?
- Can General Allocation Money be used to buy down a Designated Player?
- What happens to an out-of-contract MLS player who is not eligible for free agency?

## Sources

The current knowledge base uses publicly available official information from:

- Major League Soccer
- MLS Players Association

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt