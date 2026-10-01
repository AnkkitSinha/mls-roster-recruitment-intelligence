import os
from google import genai


# GET API KEY

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )


# CREATE CLIENT

client = genai.Client(
    api_key=api_key
)


# TEST API

print("Testing Gemini API...")

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Reply with exactly: GEMINI API WORKING"
)

print(response.text)