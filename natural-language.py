import http.client
import json
import os
import sys
from pathlib import Path


API_KEY_ENV_VAR = "GEMINI_API_KEY"
HOST = "generativelanguage.googleapis.com"
MODEL = "gemini-3-flash-preview"
PATH = f"/v1beta/models/{MODEL}:generateContent"
QUIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}
TRANSLATION_CACHE = {}


def load_dotenv():
    env_file = Path(".env")

    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_api_key():
    load_dotenv()
    api_key = os.getenv(API_KEY_ENV_VAR, "").strip()

    if not api_key:
        raise RuntimeError(
            f"Set {API_KEY_ENV_VAR} in your environment or in a local .env file."
        )

    return api_key


def build_payload(asl_sentence):
    return json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"ASL->English. Answer only:\n{asl_sentence}",
                        }
                    ]
                }
            ],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": 0,
                "maxOutputTokens": 24,
                "responseMimeType": "text/plain",
                "thinkingConfig": {
                    "thinkingLevel": "minimal",
                },
            },
        },
        separators=(",", ":"),
    )


def translate_asl_to_english(connection, api_key, asl_sentence):
    cache_key = asl_sentence.lower()
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    connection.request(
        "POST",
        PATH,
        body=build_payload(asl_sentence),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    response = connection.getresponse()
    response_body = response.read().decode("utf-8")

    if response.status >= 400:
        raise RuntimeError(f"Gemini API request failed: {response_body}")

    result = json.loads(response_body)

    try:
        english_sentence = (
            result["candidates"][0]["content"]["parts"][0]["text"].strip()
        )
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected Gemini API response: {result}") from error

    TRANSLATION_CACHE[cache_key] = english_sentence
    return english_sentence


def main():
    print("Enter ASL grammar sentences. Type /quit or /exit to stop.\n")

    try:
        api_key = get_api_key()
    except RuntimeError as error:
        print(error)
        return 1

    connection = http.client.HTTPSConnection(HOST, timeout=10)

    try:
        while True:
            asl_sentence = input("ASL> ").strip()

            if asl_sentence.lower() in QUIT_COMMANDS:
                break

            if not asl_sentence:
                continue

            try:
                english_sentence = translate_asl_to_english(
                    connection,
                    api_key,
                    asl_sentence,
                )
            except RuntimeError as error:
                print(error)
                return 1

            print(f"English> {english_sentence}\n")
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
