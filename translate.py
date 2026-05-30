import html
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


API_KEY = "AIzaSyAXaSmwVEnEjShjTETCbACWlGwqvjUZId8"
TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

LANGUAGE_CODES = {
    "arabic": "ar",
    "chinese": "zh",
    "chinese simplified": "zh-CN",
    "chinese traditional": "zh-TW",
    "english": "en",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "tagalog": "tl",
    "vietnamese": "vi",
}


def normalize_language(language):
    clean_language = language.strip().lower()
    return LANGUAGE_CODES.get(clean_language, clean_language)


def translate_sentence(sentence, target_language):
    target_code = normalize_language(target_language)
    data = urllib.parse.urlencode(
        {
            "q": sentence,
            "target": target_code,
            "format": "text",
            "key": API_KEY,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        TRANSLATE_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google Translation API request failed: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach Google Translation API: {error.reason}"
        ) from error

    try:
        translated_text = result["data"]["translations"][0]["translatedText"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(
            f"Unexpected Google Translation API response: {result}"
        ) from error

    return html.unescape(translated_text)


def main():
    sentence = input("Enter a sentence to translate: ").strip()
    target_language = input(
        "Enter the language to translate to: "
    ).strip()

    if not sentence:
        print("Please enter a sentence to translate.")
        return 1

    if not target_language:
        print("Please enter a target language.")
        return 1

    try:
        translation = translate_sentence(sentence, target_language)
    except RuntimeError as error:
        print(error)
        return 1

    print(f"\nTranslation: {translation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
