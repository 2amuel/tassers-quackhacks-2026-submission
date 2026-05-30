import os
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play


API_KEY = "sk_e5a40034935a1c992d3b42e098f7b07befa48ec841720182"
API_KEY_ENV_VAR = "ELEVENLABS_API_KEY"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_MODEL_ID = "eleven_v3"
OUTPUT_FILE = Path("speech.mp3")


def get_api_key():
    api_key = API_KEY.strip() or os.getenv(API_KEY_ENV_VAR, "").strip()

    if not api_key or api_key == "paste-your-elevenlabs-api-key-here":
        raise RuntimeError(
            f"Paste your ElevenLabs API key into API_KEY or set {API_KEY_ENV_VAR}."
        )

    return api_key


def get_output_file():
    filename = input(f"Enter output filename [{OUTPUT_FILE.name}]: ").strip()

    if not filename:
        return OUTPUT_FILE

    output_file = Path(filename)
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".mp3")

    return output_file


def main():
    sentence = input("Enter a sentence to turn into speech: ").strip()

    if not sentence:
        print("No sentence entered. Exiting.")
        return

    output_file = get_output_file()

    client = ElevenLabs(api_key=get_api_key())

    audio = client.text_to_speech.convert(
        text=sentence,
        voice_id=DEFAULT_VOICE_ID,
        model_id=DEFAULT_MODEL_ID,
        output_format="mp3_44100_128",
    )

    audio_bytes = b"".join(audio)
    output_file.write_bytes(audio_bytes)
    print(f"Saved speech to {output_file.resolve()}")

    play(audio_bytes)


if __name__ == "__main__":
    main()
