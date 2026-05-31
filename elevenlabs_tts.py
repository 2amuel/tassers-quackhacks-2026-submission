import os

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play


API_KEY = "sk_e5a40034935a1c992d3b42e098f7b07befa48ec841720182"
API_KEY_ENV_VAR = "ELEVENLABS_API_KEY"
DEFAULT_VOICE_NAME = "George"
DEFAULT_MODEL_ID = "eleven_v3"


def get_api_key():
    api_key = API_KEY.strip() or os.getenv(API_KEY_ENV_VAR, "").strip()

    if not api_key or api_key == "paste-your-elevenlabs-api-key-here":
        raise RuntimeError(
            f"Paste your ElevenLabs API key into API_KEY or set {API_KEY_ENV_VAR}."
        )

    return api_key


def get_voice_name_or_id():
    voice = input(f"Enter ElevenLabs voice name [{DEFAULT_VOICE_NAME}]: ").strip()
    return voice or DEFAULT_VOICE_NAME


def resolve_voice_id(client, voice_name_or_id):
    voices_response = client.voices.get_all()
    voices = voices_response.voices
    search_text = voice_name_or_id.casefold()

    matching_ids = [
        voice for voice in voices if voice.voice_id.casefold() == search_text
    ]
    if len(matching_ids) == 1:
        return matching_ids[0].voice_id

    exact_name_matches = [
        voice for voice in voices if voice.name.casefold() == search_text
    ]

    if len(exact_name_matches) == 1:
        return exact_name_matches[0].voice_id

    if len(exact_name_matches) > 1:
        voice_names = ", ".join(
            f"{voice.name} ({voice.voice_id})" for voice in exact_name_matches
        )
        raise RuntimeError(
            f"More than one voice is named '{voice_name_or_id}'. Use one of these voice IDs instead: {voice_names}"
        )

    partial_name_matches = [
        voice for voice in voices if search_text in voice.name.casefold()
    ]

    if len(partial_name_matches) == 1:
        return partial_name_matches[0].voice_id

    if len(partial_name_matches) > 1:
        voice_names = ", ".join(
            f"{voice.name} ({voice.voice_id})" for voice in partial_name_matches
        )
        raise RuntimeError(
            f"'{voice_name_or_id}' matched more than one voice. Type more of the name or use a voice ID: {voice_names}"
        )

    available_voice_names = ", ".join(sorted(voice.name for voice in voices))
    raise RuntimeError(
        f"No ElevenLabs voice containing '{voice_name_or_id}' was found.\n"
        f"Available voices: {available_voice_names}"
    )


def main():
    sentence = input("Enter a sentence to turn into speech: ").strip()

    if not sentence:
        print("No sentence entered. Exiting.")
        return

    voice_name_or_id = get_voice_name_or_id()

    client = ElevenLabs(api_key=get_api_key())
    voice_id = resolve_voice_id(client, voice_name_or_id)

    audio = client.text_to_speech.convert(
        text=sentence,
        voice_id=voice_id,
        model_id=DEFAULT_MODEL_ID,
        output_format="mp3_44100_128",
    )

    audio_bytes = b"".join(audio)
    play(audio_bytes)


if __name__ == "__main__":
    main()
