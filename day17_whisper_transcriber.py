
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError


# Load API key from the .env file
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".mp4", ".mpeg",
    ".mpga", ".m4a", ".ogg", ".webm", ".flac"
}

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


def initialize_client():
    """Initialize the Groq API client."""
    if not API_KEY:
        print("[ERROR] GROQ_API_KEY was not found.")
        print("Check that your .env file contains GROQ_API_KEY.")
        return None

    return Groq(api_key=API_KEY)


def validate_audio_file(file_path):
    """Validate the audio file before sending it to the API."""
    path = Path(file_path).expanduser()

    if not path.exists():
        print("[ERROR] File does not exist. Check the path.")
        return None

    if not path.is_file():
        print("[ERROR] The supplied path is not a file.")
        return None

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        print("[ERROR] Unsupported audio format.")
        print("Use WAV, MP3, MP4, M4A, FLAC, OGG, or WEBM.")
        return None

    if path.stat().st_size == 0:
        print("[ERROR] The audio file is empty.")
        return None

    if path.stat().st_size > MAX_FILE_SIZE:
        print("[ERROR] The file exceeds the 25 MB free-tier limit.")
        return None

    return path


def transcribe_audio(client, file_path):
    """Send the audio file to Groq Whisper and return its metadata."""
    path = validate_audio_file(file_path)

    if path is None:
        return False

    try:
        print("\n[PROCESSING] Sending audio to Groq Whisper API...")

        with path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(path.name, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                temperature=0.0
            )

        print("\n[SUCCESS] Transcription completed.")
        print("\n--- TRANSCRIBED TEXT ---")
        print(transcription.text)

        # Convert the complete response to a dictionary
        if hasattr(transcription, "model_dump"):
            result = transcription.model_dump()
        else:
            result = dict(transcription)

        print("\n--- SEGMENT METADATA ---")
        print(json.dumps(result.get("segments", []), indent=2))

        # Save the complete response for review and Day 18
        output_path = path.with_name(
            f"{path.stem}_transcription.json"
        )

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, ensure_ascii=False)

        print(f"\n[INFO] Full transcription saved to: {output_path}")
        return True

    except RateLimitError:
        print("[ERROR] Groq API rate limit reached.")
        print("Check your current usage limits and try again later.")
    except APIStatusError as error:
        print(f"[ERROR] Groq API returned HTTP {error.status_code}.")
        print(str(error))
    except APIConnectionError:
        print("[ERROR] Could not connect to Groq.")
        print("Check your internet connection and try again.")
    except Exception as error:
        print(f"[ERROR] Unexpected error: {error}")

    return False


def main():
    """Run the interactive transcription workflow."""
    print("Groq Whisper Transcription Module Initialized.")

    client = initialize_client()
    if client is None:
        return

    while True:
        file_path = input(
            "\nEnter an audio file path (or press Enter to exit): "
        ).strip().strip('"')

        if not file_path:
            print("Exiting Day 17 transcription module.")
            break

        if transcribe_audio(client, file_path):
            break


if __name__ == "__main__":
    main()