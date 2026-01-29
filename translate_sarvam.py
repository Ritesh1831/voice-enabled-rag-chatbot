"""
translate_sarvam.py

RAG-safe helper to translate text -> English using Sarvam Translate API.
Designed to fail softly inside an end-to-end pipeline.
"""

from __future__ import annotations
import os
import requests
from typing import Optional, Dict

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
API_KEY_HEADER = "api-subscription-key"

DEFAULT_MODEL = "mayura:v1"
DEFAULT_TARGET = "en-IN"
DEFAULT_MAX_INPUT_CHARS = 100


class SarvamError(Exception):
    pass


def translate_to_english(
    text: str,
    api_key: Optional[str] = None,
    source_language_code: str = "hi-IN",
    target_language_code: str = DEFAULT_TARGET,
    model: str = DEFAULT_MODEL,
    max_input_chars: Optional[int] = DEFAULT_MAX_INPUT_CHARS,
    truncate_if_long: bool = True,
    timeout_seconds: int = 15,
) -> Dict[str, Optional[str]]:
    """
    Translate text to English using Sarvam API.

    RAG-safe behavior:
    - Returns empty translation on failure
    - Never raises during runtime pipeline
    """

    if not text or not text.strip():
        return {
            "translated_text": "",
            "source_language_code": source_language_code,
            "request_id": None,
        }

    api_key = api_key or os.environ.get("SARVAM_API_KEY")
    if not api_key:
        return {
            "translated_text": "",
            "source_language_code": source_language_code,
            "request_id": None,
        }

    if max_input_chars is not None and len(text) > max_input_chars:
        if truncate_if_long:
            text = text[: max_input_chars - 3] + "..."
        else:
            return {
                "translated_text": "",
                "source_language_code": source_language_code,
                "request_id": None,
            }

    payload = {
        "input": text,
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
        "model": model,
    }

    headers = {
        API_KEY_HEADER: api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(
            SARVAM_TRANSLATE_URL,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException:
        return {
            "translated_text": "",
            "source_language_code": source_language_code,
            "request_id": None,
        }

    if resp.status_code != 200:
        return {
            "translated_text": "",
            "source_language_code": source_language_code,
            "request_id": None,
        }

    try:
        data = resp.json()
    except Exception:
        return {
            "translated_text": "",
            "source_language_code": source_language_code,
            "request_id": None,
        }

    translated_text = data.get("translated_text", "")
    return {
        "translated_text": translated_text or "",
        "source_language_code": data.get("source_language_code"),
        "request_id": data.get("request_id"),
    }


def translate_to_english_text(*, text: str, api_key: Optional[str] = None) -> str:
    """
    Convenience wrapper for RAG pipeline.
    Always returns a string.
    """
    result = translate_to_english(text=text, api_key=api_key)
    return result.get("translated_text", "")


# CLI behavior preserved
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text to English using Sarvam API")
    parser.add_argument("--text", "-t", required=True)
    parser.add_argument("--key", "-k")
    parser.add_argument("--no-truncate", action="store_true")
    args = parser.parse_args()

    res = translate_to_english(
        text=args.text,
        api_key=args.key,
        truncate_if_long=(not args.no_truncate),
    )

    if res["translated_text"]:
        print("Source language:", res.get("source_language_code"))
        print("Translated text:", res.get("translated_text"))
    else:
        print("Translation failed or returned empty output.")
