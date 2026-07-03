# translate_sarvam.py
"""
Translation helper using Sarvam AI — supports any Indian language pair.
RAG-safe: always returns a string, never raises during pipeline execution.
"""

from __future__ import annotations

import os
import re
import time
import logging
import requests
from typing import Optional, Dict

logger = logging.getLogger("translate_sarvam")

SARVAM_TRANSLATE_URL    = "https://api.sarvam.ai/translate"
API_KEY_HEADER          = "api-subscription-key"
DEFAULT_MODEL           = "mayura:v1"
DEFAULT_MAX_INPUT_CHARS = 1500
_RETRY_SLEEP_S          = 1.0


def translate_text(
    text: str,
    source_language_code: str,
    target_language_code: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_input_chars: Optional[int] = DEFAULT_MAX_INPUT_CHARS,
    truncate_if_long: bool = True,
    timeout_seconds: int = 20,
    retries: int = 1,
) -> Dict[str, Optional[str]]:
    """
    Translate text between any Sarvam-supported language pair.

    Supported codes: hi-IN  bn-IN  te-IN  ta-IN  mr-IN  gu-IN  kn-IN  ml-IN  pa-IN  en-IN
    Returns {"translated_text": str, "source_language_code": str, "request_id": str | None}.
    """
    _empty = {"translated_text": "", "source_language_code": source_language_code, "request_id": None}

    if not text or not text.strip():
        return _empty

    api_key = api_key or os.environ.get("SARVAM_API_KEY")
    if not api_key:
        logger.warning("SARVAM_API_KEY not set — skipping translation.")
        return _empty

    if max_input_chars is not None and len(text) > max_input_chars:
        if truncate_if_long:
            text = text[: max_input_chars - 3] + "..."
        else:
            logger.warning("Text too long (%d chars) — returning empty.", len(text))
            return _empty

    payload = {
        "input":                text,
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
        "model":                model,
    }
    headers = {API_KEY_HEADER: api_key, "Content-Type": "application/json", "Accept": "application/json"}

    for attempt in range(retries + 1):
        try:
            resp = requests.post(SARVAM_TRANSLATE_URL, json=payload, headers=headers, timeout=timeout_seconds)
        except requests.RequestException as e:
            logger.warning("Sarvam request error (attempt %d): %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(_RETRY_SLEEP_S)
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
                return {
                    "translated_text":      (data.get("translated_text") or "").strip(),
                    "source_language_code": data.get("source_language_code", source_language_code),
                    "request_id":           data.get("request_id"),
                }
            except Exception as e:
                logger.warning("Failed to parse Sarvam response: %s", e)
                return _empty

        if 400 <= resp.status_code < 500:
            logger.error("Sarvam client error %d: %s", resp.status_code, resp.text[:200])
            return _empty

        logger.warning("Sarvam HTTP %d (attempt %d)", resp.status_code, attempt + 1)
        if attempt < retries:
            time.sleep(_RETRY_SLEEP_S)

    return _empty


def translate_long_text(
    text: str,
    source_language_code: str,
    target_language_code: str,
    api_key: Optional[str] = None,
    chunk_size: int = 1200,
) -> str:
    """
    Translate arbitrarily long text by splitting on sentence boundaries.
    Useful for translating Groq answers that may exceed the single-call limit.
    """
    if not text:
        return ""
    if len(text) <= chunk_size:
        return translate_text(text, source_language_code, target_language_code, api_key,
                              max_input_chars=None).get("translated_text", "")

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    translated = []
    for chunk in chunks:
        result = translate_text(chunk, source_language_code, target_language_code,
                                api_key, max_input_chars=None)
        if result.get("translated_text"):
            translated.append(result["translated_text"])

    return " ".join(translated)


# ── Convenience wrappers ──────────────────────────────────────────────────────

def translate_to_english(
    text: str,
    api_key: Optional[str] = None,
    source_language_code: str = "hi-IN",
    target_language_code: str = "en-IN",
    **kwargs,
) -> Dict[str, Optional[str]]:
    return translate_text(text, source_language_code, target_language_code, api_key, **kwargs)


def translate_to_english_text(*, text: str, api_key: Optional[str] = None) -> str:
    return translate_to_english(text=text, api_key=api_key).get("translated_text", "")


def translate_to_hindi(
    text: str,
    api_key: Optional[str] = None,
    source_language_code: str = "en-IN",
    **kwargs,
) -> str:
    return translate_text(text, source_language_code, "hi-IN", api_key, **kwargs).get("translated_text", "")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text using Sarvam AI")
    parser.add_argument("--text", "-t", required=True)
    parser.add_argument("--src",  "-s", default="hi-IN")
    parser.add_argument("--tgt",  "-g", default="en-IN")
    parser.add_argument("--key",  "-k")
    args = parser.parse_args()

    res = translate_text(args.text, args.src, args.tgt, api_key=args.key)
    if res["translated_text"]:
        print(f"Source    : {res['source_language_code']}")
        print(f"Translated: {res['translated_text']}")
    else:
        print("Translation failed or returned empty output.")