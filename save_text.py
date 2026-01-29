# save_text.py
"""
Utility to save cleaned text to a .txt file with UTF-8 encoding.
Ensures parent directory exists.
"""

import os
import io
from typing import Optional


def save_to_txt(text: str, output_path: str, max_chars: Optional[int] = None) -> Optional[str]:
    """
    Save text to output_path (creates directories as needed).
    Returns absolute path, or None if text is empty.
    """
    if not text or not text.strip():
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if max_chars is not None and isinstance(max_chars, int):
        text = text[:max_chars]

    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return os.path.abspath(output_path)
