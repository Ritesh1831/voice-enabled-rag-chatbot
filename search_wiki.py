# search_wiki.py
"""
Robust Wikipedia article resolver using REST API.

"""

import requests
import logging
from typing import Optional, Dict
from urllib.parse import quote

WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

HEADERS = {
    "User-Agent": "Voice-RAG-Chatbot/1.0 (contact: student@example.com)",
    "Accept": "student/json"
}


def search_wikipedia(query: str, lang: str = "en", timeout: int = 10) -> Optional[Dict]:
    """
    Try resolving query directly as a Wikipedia title using REST API.
    Returns dict with title if found, else None.
    """

    if not query or not query.strip():
        return None

    title_encoded = quote(query.replace(" ", "_"))
    url = WIKI_REST_SUMMARY.format(title_encoded)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None

        data = resp.json()

        # Page not found case
        if data.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
            return None

        return {
            "title": data.get("title"),
            "pageid": None,  # REST API does not expose pageid
            "summary": data.get("extract"),
        }

    except requests.RequestException as e:
        logging.warning("Wikipedia REST API request failed: %s", e)
        logging.info("Wikipedia lookup failed for exact title: %s", query)

        return None
