# search_wiki.py
"""
Wikipedia article resolver using REST API + OpenSearch + word-level fallback.
Handles ASR mishearings (e.g. "Elber Tine Stein" → "Albert Einstein") via
progressive fallback strategies.
"""

import requests
import logging
from typing import Optional, Dict
from urllib.parse import quote

logger = logging.getLogger("search_wiki")

WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_OPENSEARCH   = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "Voice-RAG-Chatbot/3.0 (educational project)",
    "Accept":     "application/json",
}


def _rest_lookup(title: str, timeout: int) -> Optional[Dict]:
    """Direct REST title lookup — exact match only."""
    url = WIKI_REST_SUMMARY.format(quote(title.replace(" ", "_")))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("type", "").endswith("not_found"):
            return None
        resolved = data.get("title")
        if not resolved:
            return None
        return {"title": resolved, "pageid": data.get("pageid"), "summary": data.get("extract")}
    except requests.RequestException as e:
        logger.debug("REST lookup failed for '%s': %s", title, e)
        return None


def _opensearch_lookup(query: str, timeout: int) -> Optional[Dict]:
    """
    Wikipedia OpenSearch — handles partial and fuzzy queries.
    Returns the top suggestion's full REST result, or None.
    """
    params = {"action": "opensearch", "search": query, "limit": 3, "namespace": 0, "format": "json"}
    try:
        resp = requests.get(WIKI_OPENSEARCH, params=params, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        data   = resp.json()
        titles = data[1] if len(data) > 1 else []
        if not titles:
            return None
        logger.info("OpenSearch: '%s' → '%s'", query, titles[0])
        return _rest_lookup(titles[0], timeout)
    except Exception as e:
        logger.debug("OpenSearch failed for '%s': %s", query, e)
        return None


def search_wikipedia(query: str, lang: str = "en", timeout: int = 10) -> Optional[Dict]:
    """
    Resolve a query to a Wikipedia article using three strategies:
      1. Direct REST lookup (exact title match)
      2. OpenSearch fuzzy match on the full query
      3. OpenSearch on each individual word (longest words first)
         — catches ASR mishearings like "Elber Tine Stein" → tries "Stein" → Einstein
    """
    if not query or not query.strip():
        return None

    result = _rest_lookup(query.strip(), timeout)
    if result:
        return result

    result = _opensearch_lookup(query.strip(), timeout)
    if result:
        return result

    # Word-level fallback: try content words individually, longest first
    # (surnames and key nouns are usually longer than noise words)
    words = sorted(
        [w for w in query.strip().split() if len(w) > 3],
        key=len,
        reverse=True,
    )
    for word in words:
        result = _opensearch_lookup(word, timeout)
        if result:
            logger.info("Word-level fallback: '%s' → '%s'", word, result.get("title"))
            return result

    logger.info("Wikipedia: no article found for '%s'", query)
    return None