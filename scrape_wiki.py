# scrape_wiki.py
"""
Fetch and clean Wikipedia article text.

Fetch strategy (in order):
  1. MediaWiki action=query API — returns full article plain text (~50k chars)
  2. REST summary API           — intro paragraph only (fallback)
  3. HTML scrape                — first 5 paragraphs (last resort)
"""

import os
import re
import logging
import requests
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger("scrape_wiki")

WIKI_QUERY_API    = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_HTML_PAGE    = "https://en.wikipedia.org/wiki/{}"

WIKI_MAX_CHARS = int(os.environ.get("WIKI_MAX_CHARS", "50000"))

HEADERS = {
    "User-Agent": "Voice-RAG-Chatbot/3.0 (educational project)",
    "Accept":     "application/json",
}


def _fetch_full_article(title: str, timeout: int) -> Optional[str]:
    """MediaWiki query API — full plain-text article."""
    params = {
        "action":          "query",
        "titles":          title,
        "prop":            "extracts",
        "exintro":         False,
        "explaintext":     True,
        "exsectionformat": "plain",
        "format":          "json",
    }
    try:
        resp  = requests.get(WIKI_QUERY_API, params=params, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        pages = resp.json().get("query", {}).get("pages", {})
        if not pages:
            return None
        page  = next(iter(pages.values()))
        if page.get("missing") is not None:
            return None
        text = page.get("extract", "")
        if text:
            logger.info("MediaWiki API: fetched %d chars for '%s'", len(text), title)
        return text or None
    except Exception as e:
        logger.debug("MediaWiki API failed for '%s': %s", title, e)
        return None


def _fetch_rest_summary(title: str, timeout: int) -> Optional[str]:
    url = WIKI_REST_SUMMARY.format(quote(title.replace(" ", "_")))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        return resp.json().get("extract") if resp.status_code == 200 else None
    except Exception as e:
        logger.debug("REST summary failed for '%s': %s", title, e)
        return None


def _fetch_html_fallback(title: str, timeout: int) -> Optional[str]:
    url = WIKI_HTML_PAGE.format(quote(title.replace(" ", "_")))
    try:
        resp = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=timeout)
        if resp.status_code != 200:
            return None
        pars = re.findall(r"<p[^>]*>(.*?)</p>", resp.text, flags=re.S | re.I)

        def strip(html):
            t = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
            t = re.sub(r"<.*?>", "", t)
            return re.sub(r"\s+", " ", t).strip()

        clean = [strip(p) for p in pars if len(strip(p)) > 40][:5]
        return "\n\n".join(clean) if clean else None
    except Exception as e:
        logger.debug("HTML fallback failed for '%s': %s", title, e)
        return None


def fetch_plain_extract(title: str, timeout: int = 15) -> Optional[str]:
    if not title:
        return None
    text = (
        _fetch_full_article(title, timeout)
        or _fetch_rest_summary(title, timeout)
        or _fetch_html_fallback(title, timeout)
    )
    if not text:
        logger.warning("All fetch strategies failed for '%s'", title)
        return None
    if len(text) > WIKI_MAX_CHARS:
        text = text[:WIKI_MAX_CHARS]
    return text


def clean_wikipedia_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[[^\]]{0,20}\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()