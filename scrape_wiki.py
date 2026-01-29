# scrape_wiki.py
"""
Fetch and clean Wikipedia article text using REST API.
This avoids MediaWiki API which may return 403 on some networks.
If the REST summary is unavailable, we fall back to fetching the article HTML
and extracting the first paragraph(s) from the page.
"""

import requests
import logging
import re
from urllib.parse import quote
from typing import Optional

WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_HTML_PAGE = "https://en.wikipedia.org/wiki/{}"

HEADERS = {
    "User-Agent": "AI4Bharat-RAG-Assignment/1.0 (contact: applicant@example.com)",
    "Accept": "application/json"
}


def _strip_html_tags(html: str) -> str:
    # Remove tags and decode some HTML entities simply
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<.*?>", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_plain_extract(title: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch plain text summary of a Wikipedia page using REST API.
    If REST summary is not available, attempt a simple HTML scrape of the page
    and return the leading paragraph(s).
    Returns None on failure.
    """
    if not title:
        return None

    title_encoded = quote(title.replace(" ", "_"))
    url = WIKI_REST_SUMMARY.format(title_encoded)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract")
            if extract:
                return extract

        # If REST request returns 404/other or lacks extract, fall back to HTML fetch
    except requests.RequestException as e:
        logging.debug("REST summary request failed (will try HTML fallback): %s", e)

    # HTML fallback: fetch the wiki page and extract first meaningful <p> blocks
    html_url = WIKI_HTML_PAGE.format(title_encoded)
    try:
        resp = requests.get(html_url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=timeout)
        if resp.status_code != 200:
            logging.warning("Failed to fetch HTML page: %s (status %s)", html_url, resp.status_code)
            return None

        html = resp.text

        # Find content inside the main content area: paragraphs inside <p> tags.
        # This is a simple heuristic: look for <p>...</p> and take the first few non-empty ones.
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.S | re.I)
        clean_pars = []
        for p in paragraphs:
            txt = _strip_html_tags(p)
            # ignore very short or tag-only paragraphs (e.g., disambiguation boxes)
            if txt and len(txt) > 30:
                clean_pars.append(txt)
            if len(clean_pars) >= 3:
                break

        if not clean_pars:
            logging.warning("HTML fallback found no suitable paragraphs for: %s", title)
            return None

        # join a couple of intro paragraphs to form an extract
        return "\n\n".join(clean_pars)

    except requests.RequestException as e:
        logging.warning("HTML fetch failed for %s: %s", html_url, e)
        return None
    except Exception as e:
        logging.warning("Error parsing HTML fallback for %s: %s", html_url, e)
        return None


def clean_wikipedia_text(text: str) -> str:
    """
    Basic cleaning for RAG-ready text.
    """
    if not text:
        return ""

    # Remove citation artifacts if any
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
