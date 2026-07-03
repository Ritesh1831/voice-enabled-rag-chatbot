# main.py
"""
CLI orchestrator for Task-1: search -> fetch -> clean -> save.
"""

import argparse
import logging
import sys
import os
import time

from search_wiki import search_wikipedia
from scrape_wiki import fetch_plain_extract, clean_wikipedia_text
from save_text import save_to_txt


def make_filename_safe(title: str) -> str:
    safe = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
    return safe.replace(" ", "_").lower()


def run_pipeline(query: str, out_dir: str, filename: str = None, lang: str = "en"):
    search_result = search_wikipedia(query, lang=lang)
    if not search_result:
        return None

    title = search_result.get("title")
    extract = fetch_plain_extract(title)
    if not extract:
        return None

    clean_text = clean_wikipedia_text(extract)
    if not clean_text:
        return None

    out_name = filename if filename else make_filename_safe(title) + ".txt"
    out_path = os.path.join(out_dir, out_name)
    return save_to_txt(clean_text, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Task-1: Fetch Wikipedia article text")
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-o", "--out_dir", default="./outputs")
    parser.add_argument("-f", "--filename")
    parser.add_argument("-l", "--lang", default="en")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start = time.time()
    result = run_pipeline(args.query, args.out_dir, args.filename, args.lang)

    if result:
        logging.info("Saved: %s (%.2fs)", result, time.time() - start)
        print(result)
        sys.exit(0)
    else:
        logging.error("Wikipedia pipeline failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
