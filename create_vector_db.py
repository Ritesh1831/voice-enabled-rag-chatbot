# create_vector_db.py
"""
Create or update a FAISS vector database from a scraped Wikipedia article.

Skips re-indexing if the source has already been added (checked via metadata file).

Usage:
    python create_vector_db.py --input ./outputs/albert_einstein.txt --output_dir ./vector_db/
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Optional
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError(
        "Missing packages. Install with:\n"
        "  pip install langchain langchain-community langchain-huggingface "
        "sentence-transformers faiss-cpu\n\n"
        f"Original error: {e}"
    )


def load_text_file(path: str) -> Optional[str]:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def already_indexed(source_name: str, meta_path: Path) -> bool:
    """Return True if this source was previously embedded into the index."""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return any(entry.get("source") == source_name for entry in meta)
    except Exception:
        return False


def make_chunks(text: str, chunk_size: int, chunk_overlap: int, source_name: Optional[str]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    prefix = f"{source_name}__" if source_name else ""
    return [{"id": f"{prefix}{i}", "text": d} for i, d in enumerate(splitter.split_text(text))]


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_or_create_faiss(persist_dir: str, embeddings) -> FAISS:
    p = Path(persist_dir)
    if p.exists() and (p / "index.faiss").exists():
        try:
            return FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Warning: could not load existing index ({e}); creating fresh.")
    return FAISS.from_texts(["__init__"], embeddings)


def main():
    parser = argparse.ArgumentParser(description="Create / update FAISS vector DB")
    parser.add_argument("--input",         "-i", required=True)
    parser.add_argument("--output_dir",    "-o", default="./vector_db")
    parser.add_argument("--chunk_size",    type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--model",         default="all-MiniLM-L6-v2")
    parser.add_argument("--force",         action="store_true", help="Re-index even if source already exists")
    args = parser.parse_args()

    text = load_text_file(args.input)
    if not text:
        print("Input is empty or missing — nothing to embed.")
        return

    source_name = Path(args.input).stem
    out         = Path(args.output_dir)
    meta_path   = out / "chunks_metadata.json"

    if not args.force and already_indexed(source_name, meta_path):
        print(f"Skipping '{source_name}' — already in the index. Use --force to re-index.")
        return

    chunks = make_chunks(text, args.chunk_size, args.chunk_overlap, source_name)
    if not chunks:
        print("No chunks produced.")
        return

    emb   = get_embeddings(args.model)
    store = load_or_create_faiss(args.output_dir, emb)
    store.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[{"chunk_id": c["id"], "source": source_name} for c in chunks],
    )

    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))

    existing = []
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    new_meta = [{"id": c["id"], "length": len(c["text"]), "source": source_name} for c in chunks]
    meta_path.write_text(json.dumps(existing + new_meta, indent=2), encoding="utf-8")

    print(f"Done — added {len(chunks)} chunks from '{source_name}' into '{args.output_dir}'.")


if __name__ == "__main__":
    main()