# create_vector_db.py
"""
create_vector_db.py

Creates or updates a vector database for a scraped Wikipedia article using LangChain + SentenceTransformers + FAISS.

Usage:
    python create_vector_db.py --input ./task1_data_collection/outputs/albert_einstein.txt \
                               --output_dir ./vector_db/ \
                               --chunk_size 1000 --chunk_overlap 200

Notes:
- Chunk IDs are now namespaced with the source filename stem to avoid collisions.
- When saving metadata, existing metadata (if any) is preserved and appended.
"""

from __future__ import annotations
import os
import argparse
import json
from typing import List, Dict, Optional
from pathlib import Path

# --- Imports with friendly error messages ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError(
        "Missing required packages. Install with:\n"
        "pip install langchain sentence-transformers faiss-cpu\n\n"
        f"Original error: {e}"
    )


def load_text_file(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return text if text else None


def make_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] | None = None,
    source_name: Optional[str] = None,
) -> List[Dict]:
    """
    Create chunks using RecursiveCharacterTextSplitter.

    Returns list of dicts: {"id": idx_or_namespaced, "text": chunk_text}
    """
    if not text:
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = splitter.split_text(text)
    chunks = []
    for i, d in enumerate(docs):
        if source_name:
            cid = f"{source_name}__{i}"
        else:
            cid = str(i)
        chunks.append({"id": cid, "text": d})
    return chunks


def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_or_create_faiss(persist_dir: str, embeddings) -> FAISS:
    persist_path = Path(persist_dir)
    if persist_path.exists():
        try:
            return FAISS.load_local(str(persist_path), embeddings)
        except Exception:
            # If loading fails, create a new empty store instead of crashing
            return FAISS.from_texts([], embeddings)
    return FAISS.from_texts([], embeddings)


def add_chunks_to_faiss(faiss_store: FAISS, chunks: List[Dict], source_name: Optional[str] = None) -> FAISS:
    """
    Add new chunks to existing FAISS index.
    """
    if not chunks:
        return faiss_store

    texts = [c["text"] for c in chunks]
    metadatas = [{"chunk_id": c["id"], "source": source_name} for c in chunks]
    faiss_store.add_texts(texts=texts, metadatas=metadatas)
    return faiss_store


def main():
    parser = argparse.ArgumentParser(description="Create or update vector DB for a scraped Wikipedia article")
    parser.add_argument("--input", "-i", required=True, help="Path to scraped .txt file")
    parser.add_argument("--output_dir", "-o", default="./vector_db", help="Directory to store FAISS index")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    text = load_text_file(args.input)
    if not text:
        print("Input text is empty or missing. Nothing to embed.")
        return

    source_name = Path(args.input).stem
    chunks = make_chunks(text, args.chunk_size, args.chunk_overlap, source_name=source_name)
    if not chunks:
        print("No chunks created. Skipping embedding.")
        return

    embeddings = get_embeddings(args.model)
    faiss_store = load_or_create_faiss(args.output_dir, embeddings)
    faiss_store = add_chunks_to_faiss(faiss_store, chunks, source_name=source_name)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(args.output_dir)

    # Merge metadata if exists
    meta_path = Path(args.output_dir) / "chunks_metadata.json"
    existing_meta = []
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = []

    new_meta = [{"id": c["id"], "length": len(c["text"]), "source": source_name} for c in chunks]
    merged = existing_meta + new_meta
    meta_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(f"Vector DB updated with {len(chunks)} chunks (source: {source_name}).")


if __name__ == "__main__":
    main()
