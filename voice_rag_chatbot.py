# voice_rag_chatbot.py
"""
Voice-enabled RAG Chatbot: end-to-end pipeline.

Flow per audio query:
1. Call external ASR service -> Hindi text
2. Translate Hindi -> English via Sarvam 
3. Try to resolve a Wikipedia page for the English question .
   If found, fetch, clean, save .txt and chunk+embed -> update FAISS.
4. Retrieve top-2 contexts from the (updated) vector DB.
5. Ask Gemini (GenAI) model using contexts and return answer.

Make sure:
- ASR service is running and reachable via ASR_URL.
- translate_sarvam.py from is present and importable.
- The files (search_wiki.py, scrape_wiki.py, save_text.py) are present.
- GEMINI_API_KEY and SARVAM_API_KEY set (or in .env; load_dotenv is used).

Environment variables (examples):
- ASR_URL (default: http://localhost:8000/transcribe)
- VECTOR_DB_DIR (default: ./vector_db)
- VECTOR_DB_PERSIST (bool: default True -> persist after updates)
- EMBEDDING_MODEL (default: all-MiniLM-L6-v2)
- SARVAM_API_KEY
- GEMINI_API_KEY
- GEMINI_MODEL (default gemini-3-flash-preview)

Run:
    pip install -r requirements.txt
    python -m uvicorn voice_rag_chatbot:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations
import os
import io
import json
import logging
import tempfile
import time
from typing import Optional, List

# load .env (if present)
from dotenv import load_dotenv
load_dotenv()

import requests
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Task-1 helpers (must be in same project)
try:
    from search_wiki import search_wikipedia
    from scrape_wiki import fetch_plain_extract, clean_wikipedia_text
    from save_text import save_to_txt
except Exception as e:
    raise ImportError("Task-1 files (search_wiki, scrape_wiki, save_text) must be present and importable.") from e

# LangChain / FAISS imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError("Missing langchain_text_splitters / langchain_community / langchain_huggingface. "
                      "Install correct packages from requirements.txt") from e

# translation helper (Task-4)
try:
    # keep previous convenience wrapper and also import the general function so we can translate the answer
    from translate_sarvam import translate_to_english_text, translate_to_english
except Exception as e:
    raise ImportError("translate_sarvam.py (Task-4) must be present and provide translate_to_english_text(...) and translate_to_english(...).") from e

# Gemini (GenAI SDK)
try:
    from google import genai
except Exception:
    raise ImportError("Missing google-genai package. Install from requirements.txt (google-genai)")

# Configuration (env vars)

ASR_URL = os.environ.get("ASR_URL", "http://localhost:8000/transcribe")
VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR", "./vector_db")
VECTOR_DB_PERSIST = os.environ.get("VECTOR_DB_PERSIST", "true").lower() in ("1", "true", "yes")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# small directory for scraped outputs
SCRAPED_OUTPUT_DIR = os.environ.get("SCRAPED_OUTPUT_DIR", "./task1_outputs")

# chunker defaults (same as Task-2)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_rag_chatbot")

app = FastAPI(title="Voice RAG Chatbot (E2E)", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Globals
faiss_store = None
embeddings = None

# Vector DB helpers
def init_embeddings():
    global embeddings
    if embeddings is None:
        logger.info("Initializing embedding model: %s", EMBEDDING_MODEL)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return embeddings

def load_vector_db(persist_dir: str = VECTOR_DB_DIR):
    """
    Load an existing FAISS index if present. Otherwise leave faiss_store None (we'll create on first update).
    """
    global faiss_store, embeddings
    init_embeddings()
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
        try:
            faiss_store = FAISS.load_local(persist_dir, embeddings)
            logger.info("Loaded FAISS vector DB from %s", persist_dir)
        except Exception as e:
            logger.warning("Failed to load existing FAISS DB: %s (will create on-demand)", e)
            faiss_store = None
    else:
        logger.info("No existing FAISS DB at %s; will create on first scraped article.", persist_dir)
        faiss_store = None

def persist_vector_db(persist_dir: str = VECTOR_DB_DIR):
    global faiss_store
    if faiss_store is None:
        return
    try:
        os.makedirs(persist_dir, exist_ok=True)
        faiss_store.save_local(persist_dir)
        logger.info("Persisted FAISS DB to %s", persist_dir)
    except Exception as e:
        logger.exception("Failed to persist FAISS DB: %s", e)

def chunk_texts_and_add_to_vector_db(text: str, source: Optional[str] = None, persist: bool = True):
    """
    Chunk `text`, create embeddings, and add to the (possibly new) faiss_store.
    source: optional string (e.g., wikipedia title) added to metadata.
    """
    global faiss_store, embeddings
    if not text or not text.strip():
        return 0

    init_embeddings()

    # split into chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.split_text(text)
    if not docs:
        return 0

    texts = docs
    metadatas = [{"source": source or "wikipedia", "length": len(t)} for t in texts]

    # If no faiss_store yet, create one
    if faiss_store is None:
        logger.info("Creating new FAISS store with %d chunks...", len(texts))
        faiss_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    else:
        # Add new texts to existing store
        try:
            # LangChain FAISS wrappers commonly implement add_texts
            faiss_store.add_texts(texts, metadatas=metadatas)
        except Exception as e:
            logger.exception("Failed to add texts to existing FAISS store: %s", e)
            # As fallback, raise a clear error (do not silently fail)
            raise RuntimeError("Unable to add new vectors to FAISS store: confirm your LangChain version supports add_texts.") from e

    # persist if requested
    if persist and VECTOR_DB_PERSIST:
        persist_vector_db()

    return len(texts)

# Retrieval helpers

def retrieve_top_k(english_query: str, k: int = 2):
    """
    Retrieve top-k docs. If the FAISS store is missing, raises.
    """
    global faiss_store
    if faiss_store is None:
        raise RuntimeError("Vector DB not loaded/initialized yet.")
    # prefer similarity_search_with_score if available
    try:
        docs_scores = faiss_store.similarity_search_with_score(english_query, k=k)
        # docs_scores is list of (doc, score)
        docs = [d for d, s in docs_scores]
        scores = [s for d, s in docs_scores]
    except Exception:
        # fallback
        docs = faiss_store.similarity_search(english_query, k=k)
        scores = [None] * len(docs)
    contexts = [d.page_content for d in docs]
    return contexts, scores

# External service helpers

def call_asr_service(audio_bytes: bytes, filename: Optional[str] = None) -> dict:
    files = {"file": (filename or "upload.webm", io.BytesIO(audio_bytes))}
    try:
        resp = requests.post(ASR_URL, files=files, timeout=60)
    except Exception as e:
        logger.exception("ASR service call failed: %s", e)
        raise RuntimeError("Failed to contact ASR service") from e
    if resp.status_code != 200:
        logger.error("ASR returned HTTP %s: %s", resp.status_code, resp.text)
        raise RuntimeError(f"ASR service error HTTP {resp.status_code}")
    return resp.json()

def call_gemini_with_context(question_en: str, contexts: List[str]) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in environment.")
    system_prompt = ("You are an assistant that answers questions using the provided context snippets. "
                     "If the context does not contain the answer, say you don't know and offer a short suggestion.")
    context_text = "\n\n".join([f"Context {i+1}:\n{c.strip()}" for i, c in enumerate(contexts, start=1)]) if contexts else "No context available."
    prompt = f"{system_prompt}\n\n{context_text}\n\nQuestion (English): {question_en}\n\nAnswer concisely in English."

    try:
        client = genai.Client()
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        # response.text may be present depending on client; fallback to other attributes
        ans = getattr(response, "text", None) or (response.output if hasattr(response, "output") else str(response))
        ans = ans.strip()
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        raise RuntimeError(f"Gemini API error: {e}") from e

    return ans

# Utility: extract a wiki-style topic from a question

def extract_topic_from_question(english_question: str) -> str:
    """
    Heuristic extraction to convert a user's question into a plausible Wikipedia
    topic/title. This helps avoid searching the full question which often fails.
    Minimal heuristics:
     - remove common question words/phrases
     - if a short phrase remains, return title-cased phrase
     - otherwise pick top content words (stopword removal) and join first 3
    """
    if not english_question:
        return ""

    q = english_question.strip().lower()
    # common leading patterns to strip
    patterns = [
        "what is ", "what's ", "what are ", "what do ", "what does ", "what happened ",
        "what happens in ", "what happens ", "explain ", "define ", "tell me about ",
        "who is ", "who are ", "how does ", "how to ", "give me ", "describe "
    ]
    for p in patterns:
        if q.startswith(p):
            q = q[len(p):].strip()

    # remove trailing question mark
    if q.endswith("?"):
        q = q[:-1].strip()

    # If the remaining text is short, return title-cased
    words = q.split()
    if 0 < len(words) <= 6:
        return " ".join(w.capitalize() for w in words)

    # fallback: pick first 3 non-stopwords
    stopwords = {
        "the","in","on","of","and","to","a","an","for","with","by","from",
        "is","are","be","was","were","as","that","this","these","those","how","why"
    }
    content = [w for w in words if w not in stopwords]
    if not content:
        content = words
    chosen = content[:3]
    return " ".join(w.capitalize() for w in chosen)

# Startup: load or create vector DB / embeddings
@app.on_event("startup")
def startup_event():
    os.makedirs(SCRAPED_OUTPUT_DIR, exist_ok=True)
    init_embeddings()
    load_vector_db()

# UI 
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Voice RAG Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Inter, Arial, Helvetica, sans-serif; margin: 20px; background:#f7f9fc; color:#111; }
    .container { max-width: 900px; margin: auto; }
    .card { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(20,20,30,0.06); margin-bottom: 14px; }
    h1 { margin: 0 0 8px 0; font-size: 20px; }
    .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    button { padding: 8px 12px; border-radius:8px; border:none; cursor:pointer; background:#2563eb; color:white; }
    button.secondary { background:#e6eefc; color:#0b3c86; }
    textarea { width: 100%; height: 90px; border-radius:8px; padding:8px; border:1px solid #e6eefc; resize: vertical; }
    .small { font-size: 13px; color:#555; }
    .context { background:#f1f5f9; padding:10px; border-radius:8px; margin-bottom:8px; white-space:pre-wrap; }
    .loader { display:inline-block; width:18px; height:18px; border:3px solid #e6eefc; border-top-color:#2563eb; border-radius:50%; animation:spin 1s linear infinite; vertical-align:middle; margin-right:6px;}
    @keyframes spin { to { transform: rotate(360deg); } }
    .answer-grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    .answer-box { min-height: 100px; padding:8px; border-radius:8px; border:1px solid #e6eefc; background:#fff; }
    @media (max-width:700px) { .answer-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>Voice RAG Chatbot (end-to-end)</h1>
      <p class="small">Ask in Hindi by recording or uploading audio. The system will transcribe, translate, scrape Wikipedia (if available), update the vector DB, retrieve, and answer.</p>
      <div class="controls">
        <input type="file" id="fileInput" accept="audio/*">
        <button id="recordBtn">Record</button>
        <button id="stopBtn" class="secondary" disabled>Stop</button>
        <button id="sendBtn">Ask</button>
        <div id="status" style="margin-left:auto" class="small"></div>
      </div>
    </div>

    <div class="card">
      <h3>Transcripts</h3>
      <label>Hindi (ASR):</label>
      <textarea id="hindiText" readonly></textarea>
      <label>English (Translation):</label>
      <textarea id="englishText" readonly></textarea>
    </div>

    <div class="card">
      <h3>Retrieved Contexts (Top 2)</h3>
      <div id="contexts"></div>
    </div>

    <div class="card">
      <h3>Answer (English & Hindi)</h3>
      <div class="answer-grid">
        <div>
          <label>Answer (English):</label>
          <div id="answer_en" class="answer-box"></div>
        </div>
        <div>
          <label>Answer (Hindi):</label>
          <div id="answer_hi" class="answer-box"></div>
        </div>
      </div>
    </div>
  </div>

<script>
let recorder, chunks = [], recordedBlob = null;
const fileInput = document.getElementById('fileInput');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const sendBtn = document.getElementById('sendBtn');
const statusDiv = document.getElementById('status');

recordBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => { recordedBlob = new Blob(chunks, { type: chunks[0].type || 'audio/webm' }); statusDiv.innerHTML = 'Recorded'; };
    recorder.start();
    recordBtn.disabled = true; stopBtn.disabled = false; statusDiv.innerHTML = '<span class="loader"></span>Recording...';
  } catch (e) {
    alert('Microphone permission required.');
  }
};

stopBtn.onclick = () => {
  if (recorder && recorder.state !== 'inactive') {
    recorder.stop();
    recordBtn.disabled = false; stopBtn.disabled = true;
  }
};

async function askServer() {
  statusDiv.innerHTML = '<span class="loader"></span>Processing... this can take 10-30s for scraping+embedding';
  document.getElementById('answer_en').innerText = '';
  document.getElementById('answer_hi').innerText = '';
  document.getElementById('contexts').innerHTML = '';
  document.getElementById('hindiText').value = '';
  document.getElementById('englishText').value = '';

  const fd = new FormData();
  if (fileInput.files.length) {
    fd.append('file', fileInput.files[0]);
  } else if (recordedBlob) {
    fd.append('file', recordedBlob, 'recording.webm');
  } else {
    alert('Please upload or record audio first.');
    statusDiv.innerHTML = '';
    return;
  }

  try {
    const resp = await fetch('/chat', { method:'POST', body: fd });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(t || 'Server error');
    }
    const data = await resp.json();
    document.getElementById('hindiText').value = data.hindi_text || '';
    document.getElementById('englishText').value = data.english_text || '';
    const contexts = data.contexts || [];
    const ctxDiv = document.getElementById('contexts');
    if (contexts.length === 0) ctxDiv.innerHTML = '<div class="small">No contexts retrieved.</div>';
    else {
      contexts.forEach((c, i) => {
        const el = document.createElement('div');
        el.className = 'context';
        el.innerText = `Context ${i+1}:\n` + c;
        ctxDiv.appendChild(el);
      });
    }
    document.getElementById('answer_en').innerText = data.answer || '(no answer)';
    document.getElementById('answer_hi').innerText = data.answer_hindi || '';
    statusDiv.innerHTML = 'Done';
  } catch (err) {
    console.error(err);
    statusDiv.innerHTML = 'Error: ' + (err.message || err);
  }
}

sendBtn.onclick = askServer;
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


# Main chat route (E2E)

@app.post("/chat")
async def chat_endpoint(file: Optional[UploadFile] = File(None), request: Request = None):
    """
    1) read bytes
    2) ASR -> Hindi text
    3) Translate -> English
    4) Try Wikipedia search & scrape -> add to vector DB (always try)
    5) Retrieve top-2 contexts from (updated) vector DB
    6) Call Gemini -> answer, then translate answer to Hindi via Sarvam
    """
    # 1. Read bytes
    if file is not None:
        data = await file.read()
        filename = getattr(file, "filename", None)
    else:
        try:
            data = await request.body()
            filename = None
        except Exception:
            raise HTTPException(status_code=400, detail="No audio provided")

    if not data:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    # 2. ASR
    try:
        asr_resp = call_asr_service(data, filename=filename)
        hindi_text = asr_resp.get("text", "").strip()
    except Exception as e:
        logger.exception("ASR failure: %s", e)
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

    # 3. Translate
    try:
        english_text = translate_to_english_text(text=hindi_text, api_key=SARVAM_API_KEY)
    except Exception as e:
        logger.exception("Translation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    # 4. Wikipedia search & scrape (use a short topic extracted from english_text as query)
    wiki_title = None
    scraped_chunks = 0
    try:
        topic = extract_topic_from_question(english_text)
        # First attempt with extracted topic
        search_res = None
        if topic:
            search_res = search_wikipedia(topic, lang="en")
            if search_res:
                logger.info("Resolved topic '%s' -> article '%s'", topic, search_res.get("title"))

        # Fallback
        if search_res is None:
            search_res = search_wikipedia(english_text, lang="en")
            if search_res:
                logger.info("Resolved using full question -> article '%s'", search_res.get("title"))

        if search_res and search_res.get("title"):
            wiki_title = search_res["title"]
            # fetch article extract
            text_extract = fetch_plain_extract(wiki_title)
            if text_extract:
                clean_text = clean_wikipedia_text(text_extract)
                # save file for debug / record-keeping
                safe_name = wiki_title.replace(" ", "_")[:120]
                outfname = os.path.join(SCRAPED_OUTPUT_DIR, f"{int(time.time())}_{safe_name}.txt")
                save_to_txt(clean_text, outfname)
                # chunk+embed and add to vector DB
                scraped_chunks = chunk_texts_and_add_to_vector_db(clean_text, source=wiki_title, persist=True)
    except Exception as e:
        # don't block pipeline if scraping fails; log and continue
        logger.exception("Wikipedia scraping/update failed (continuing): %s", e)

    # 5. Retrieval (attempt)
    contexts = []
    try:
        contexts, scores = retrieve_top_k(english_text, k=2)
        if (not contexts or len(contexts) == 0) and scraped_chunks > 0:
            
            if wiki_title:
                logger.info("Retrying retrieval using wiki title after adding chunks: %s", wiki_title)
                contexts, scores = retrieve_top_k(wiki_title, k=2)
    except Exception as e:
        logger.warning("Retrieval failed (vector DB may be empty): %s", e)
        contexts = []

    # 6. LLM (Gemini)
    try:
        answer = call_gemini_with_context(english_text, contexts)
    except Exception as e:
        logger.exception("LLM failed: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")

    # Translate the final answer to Hindi using Sarvam
    answer_hindi = ""
    try:
        if SARVAM_API_KEY:
            # reuse translate_to_english but override target/source codes for EN->HI
            # increase max_input_chars for longer answers
            trans = translate_to_english(
                text=answer,
                api_key=SARVAM_API_KEY,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                max_input_chars=1000,
                truncate_if_long=True
            )
            answer_hindi = trans.get("translated_text", "") or ""
    except Exception as e:
        logger.warning("Failed to translate answer to Hindi via Sarvam: %s", e)
        answer_hindi = ""

    # Return final JSON
    return JSONResponse({
        "hindi_text": hindi_text,
        "english_text": english_text,
        "wiki_title": wiki_title,
        "added_chunks": scraped_chunks,
        "contexts": contexts,
        "answer": answer,
        "answer_hindi": answer_hindi,
    })
