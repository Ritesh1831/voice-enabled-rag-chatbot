# voice_rag_chatbot.py
"""
Voice-enabled RAG Chatbot — single-service, deployment-ready.

Pipeline per audio query:
  1. Groq Whisper  →  Hindi transcription  (no separate ASR server needed)
  2. Sarvam        →  English translation
  3. Wikipedia     →  scrape + chunk + embed  (skipped if title already indexed)
  4. FAISS         →  retrieve top-k contexts above similarity threshold
  5. Groq LLM      →  answer with conversation history + wiki hint
  6. Sarvam        →  translate answer back to Hindi

Environment variables (.env):
  GROQ_API_KEY        (required)
  SARVAM_API_KEY      (required)
  GROQ_MODEL          default: llama-3.3-70b-versatile
  VECTOR_DB_DIR       default: ./vector_db
  EMBEDDING_MODEL     default: all-MiniLM-L6-v2
  SCRAPED_OUTPUT_DIR  default: ./task1_outputs
  CHUNK_SIZE          default: 1000
  CHUNK_OVERLAP       default: 200
  RETRIEVAL_TOP_K     default: 3
  RETRIEVAL_THRESHOLD default: 1.2  (L2 distance; lower = stricter)
  RATE_LIMIT_MAX      default: 10 requests / 60 s per IP

Run:
  uvicorn voice_rag_chatbot:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from search_wiki import search_wikipedia
    from scrape_wiki import fetch_plain_extract, clean_wikipedia_text
    from save_text import save_to_txt
except Exception as e:
    raise ImportError("Task-1 helpers (search_wiki, scrape_wiki, save_text) must be importable.") from e

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError("LangChain packages missing — see requirements.txt.") from e

try:
    from translate_sarvam import translate_to_english_text, translate_long_text
except Exception as e:
    raise ImportError("translate_sarvam.py must export translate_to_english_text and translate_long_text.") from e

try:
    from groq import Groq as GroqClient
except Exception as e:
    raise ImportError("groq package missing — run: pip install groq") from e


# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY")
GROQ_MODEL          = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
SARVAM_API_KEY      = os.environ.get("SARVAM_API_KEY")
VECTOR_DB_DIR       = os.environ.get("VECTOR_DB_DIR", "./vector_db")
EMBEDDING_MODEL     = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SCRAPED_OUTPUT_DIR  = os.environ.get("SCRAPED_OUTPUT_DIR", "./task1_outputs")
CHUNK_SIZE          = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP       = int(os.environ.get("CHUNK_OVERLAP", "200"))
RETRIEVAL_TOP_K     = int(os.environ.get("RETRIEVAL_TOP_K", "3"))
RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "1.2"))
RATE_LIMIT_MAX      = int(os.environ.get("RATE_LIMIT_MAX", "10"))
RATE_LIMIT_WINDOW   = 60  # seconds

INDEXED_TITLES_FILE = os.path.join(VECTOR_DB_DIR, "indexed_titles.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_rag_chatbot")


# ── Globals ───────────────────────────────────────────────────────────────────
faiss_store:    Optional[FAISS] = None
embeddings:     Optional[HuggingFaceEmbeddings] = None
indexed_titles: set[str] = set()          # tracks which Wikipedia articles are in the DB
_conversation:  list[dict] = []           # rolling conversation history for Groq (demo-grade, not multi-user)
_rate_store:    dict = defaultdict(list)  # IP → list of request timestamps


# ── Rate limiting ─────────────────────────────────────────────────────────────
def is_rate_limited(ip: str) -> bool:
    now = time.time()
    reqs = _rate_store[ip]
    reqs[:] = [t for t in reqs if now - t < RATE_LIMIT_WINDOW]
    if len(reqs) >= RATE_LIMIT_MAX:
        return True
    reqs.append(now)
    return False


# ── Conversation history ──────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 3  # last 3 Q&A pairs sent to Groq for context

def get_history_messages() -> list[dict]:
    return _conversation[-(MAX_HISTORY_TURNS * 2):]

def update_history(question: str, answer: str):
    _conversation.append({"role": "user",      "content": question})
    _conversation.append({"role": "assistant", "content": answer})


# ── Indexed titles persistence ────────────────────────────────────────────────
def load_indexed_titles():
    global indexed_titles
    if os.path.exists(INDEXED_TITLES_FILE):
        try:
            indexed_titles = set(json.loads(open(INDEXED_TITLES_FILE).read()))
        except Exception:
            indexed_titles = set()

def save_indexed_titles():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    with open(INDEXED_TITLES_FILE, "w") as f:
        json.dump(list(indexed_titles), f)


# ── Embeddings & vector DB ────────────────────────────────────────────────────
def init_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return embeddings


def load_vector_db():
    global faiss_store
    init_embeddings()
    index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    if os.path.isdir(VECTOR_DB_DIR) and os.path.exists(index_path):
        try:
            faiss_store = FAISS.load_local(
                VECTOR_DB_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS DB from %s (%d indexed titles)", VECTOR_DB_DIR, len(indexed_titles))
        except Exception as e:
            logger.warning("Could not load FAISS DB: %s", e)
            faiss_store = None


def persist_vector_db():
    if faiss_store is None:
        return
    try:
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        faiss_store.save_local(VECTOR_DB_DIR)
    except Exception as e:
        logger.exception("Failed to persist FAISS DB: %s", e)


def add_article_to_vector_db(text: str, source: str) -> int:
    global faiss_store
    if not text.strip():
        return 0
    init_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    meta = [{"source": source, "length": len(t)} for t in chunks]

    if faiss_store is None:
        faiss_store = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=meta)
    else:
        faiss_store.add_texts(chunks, metadatas=meta)

    persist_vector_db()
    indexed_titles.add(source)
    save_indexed_titles()
    return len(chunks)


def retrieve_top_k(query: str, k: int = RETRIEVAL_TOP_K) -> tuple[list[str], list[float]]:
    if faiss_store is None:
        return [], []
    try:
        results = faiss_store.similarity_search_with_score(query, k=k)
        # Filter by L2 distance threshold — higher distance = less relevant
        filtered = [(doc, score) for doc, score in results if score <= RETRIEVAL_THRESHOLD]
        if not filtered:
            logger.info("All retrieved chunks below similarity threshold (threshold=%.2f)", RETRIEVAL_THRESHOLD)
        return [d.page_content for d, _ in filtered], [float(s) for _, s in filtered]
    except Exception as e:
        logger.warning("Retrieval error: %s", e)
        return [], []


# ── ASR via Groq Whisper ──────────────────────────────────────────────────────
def call_asr(audio_bytes: bytes, filename: Optional[str] = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    suffix = ".webm"
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        client = GroqClient(api_key=GROQ_API_KEY)
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(filename or f"audio{suffix}", f.read()),
                model="whisper-large-v3",
                language="hi",
                response_format="text",
            )
        # response_format="text" returns str directly
        if isinstance(result, str):
            return result.strip()
        return (getattr(result, "text", "") or "").strip()

    except Exception as e:
        raise RuntimeError(f"Groq Whisper ASR failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── LLM via Groq ─────────────────────────────────────────────────────────────
def call_groq(question_en: str, contexts: list[str], wiki_title: Optional[str] = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    context_block = (
        "\n\n".join(f"[Context {i+1}]\n{c.strip()}" for i, c in enumerate(contexts))
        if contexts else "No context available."
    )

    # Tell the LLM what article was retrieved when ASR may have mangled the name
    topic_hint = (
        f"Note: The user's question mentions '{question_en}'. "
        f"The closest Wikipedia article retrieved is about '{wiki_title}'. "
        f"Base your answer on the context below, which covers {wiki_title}.\n\n"
        if wiki_title else ""
    )

    system_prompt = (
        "You are a concise, accurate assistant that answers questions based on "
        "the provided context. If the context is insufficient, say so briefly. "
        "Never fabricate facts. Keep answers to 2-4 sentences."
    )
    user_message = (
        f"{topic_hint}"
        f"Context:\n{context_block}\n\n"
        f"Question: {question_en}"
    )

    # Include rolling conversation history so the LLM knows what was discussed
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history_messages())
    messages.append({"role": "user", "content": user_message})

    client = GroqClient(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        top_p=0.9,
    )
    return (response.choices[0].message.content or "").strip()


# ── Topic extraction ──────────────────────────────────────────────────────────
_STRIP_PREFIXES = [
    "what is ", "what's ", "what are ", "what was ", "what were ",
    "what do ", "what does ", "what happened to ", "what happened in ",
    "explain ", "define ", "tell me about ", "give me information on ",
    "who is ", "who was ", "who are ", "who were ",
    "how does ", "how did ", "how to ", "describe ", "summarize ",
]
_STOPWORDS = {
    "the","in","on","of","and","to","a","an","for","with","by","from",
    "is","are","be","was","were","as","that","this","these","those",
    "how","why","when","where","which","please","can","you","me","i",
}

def extract_topic(question: str) -> str:
    if not question:
        return ""
    q = question.strip().lower().rstrip("?").strip()
    for p in _STRIP_PREFIXES:
        if q.startswith(p):
            q = q[len(p):].strip()
            break
    words = q.split()
    if 0 < len(words) <= 5:
        return " ".join(w.capitalize() for w in words)
    content = [w for w in words if w not in _STOPWORDS]
    return " ".join(w.capitalize() for w in (content or words)[:4])


# ── Startup validation ────────────────────────────────────────────────────────
def validate_env():
    for key in ("GROQ_API_KEY", "SARVAM_API_KEY"):
        if not os.environ.get(key):
            logger.warning("Missing env var: %s — dependent features will fail silently", key)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_env()
    os.makedirs(SCRAPED_OUTPUT_DIR, exist_ok=True)
    init_embeddings()
    load_indexed_titles()
    load_vector_db()
    logger.info("Voice RAG Chatbot ready | LLM: %s | indexed titles: %d", GROQ_MODEL, len(indexed_titles))
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice RAG Chatbot", version="3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── UI ────────────────────────────────────────────────────────────────────────
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Voice RAG Chatbot</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root {
  --bg:#0f1117;--surface:#1a1d27;--surface2:#22263a;--border:#2e3350;
  --accent:#6366f1;--accent2:#818cf8;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--radius:12px;
  --shadow:0 4px 24px rgba(0,0,0,.4);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

.layout{display:grid;grid-template-columns:240px 1fr;min-height:100vh}
@media(max-width:768px){.layout{grid-template-columns:1fr}}

.sidebar{background:var(--surface);border-right:1px solid var(--border);padding:20px 14px;display:flex;flex-direction:column;gap:18px}
@media(max-width:768px){.sidebar{display:none}}
.sidebar h2{font-size:14px;font-weight:700;letter-spacing:.5px}
.sidebar p{font-size:11px;color:var(--text3);margin-top:3px}

.step-list{display:flex;flex-direction:column;gap:5px;margin-top:4px}
.step{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;font-size:12px;color:var(--text3);transition:.2s}
.step.active{background:var(--surface2);color:var(--accent2)}
.step.done{color:var(--green)}
.step.error{color:var(--red);background:rgba(239,68,68,.08)}
.step-icon{width:20px;height:20px;border-radius:50%;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0}
.step.active .step-icon{background:var(--accent);color:#fff;animation:pulse 1.4s infinite}
.step.done .step-icon{background:var(--green);color:#fff}
.step.error .step-icon{background:var(--red);color:#fff}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

.model-badge{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:9px 11px;font-size:11px;color:var(--text2);line-height:1.8}
.model-badge span{color:var(--accent2);font-weight:600}

.main{padding:20px;display:flex;flex-direction:column;gap:16px;max-width:860px;margin:0 auto;width:100%}

.page-title{font-size:20px;font-weight:700;background:linear-gradient(135deg,var(--accent2),#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.page-sub{font-size:12px;color:var(--text3);margin-top:3px}

.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;box-shadow:var(--shadow)}
.card-title{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px}

.audio-row{display:flex;flex-wrap:wrap;gap:9px;align-items:center}
.file-label{display:flex;align-items:center;gap:7px;padding:8px 14px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;cursor:pointer;font-size:12px;color:var(--text2);transition:.2s}
.file-label:hover{border-color:var(--accent);color:var(--text)}
#fileInput{display:none}
.file-name{font-size:11px;color:var(--text3);align-self:center}

button{padding:8px 16px;border-radius:8px;border:none;cursor:pointer;font-size:12px;font-weight:600;transition:.15s}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover:not(:disabled){background:var(--accent2)}
.btn-secondary{background:var(--surface2);color:var(--text2);border:1px solid var(--border)}
.btn-secondary:hover:not(:disabled){border-color:var(--accent);color:var(--text)}
.btn-danger{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.btn-ask{padding:10px 24px;font-size:13px}
button:disabled{opacity:.4;cursor:not-allowed}

.rec-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--red);animation:blink .8s infinite;margin-right:4px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.1}}

textarea{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:9px 11px;color:var(--text);font-size:12px;resize:vertical;outline:none;font-family:inherit;line-height:1.6;transition:.2s}
textarea:focus{border-color:var(--accent)}
.label{font-size:11px;color:var(--text3);margin-bottom:5px;font-weight:500}

.two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:600px){.two-col{grid-template-columns:1fr}}

.ctx-wrap{display:flex;flex-direction:column;gap:8px}
.ctx-item{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:11px;font-size:11px;color:var(--text2);line-height:1.6;white-space:pre-wrap;max-height:110px;overflow-y:auto}
.ctx-item::-webkit-scrollbar{width:3px}
.ctx-item::-webkit-scrollbar-thumb{background:var(--border)}
.ctx-num{font-size:10px;font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}
.no-ctx{font-size:12px;color:var(--text3);font-style:italic}

.answer-box{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:13px;font-size:12px;line-height:1.75;min-height:80px;color:var(--text);white-space:pre-wrap}
.answer-box.hi{font-size:13px;line-height:1.9}

.wiki-badge{display:inline-flex;align-items:center;gap:6px;background:var(--surface2);border:1px solid var(--border);padding:3px 9px;border-radius:20px;font-size:10px;color:var(--text2);margin-top:7px}
.wiki-badge .dot{width:5px;height:5px;border-radius:50%;background:var(--green)}

#statusBar{font-size:11px;color:var(--text3);padding:5px 0;min-height:20px}
#statusBar.active{color:var(--accent2)}
#statusBar.done{color:var(--green)}
#statusBar.error{color:var(--red)}

.spin{display:inline-block;width:11px;height:11px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:rotate .7s linear infinite;vertical-align:middle;margin-right:5px}
@keyframes rotate{to{transform:rotate(360deg)}}

/* History panel */
.history-list{display:flex;flex-direction:column;gap:10px;max-height:320px;overflow-y:auto}
.history-list::-webkit-scrollbar{width:3px}
.history-list::-webkit-scrollbar-thumb{background:var(--border)}
.h-item{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:11px}
.h-q{color:var(--accent2);margin-bottom:4px;font-weight:500}
.h-a{color:var(--text2);line-height:1.6}
.h-clear{float:right;font-size:10px;padding:3px 8px;margin-top:-2px}
</style>
</head>
<body>
<div class="layout">

<aside class="sidebar">
  <div>
    <h2>🎙 VOICE RAG</h2>
    <p>Hindi audio → RAG → answer</p>
  </div>
  <div>
    <div class="card-title" style="margin-bottom:8px">Pipeline</div>
    <div class="step-list" id="pipelineSteps">
      <div class="step" id="s1"><div class="step-icon">1</div>ASR — Whisper</div>
      <div class="step" id="s2"><div class="step-icon">2</div>Translate → English</div>
      <div class="step" id="s3"><div class="step-icon">3</div>Wikipedia Scrape</div>
      <div class="step" id="s4"><div class="step-icon">4</div>Embed → Vector DB</div>
      <div class="step" id="s5"><div class="step-icon">5</div>Retrieve Contexts</div>
      <div class="step" id="s6"><div class="step-icon">6</div>Groq LLM Answer</div>
      <div class="step" id="s7"><div class="step-icon">7</div>Translate → Hindi</div>
    </div>
  </div>
  <div class="model-badge">
    LLM <span id="llmModel">—</span><br>
    ASR <span>whisper-large-v3</span><br>
    Embed <span>all-MiniLM-L6-v2</span>
  </div>
</aside>

<main class="main">
  <div>
    <div class="page-title">Voice RAG Chatbot</div>
    <div class="page-sub">Ask in Hindi — get answers grounded in Wikipedia</div>
  </div>

  <div class="card">
    <div class="card-title">Audio Input</div>
    <div class="audio-row">
      <label class="file-label" for="fileInput">📁 Choose file</label>
      <input type="file" id="fileInput" accept="audio/*"/>
      <button class="btn-secondary" id="recordBtn">⏺ Record</button>
      <button class="btn-secondary btn-danger" id="stopBtn" disabled>⏹ Stop</button>
      <button class="btn-primary btn-ask" id="sendBtn">Ask ›</button>
    </div>
    <div class="file-name" id="fileName">No file selected</div>
    <div id="statusBar"></div>
  </div>

  <div class="card">
    <div class="card-title">Transcripts</div>
    <div class="two-col">
      <div>
        <div class="label">Hindi (ASR)</div>
        <textarea id="hindiText" rows="3" readonly placeholder="Hindi transcription…"></textarea>
      </div>
      <div>
        <div class="label">English (Translated)</div>
        <textarea id="englishText" rows="3" readonly placeholder="English translation…"></textarea>
      </div>
    </div>
    <div id="wikiBadge" style="display:none" class="wiki-badge">
      <span class="dot"></span><span id="wikiTitle">—</span>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Retrieved Contexts</div>
    <div class="ctx-wrap" id="contexts">
      <div class="no-ctx">Contexts will appear after your query.</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Answer</div>
    <div class="two-col">
      <div>
        <div class="label">English</div>
        <div class="answer-box" id="answerEn">—</div>
      </div>
      <div>
        <div class="label">Hindi</div>
        <div class="answer-box hi" id="answerHi">—</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">
      History
      <button class="btn-secondary h-clear" id="clearHistory">Clear</button>
    </div>
    <div class="history-list" id="historyList">
      <div class="no-ctx">No previous queries yet.</div>
    </div>
  </div>
</main>
</div>

<script>
const $ = id => document.getElementById(id);
const STEPS = ['s1','s2','s3','s4','s5','s6','s7'];
const STEP_KEYS = ['asr','translate','wiki','embed','retrieve','llm','translate_back'];
let recorder = null, recChunks = [], recordedBlob = null;
let queryHistory = [];

// ── File input ────────────────────────────────────────────────────────────────
$('fileInput').onchange = () => {
  const f = $('fileInput').files[0];
  $('fileName').textContent = f ? f.name : 'No file selected';
  recordedBlob = null;
};

// ── Record ────────────────────────────────────────────────────────────────────
$('recordBtn').onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    recChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => recChunks.push(e.data);
    recorder.onstop = () => {
      recordedBlob = new Blob(recChunks, { type: recChunks[0]?.type || 'audio/webm' });
      $('fileName').textContent = 'Recording ready ✓';
      setStatus('Recording saved — click Ask ›', 'done');
    };
    recorder.start();
    $('recordBtn').disabled = true;
    $('stopBtn').disabled = false;
    $('fileName').innerHTML = '<span class="rec-dot"></span>Recording…';
    setStatus('Listening…', 'active');
  } catch { alert('Microphone permission required.'); }
};

$('stopBtn').onclick = () => {
  if (recorder && recorder.state !== 'inactive') {
    recorder.stop();
    recorder.stream.getTracks().forEach(t => t.stop());
  }
  $('recordBtn').disabled = false;
  $('stopBtn').disabled = true;
};

// ── Pipeline steps ────────────────────────────────────────────────────────────
function resetSteps() {
  STEPS.forEach((id, i) => {
    const el = $(id);
    el.className = 'step';
    el.querySelector('.step-icon').textContent = i + 1;
  });
}

function applySteps(stepsObj) {
  STEP_KEYS.forEach((key, i) => {
    const el = $(STEPS[i]);
    const val = stepsObj[key];
    el.className = 'step ' + (val === true ? 'done' : val === false ? 'error' : '');
    el.querySelector('.step-icon').textContent =
      val === true ? '✓' : val === false ? '✗' : i + 1;
  });
}

function setStatus(msg, cls = '') {
  const el = $('statusBar');
  el.className = cls;
  el.innerHTML = cls === 'active' ? `<span class="spin"></span>${msg}` : msg;
}

// ── History ───────────────────────────────────────────────────────────────────
function addToHistory(hindi, english, answer) {
  queryHistory.unshift({ hindi, english, answer });
  if (queryHistory.length > 8) queryHistory.pop();
  renderHistory();
}

function renderHistory() {
  const el = $('historyList');
  if (!queryHistory.length) {
    el.innerHTML = '<div class="no-ctx">No previous queries yet.</div>';
    return;
  }
  el.innerHTML = queryHistory.map(h => `
    <div class="h-item">
      <div class="h-q">Q: ${esc(h.hindi)} ${h.english ? '· ' + esc(h.english) : ''}</div>
      <div class="h-a">${esc(h.answer)}</div>
    </div>`).join('');
}

$('clearHistory').onclick = () => { queryHistory = []; renderHistory(); };

// ── Ask ───────────────────────────────────────────────────────────────────────
$('sendBtn').onclick = async () => {
  const blob = $('fileInput').files[0] || recordedBlob;
  if (!blob) { alert('Please upload or record audio first.'); return; }

  $('hindiText').value = '';
  $('englishText').value = '';
  $('answerEn').textContent = '—';
  $('answerHi').textContent = '—';
  $('contexts').innerHTML = '<div class="no-ctx">Retrieving…</div>';
  $('wikiBadge').style.display = 'none';
  resetSteps();
  $('sendBtn').disabled = true;
  setStatus('Processing…', 'active');
  $('s1').className = 'step active';
  $('s1').querySelector('.step-icon').textContent = '1';

  const fd = new FormData();
  fd.append('file', blob, blob.name || 'recording.webm');

  try {
    const resp = await fetch('/chat', { method: 'POST', body: fd });
    if (resp.status === 429) throw new Error('Rate limit reached — please wait a moment.');
    if (!resp.ok) throw new Error((await resp.text()) || `HTTP ${resp.status}`);
    const data = await resp.json();

    applySteps(data.steps || {});

    $('hindiText').value   = data.hindi_text   || '';
    $('englishText').value = data.english_text || '';

    if (data.wiki_title) {
      const label = data.already_indexed
        ? `${data.wiki_title} (cached)`
        : `${data.wiki_title} (${data.added_chunks || 0} chunks indexed)`;
      $('wikiTitle').textContent = label;
      $('wikiBadge').style.display = 'inline-flex';
    }

    const ctxDiv = $('contexts');
    if (data.contexts?.length) {
      ctxDiv.innerHTML = data.contexts.map((c, i) =>
        `<div class="ctx-item"><div class="ctx-num">Context ${i+1}</div>${esc(c)}</div>`
      ).join('');
    } else {
      ctxDiv.innerHTML = '<div class="no-ctx">No relevant contexts found — answer based on LLM knowledge.</div>';
    }

    $('answerEn').textContent = data.answer       || '—';
    $('answerHi').textContent = data.answer_hindi || '—';
    if (data.model) $('llmModel').textContent = data.model;

    addToHistory(data.hindi_text, data.english_text, data.answer);
    setStatus('Done ✓', 'done');

  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
  } finally {
    $('sendBtn').disabled = false;
  }
};

function esc(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "model":           GROQ_MODEL,
        "indexed_titles":  len(indexed_titles),
        "vector_db_ready": faiss_store is not None,
    }


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(
    file: Optional[UploadFile] = File(None),
    request: Request = None,
):
    ip = request.client.host if request else "unknown"
    if is_rate_limited(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")

    if file is not None:
        data     = await file.read()
        filename = getattr(file, "filename", None)
    else:
        try:
            data     = await request.body()
            filename = None
        except Exception:
            raise HTTPException(400, "No audio provided.")

    if not data:
        raise HTTPException(400, "Empty audio payload.")

    steps = {k: None for k in ("asr", "translate", "wiki", "embed", "retrieve", "llm", "translate_back")}

    # 1. ASR
    try:
        hindi_text  = call_asr(data, filename=filename)
        steps["asr"] = True
        logger.info("ASR: '%s'", hindi_text[:80])
    except Exception as e:
        steps["asr"] = False
        logger.exception("ASR failed: %s", e)
        raise HTTPException(500, f"ASR failed: {e}")

    # 2. Translate Hindi → English
    try:
        english_text = translate_to_english_text(text=hindi_text, api_key=SARVAM_API_KEY)
        steps["translate"] = bool(english_text)
    except Exception as e:
        steps["translate"] = False
        logger.exception("Translation failed: %s", e)
        english_text = ""

    # 3. Wikipedia scrape (skip if already indexed)
    wiki_title     = None
    scraped_chunks = 0
    already_indexed = False

    try:
        topic      = extract_topic(english_text)
        search_res = (search_wikipedia(topic, lang="en") if topic else None) \
                  or search_wikipedia(english_text, lang="en")

        if search_res and search_res.get("title"):
            wiki_title = search_res["title"]

            if wiki_title in indexed_titles:
                already_indexed = True
                steps["wiki"]  = True
                steps["embed"] = True
                logger.info("Skipping scrape — '%s' already indexed", wiki_title)
            else:
                raw_text = fetch_plain_extract(wiki_title)
                if raw_text:
                    clean_text = clean_wikipedia_text(raw_text)
                    safe_name  = wiki_title.replace(" ", "_")[:100]
                    save_to_txt(clean_text, os.path.join(SCRAPED_OUTPUT_DIR, f"{int(time.time())}_{safe_name}.txt"))
                    scraped_chunks = add_article_to_vector_db(clean_text, source=wiki_title)
                    steps["wiki"]  = True
                    steps["embed"] = scraped_chunks > 0
                    logger.info("Indexed %d chunks from '%s'", scraped_chunks, wiki_title)
                else:
                    steps["wiki"]  = False
                    steps["embed"] = False
    except Exception as e:
        steps["wiki"]  = False
        steps["embed"] = False
        logger.exception("Wikipedia step failed (continuing): %s", e)

    # 4. Retrieve
    contexts, _ = retrieve_top_k(english_text, k=RETRIEVAL_TOP_K)
    if not contexts and wiki_title:
        contexts, _ = retrieve_top_k(wiki_title, k=RETRIEVAL_TOP_K)
    steps["retrieve"] = len(contexts) > 0

    # 5. LLM
    try:
        answer = call_groq(english_text, contexts, wiki_title=wiki_title)
        update_history(english_text, answer)
        steps["llm"] = True
    except Exception as e:
        steps["llm"] = False
        logger.exception("Groq LLM failed: %s", e)
        raise HTTPException(500, f"LLM failed: {e}")

    # 6. Translate answer → Hindi
    answer_hindi = ""
    try:
        if SARVAM_API_KEY:
            answer_hindi = translate_long_text(
                text=answer,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                api_key=SARVAM_API_KEY,
            )
            steps["translate_back"] = bool(answer_hindi)
    except Exception as e:
        steps["translate_back"] = False
        logger.warning("Answer→Hindi translation failed: %s", e)

    return JSONResponse({
        "hindi_text":      hindi_text,
        "english_text":    english_text,
        "wiki_title":      wiki_title,
        "already_indexed": already_indexed,
        "added_chunks":    scraped_chunks,
        "contexts":        contexts,
        "answer":          answer,
        "answer_hindi":    answer_hindi,
        "model":           GROQ_MODEL,
        "steps":           steps,
    })