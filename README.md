# Voice-Enabled RAG Chatbot (Hindi → English)

An end-to-end **voice-enabled Retrieval Augmented Generation (RAG) chatbot**.  
Users ask questions in **Hindi via audio**. The system transcribes the speech, translates it to English, dynamically scrapes relevant Wikipedia content, builds and updates a vector database, retrieves relevant context, and generates a grounded answer using an LLM. The final answer is displayed in **both English and Hindi**.

🔗 **Live Demo:** [voice-enabled-rag-chatbot.onrender.com](https://voice-enabled-rag-chatbot.onrender.com)

---

<img width="1918" height="1026" alt="image" src="https://github.com/user-attachments/assets/f2f846dc-d8c8-4b2c-82f2-be2953ca20e1" />


---

## Pipeline

Each user query runs through this 7-step pipeline:

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **ASR** | Groq Whisper (`whisper-large-v3`) transcribes Hindi audio |
| 2 | **Translation** | Sarvam AI translates Hindi → English |
| 3 | **Wikipedia** | Relevant article is searched and scraped |
| 4 | **Embedding** | Article is chunked and embedded into FAISS (skipped if already indexed) |
| 5 | **Retrieval** | Top-3 semantically similar chunks are retrieved |
| 6 | **LLM** | Groq (`llama-3.3-70b-versatile`) generates a grounded answer with conversation history |
| 7 | **Translation** | Answer is translated back to Hindi via Sarvam AI |

---

## Tech Stack

- **Backend:** FastAPI
- **ASR:** Groq Whisper (`whisper-large-v3`)
- **Translation:** Sarvam AI Translate API
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Vector Store:** FAISS + SentenceTransformers
- **Embeddings:** `paraphrase-MiniLM-L3-v2`
- **Frontend:** HTML + JavaScript (browser audio recording)
- **Deployment:** Render (free tier)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ritesh1831/voice-enabled-rag-chatbot
cd voice-enabled-rag-chatbot
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
SARVAM_API_KEY=your_sarvam_key_here
GROQ_MODEL=llama-3.3-70b-versatile
VECTOR_DB_DIR=./vector_db
SCRAPED_OUTPUT_DIR=./task1_outputs
EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
```

### 5. Run the chatbot

```bash
uvicorn voice_rag_chatbot:app --host 0.0.0.0 --port 9000
```

Open in browser: `http://localhost:9000`

---

## How to Use

1. Record a Hindi audio question using the **Record** button, or upload an audio file
2. Click **Ask ›**
3. The UI displays:
   - Hindi transcription and English translation
   - Which Wikipedia article was found and indexed
   - Top retrieved context chunks
   - Final answer in English and Hindi
   - A scrollable history of previous queries

---

## Project Structure

```
.
├── voice_rag_chatbot.py      # Main app — end-to-end RAG pipeline + UI
├── translate_sarvam.py       # Sarvam translation helper (Hindi ↔ English)
├── search_wiki.py            # Wikipedia title resolver with fuzzy fallback
├── scrape_wiki.py            # Wikipedia content scraper (full article)
├── save_text.py              # Utility to save cleaned text to disk
├── create_vector_db.py       # CLI tool for chunking + FAISS index creation
├── main.py                   # CLI orchestrator for Wikipedia pipeline
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **Single service** — Groq handles both ASR and LLM, eliminating the need for a separate ASR microservice
- **Duplicate prevention** — already-indexed Wikipedia articles are skipped on repeat queries
- **Similarity threshold** — FAISS results below a relevance score are filtered out to reduce noise
- **Conversation history** — last 3 Q&A pairs are passed to the LLM for contextual follow-up
- **RAG-safe pipeline** — every step fails softly; the system always returns a response even if individual steps fail
- **Wiki title hint** — when ASR mishears a name, the LLM is told what article was actually retrieved so it answers correctly

---

## Observations

- Extracting a clean topic from the question (rather than searching the full question) significantly improves Wikipedia article resolution
- Groq Whisper handles Hindi speech well, including casual and accented speech
- Translating both the query and the final answer improves usability for non-English users
- Dynamically updating the vector DB per query works well for exploratory questions

---

## Challenges

- Forcing CPU-only PyTorch on deployment to stay within free tier memory limits (512MB)
- Wikipedia REST API returns summary only — full article requires the MediaWiki query API
- ASR mishearing foreign proper nouns (e.g. "Einstein" → "Einshtein") — handled via word-level Wikipedia search fallback and LLM wiki title hint
- `pydub` / `audioop` incompatibility with Python 3.13+ — replaced with direct `ffmpeg` subprocess call, then eliminated entirely by moving to Groq Whisper

---

## Author

Built step-by-step by first validating each component independently (ASR, translation, scraping, vector DB, retrieval) and then integrating them into a single end-to-end system. Focus was on robustness and real-world usability.
