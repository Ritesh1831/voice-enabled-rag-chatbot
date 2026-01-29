# Voice-Enabled RAG Chatbot (Hindi → English)

This repository contains an end-to-end **voice-enabled Retrieval Augmented Generation (RAG) chatbot**.  
The system allows users to ask questions in **Hindi via audio**, automatically transcribes the speech, translates it to English, dynamically scrapes relevant Wikipedia content, builds/updates a vector database, retrieves relevant context, and generates an answer using an LLM.  
The final answer is displayed in **both English and Hindi**.

---

## Project Overview

**Pipeline flow (per user query):**

1. User uploads or records a Hindi audio question via the UI  
2. Audio is transcribed using **AI4Bharat ASR (indicwav2vec)**  
3. Transcribed Hindi text is translated to English using **Sarvam Translation API**  
4. The English query is used to dynamically:
   - Search Wikipedia
   - Scrape relevant article content
   - Clean and store text
   - Chunk and embed content
   - Update a FAISS vector database
5. Top-2 relevant chunks are retrieved from the vector store
6. **Gemini (free API)** is used to generate the final answer using retrieved context
7. The answer is translated back to Hindi and shown in the UI

---

## Tech Stack

- **Backend:** FastAPI  
- **ASR:** AI4Bharat `indicwav2vec`  
- **Translation:** Sarvam Translate API  
- **Vector Store:** FAISS + SentenceTransformers  
- **LLM:** Google Gemini (free tier)  
- **Frontend:** HTML + JavaScript (browser audio recording)  
- **Embeddings:** `all-MiniLM-L6-v2`

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd <repo-name>


### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg (Required for ASR)

FFmpeg must be available in system PATH.

* **Windows:** Download from [https://ffmpeg.org](https://ffmpeg.org) and add to PATH
* **Linux:**

```bash
sudo apt install ffmpeg
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_sarvam_key_here
GEMINI_API_KEY=your_gemini_key_here
ASR_URL=http://localhost:8000/transcribe
```

> All API keys are read securely from `.env`.

---

## How to Run the Project

### Step 1: Start the ASR Service (Task-3)

```bash
python -m uvicorn asr_app:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
http://localhost:8000/health
```

---

### Step 2: Start the Voice RAG Chatbot

```bash
python -m uvicorn voice_rag_chatbot:app --host 0.0.0.0 --port 9000
```

Open in browser:

```
http://127.0.0.1:9000
```

---

## How to Use the UI

1. Record or upload a Hindi audio question
2. Click **Ask**
3. The UI will display:

   * Hindi transcription
   * English translation
   * Top-2 retrieved Wikipedia contexts
   * Final answer in **English and Hindi**

---

## Project Structure (Key Files)

```
.
├── asr_app.py                # AI4Bharat ASR FastAPI service
├── translate_sarvam.py       # Sarvam translation helper
├── search_wiki.py            # Wikipedia title resolver
├── scrape_wiki.py            # Wikipedia content scraper
├── save_text.py              # Utility to save cleaned text
├── create_vector_db.py       # Chunking + FAISS vector DB creation
├── voice_rag_chatbot.py      # End-to-end RAG chatbot (main app)
├── requirements.txt
├── README.md
```

---

## Observations

* Directly querying Wikipedia with full questions often fails; extracting a **clean topic from the question** significantly improves article resolution.
* Dynamically updating the vector database **per query** works well for exploratory questions but is computationally expensive.
* AI4Bharat ASR performs reliably for Hindi, even with casual speech.
* Translating both the **query and final answer** improves usability for non-English users.
* Separating ASR as an independent service keeps the RAG pipeline modular and scalable.

---

## Challenges Faced

* **FFmpeg availability issues** across terminals and environments (especially Windows + VS Code).
* Wikipedia REST API returns limited content (summary only), requiring careful chunking.
* Ensuring embeddings are added to FAISS before retrieval within the same request.
* Managing API token limits while translating long LLM responses.
* Handling cases where no relevant Wikipedia article exists without breaking the pipeline.

---

## Future Improvements

* Cache embeddings for frequently asked topics
* Add multilingual Wikipedia scraping
* Add streaming ASR and LLM responses
* Improve topic extraction using lightweight NLP instead of heuristics

---

## Author Notes

This project was implemented step-by-step by first validating each component independently
(ASR, Translation, Scraping, Vector DB, Retrieval) and then integrating them into a single
end-to-end RAG system.

The focus was on **robustness, clarity, and real-world usability** rather than shortcuts.

---

```
::contentReference[oaicite:0]{index=0}
```
