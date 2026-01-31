# asr_app.py
"""
FastAPI ASR service (language-specific).
"""

import os
import tempfile
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse

import torch
from transformers import pipeline
from pydub import AudioSegment

# Configuration
MODEL_ID = "ai4bharat/indicwav2vec-hindi"
TARGET_SAMPLE_RATE = 16000

# Logging & globals
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("asr_app")

asr_pipeline = None
device_name = "cpu"

# Utilities
def _get_suffix_from_filename(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return ".tmp"
    return "." + filename.split(".")[-1].lower()


def convert_upload_to_wav_file(in_bytes: bytes, filename_hint: Optional[str]) -> Optional[str]:
    """
    Convert uploaded audio bytes into 16kHz mono WAV file.
    Returns wav path or None on failure.
    """
    input_tmp = None
    output_tmp = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=_get_suffix_from_filename(filename_hint)) as f:
            f.write(in_bytes)
            input_tmp = f.name

        audio = AudioSegment.from_file(input_tmp)
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            output_tmp = f.name
            audio.export(output_tmp, format="wav")

        return output_tmp

    except Exception as e:
        logger.warning("Audio conversion failed: %s", e)
        return None

    finally:
        if input_tmp and os.path.exists(input_tmp):
            try:
                os.unlink(input_tmp)
            except Exception:
                pass


def transcribe_wav_file(wav_path: str) -> str:
    """
    Transcribe WAV file using ASR pipeline.
    """
    if asr_pipeline is None or not wav_path:
        return ""

    try:
        out = asr_pipeline(wav_path)
    except Exception as e:
        logger.warning("ASR inference failed: %s", e)
        return ""

    if isinstance(out, dict):
        return out.get("text", "") or ""
    if isinstance(out, str):
        return out

    try:
        return out[0].get("text", "")
    except Exception:
        return ""


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_pipeline, device_name

    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"
    if not MODEL_ID:
        logger.error("ASR_MODEL_ID not set in environment; ASR will be unavailable.")
        asr_pipeline = None
        yield
        return
    logger.info("Loading ASR model on %s", device_name)

    try:
        asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_ID,
            device=device,
            trust_remote_code=True,
        )
        logger.info("ASR pipeline ready.")
    except Exception as e:
        logger.error("ASR model failed to load: %s", e)
        asr_pipeline = None

    yield
    logger.info("ASR service shutting down")


# FastAPI app
app = FastAPI(title="ASR Service", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if asr_pipeline else "degraded",
        "model": MODEL_ID,
        "device": device_name,
    }


@app.post("/transcribe")
async def transcribe_endpoint(file: Optional[UploadFile] = File(None), request: Request = None):
    """
    RAG-safe ASR endpoint.
    Always returns JSON with 'text' field.
    """
    try:
        if file:
            data = await file.read()
            filename = file.filename
        else:
            data = await request.body()
            filename = None
    except Exception:
        return JSONResponse({"text": "", "model": MODEL_ID or ""})

    if not data:
        return JSONResponse({"text": "", "model": MODEL_ID or ""})

    wav_path = convert_upload_to_wav_file(data, filename)
    text = transcribe_wav_file(wav_path)

    if wav_path and os.path.exists(wav_path):
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    return JSONResponse({
        "text": text,
        "model": MODEL_ID or "",
    })

