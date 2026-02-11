
import io
import math
import os
import re
from collections import Counter
from statistics import mean, stdev
from typing import List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# PDF and DOCX parsers
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

MAX_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}

app = FastAPI(title="TrustCheck API", version="0.1.0")

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Normalize errors to {\"error\": \"message\"} so the frontend can display them.
    """
    if isinstance(exc.detail, dict):
        # Expect {"error": "..."} shape; fall back to string join otherwise.
        if "error" in exc.detail:
            msg = exc.detail["error"]
        else:
            msg = "; ".join(f"{k}: {v}" for k, v in exc.detail.items())
    else:
        msg = str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": msg})


def get_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def read_text_like(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def read_pdf(file_bytes: bytes) -> str:
    # pdfminer works with file-like objects
    with io.BytesIO(file_bytes) as buf:
        return pdf_extract_text(buf) or ""


def read_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as buf:
        doc = Document(buf)
        return "\n".join(p.text for p in doc.paragraphs)


def extract_text(file: UploadFile, file_bytes: bytes) -> Tuple[str, str]:
    """
    Return (text, note). Note is used for image fallback message.
    """
    ext = get_extension(file.filename or "")
    if ext in {".txt", ".md", ".csv", ".json"}:
        return read_text_like(file_bytes), ""
    if ext == ".pdf":
        return read_pdf(file_bytes), ""
    if ext == ".docx":
        return read_docx(file_bytes), ""
    if ext in {".png", ".jpg", ".jpeg"}:
        return "", "Image uploaded (no text extracted in MVP)."
    raise HTTPException(
        status_code=400,
        detail={"error": "Unsupported file type. Please upload txt, md, pdf, docx, png, jpg, jpeg."},
    )


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def split_sentences(text: str) -> List[str]:
    # Simple sentence split; strip empties
    parts = re.split(r"[.?!]", text)
    return [p.strip() for p in parts if p.strip()]


def repetition_rate(words: List[str]) -> float:
    if len(words) < 4:
        return 0.0
    shingles = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = Counter(shingles)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / max(len(shingles), 1)


def sentence_length_stats(sentences: List[str]) -> Tuple[float, float]:
    lengths = [len(s.split()) for s in sentences if s.split()]
    if not lengths:
        return 0.0, 0.0
    if len(lengths) == 1:
        return float(lengths[0]), 0.0
    return mean(lengths), stdev(lengths)


def punctuation_ratio(text: str) -> float:
    punct_chars = ",;:-()\"'"
    punct_count = sum(1 for ch in text if ch in punct_chars)
    return punct_count / max(len(text), 1)


def rare_word_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_ai_score(text: str) -> Tuple[int, List[str]]:
    """
    Combine heuristic signals into an integer 0-100 score.
    Returns score and list of reasoning bullet strings.
    """
    reasoning: List[str] = []
    words = tokenize_words(text)
    sentences = split_sentences(text)

    if len(text) < 200:
        reasoning.append("Not enough text to judge confidently.")
        return 50, reasoning

    rep_rate = repetition_rate(words)
    mean_len, stdev_len = sentence_length_stats(sentences)
    punct_ratio = punctuation_ratio(text)
    diversity = rare_word_ratio(words)
    burstiness = stdev_len / (mean_len + 1e-6) if mean_len > 0 else 0.0

    score = 50

    # Repetition weight
    if rep_rate > 0.2:
        score += 25
        reasoning.append("Some phrases repeat a lot.")
    elif rep_rate > 0.1:
        score += 15
        reasoning.append("Repeated 4-word phrases detected.")
    else:
        reasoning.append("Low phrase repetition.")

    # Sentence uniformity
    if stdev_len < 5 and mean_len > 0:
        score += 15
        reasoning.append("Many sentences have similar length.")
    elif burstiness > 0.8:
        score -= 10
        reasoning.append("Sentence lengths vary naturally.")

    # Punctuation ratio extremes
    if punct_ratio < 0.01 or punct_ratio > 0.2:
        score += 8
        reasoning.append("Punctuation usage looks uniform or extreme.")
    else:
        reasoning.append("Punctuation looks typical.")

    # Lexical diversity
    if diversity < 0.3:
        score += 15
        reasoning.append("Low lexical diversity (many repeated words).")
    elif diversity > 0.55:
        score -= 10
        reasoning.append("Good variety of words present.")

    # Burstiness adjustment
    if burstiness < 0.3 and mean_len > 0:
        score += 10
        reasoning.append("Writing style looks very uniform.")
    elif burstiness > 1.2:
        score -= 10
        reasoning.append("High variation between sentences.")

    score = max(0, min(100, int(round(score))))
    return score, reasoning


def label_from_score(score: int) -> str:
    if score <= 39:
        return "Likely Human"
    if score <= 69:
        return "Unclear"
    return "Likely AI"


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Size check by reading into memory cautiously
    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        return JSONResponse(
            status_code=413, content={"error": "File too large. Max 10MB."}
        )

    ext = get_extension(file.filename or "")
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unsupported file type. Please upload txt, md, pdf, docx, png, jpg, jpeg."},
        )

    try:
        text, note = extract_text(file, file_bytes)
    except HTTPException as exc:
        # Re-raise structured errors
        raise exc
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Could not read file."})

    if note:
        # Image path: no text extracted
        reasoning = ["This version does not read text from images."]
        score = 50
        label = "Unclear"
        extracted_preview = note
    else:
        score, reasoning = compute_ai_score(text)
        label = label_from_score(score)
        preview_len = min(len(text), 1000)
        extracted_preview = text[:preview_len] if text else "No text could be extracted."

    return {
        "ai_likelihood_score": score,
        "label": label,
        "reasoning": reasoning[:7],
        "extracted_preview": extracted_preview,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
