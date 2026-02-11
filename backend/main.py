
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from statistics import mean, pstdev, stdev
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# PDF and DOCX parsers
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

MAX_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "as",
    "if", "then", "than", "so", "because", "while", "about", "into", "over", "under", "between", "through",
    "we", "you", "they", "he", "she", "i", "me", "my", "our", "your", "their", "them", "his", "her",
}

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
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


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


def repeated_line_ratio(text: str) -> float:
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return 0.0
    counts = Counter(lines)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / len(lines)


def paragraph_uniformity(text: str) -> float:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    lengths = [len(tokenize_words(p)) for p in paragraphs if p]
    if len(lengths) <= 1:
        return 0.0
    avg = mean(lengths)
    return 1.0 - min(1.0, pstdev(lengths) / (avg + 1e-6))


def stopword_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    stop_count = sum(1 for w in words if w in STOPWORDS)
    return stop_count / len(words)


def punctuation_entropy(text: str) -> float:
    punct_chars = [ch for ch in text if ch in ",;:-()\"'!?" ]
    if len(punct_chars) < 5:
        return 0.0
    counts = Counter(punct_chars)
    total = len(punct_chars)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * (0 if p <= 0 else math.log2(p))
    return entropy


def split_chunks(text: str, chunk_words: int = 280) -> List[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunks.append(" ".join(words[i : i + chunk_words]))
    return chunks


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
    line_repeat = repeated_line_ratio(text)
    para_uniform = paragraph_uniformity(text)
    stop_ratio = stopword_ratio(words)
    punct_ent = punctuation_entropy(text)

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

    # Repeated lines signal
    if line_repeat > 0.2:
        score += 8
        reasoning.append("Many lines are repeated or templated.")

    # Paragraph uniformity can indicate generated structure
    if para_uniform > 0.85:
        score += 7
        reasoning.append("Paragraph lengths are very uniform.")

    # Stopword balance tends to sit in a mid-range for natural writing
    if stop_ratio < 0.22 or stop_ratio > 0.68:
        score += 6
        reasoning.append("Function-word usage is unusual.")

    # Very low punctuation entropy can indicate repeated punctuation habits
    if 0 < punct_ent < 1.2:
        score += 5
        reasoning.append("Punctuation pattern is repetitive.")

    score = max(0, min(100, int(round(score))))
    return score, reasoning


def label_from_score(score: int) -> str:
    if score <= 39:
        return "Likely Human"
    if score <= 69:
        return "Unclear"
    return "Likely AI"


def aggregate_chunk_scores(text: str) -> Tuple[int, List[str]]:
    chunks = split_chunks(text)
    chunk_scores: List[int] = []
    merged_reasons: List[str] = []

    for chunk in chunks:
        score, reasons = compute_ai_score(chunk)
        chunk_scores.append(score)
        merged_reasons.extend(reasons[:2])

    avg_score = int(round(mean(chunk_scores)))
    if len(chunk_scores) > 1:
        variability = pstdev(chunk_scores)
        if variability > 20:
            avg_score = max(0, avg_score - 6)
            merged_reasons.append("Different sections have mixed writing patterns.")

    # Keep unique reasoning with stable order
    seen = set()
    unique_reasons = []
    for reason in merged_reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)

    if len(chunks) > 1:
        unique_reasons.insert(0, f"Document analyzed in {len(chunks)} text segments.")

    return avg_score, unique_reasons


def call_deepseek_text_classifier(text: str) -> Optional[Dict[str, object]]:
    if not DEEPSEEK_API_KEY:
        return None

    prompt = (
        "You are scoring whether a passage is likely AI generated. "
        "Return strict JSON with keys: score (0-100 int), confidence (0-1 float), label "
        "(Likely AI|Unclear|Likely Human), reasoning (array of <=3 short strings)."
    )
    user_text = text[:5000]

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
    }

    req = urllib.request.Request(
        url=f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        result = json.loads(content)
        score = int(result.get("score", 50))
        confidence = float(result.get("confidence", 0.5))
        label = str(result.get("label", label_from_score(score)))
        reasoning = result.get("reasoning", [])
        if not isinstance(reasoning, list):
            reasoning = [str(reasoning)]
        return {
            "score": max(0, min(100, score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "label": label,
            "reasoning": [str(r) for r in reasoning[:3]],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return None


def blended_score(text: str) -> Tuple[int, str, List[str], Dict[str, object]]:
    local_score, local_reasons = aggregate_chunk_scores(text)
    local_label = label_from_score(local_score)

    metadata: Dict[str, object] = {
        "local_score": local_score,
        "llm_used": False,
        "model": DEEPSEEK_MODEL if DEEPSEEK_API_KEY else None,
    }

    llm = call_deepseek_text_classifier(text)
    if not llm:
        return local_score, local_label, local_reasons, metadata

    llm_score = int(llm["score"])
    llm_confidence = float(llm["confidence"])
    llm_weight = 0.25 + (0.35 * llm_confidence)
    llm_weight = max(0.25, min(0.60, llm_weight))

    score = int(round((1.0 - llm_weight) * local_score + llm_weight * llm_score))
    label = label_from_score(score)

    metadata.update(
        {
            "llm_used": True,
            "llm_score": llm_score,
            "llm_confidence": llm_confidence,
            "blend_weight": round(llm_weight, 3),
        }
    )

    reasons = list(local_reasons)
    reasons.append("LLM-assisted calibration applied for final score.")
    for reason in llm.get("reasoning", []):
        reasons.append(f"LLM: {reason}")

    return score, label, reasons, metadata


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
        score, label, reasoning, meta = blended_score(text)
        preview_len = min(len(text), 1000)
        extracted_preview = text[:preview_len] if text else "No text could be extracted."

    response = {
        "ai_likelihood_score": score,
        "label": label,
        "reasoning": reasoning[:7],
        "extracted_preview": extracted_preview,
    }
    if not note:
        response["metadata"] = meta
    return response


@app.get("/health")
def health():
    return {"status": "ok"}
