# TrustCheck — Content Authenticity Checker (MVP)
# TrustCheck (Content Authenticity Checker)

Single-page MVP that estimates AI-likelihood of uploaded content using lightweight heuristics. No external AI services; everything runs locally.

## Stack
- Frontend: React + TypeScript (Vite), TailwindCSS
- Backend: FastAPI (Python) + Uvicorn
- No database; files processed in memory. CORS allows `http://localhost:5173`.

## Prerequisites
- Node.js 18+ and npm
- Python 3.10+

## Setup & Run

1) Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

2) Frontend (in a second terminal)
```bash
cd frontend
npm install
npm run dev -- --port 5173
```

3) Open the app  
Navigate to http://localhost:5173 and upload a file.

## Usage Notes
- Supported uploads: txt, md, csv, json, pdf, docx, png, jpg, jpeg
- Max size: 10MB (checked client- and server-side)
- Images are accepted but **no OCR in this MVP**; you’ll see a neutral “Unclear” result.
- Labels use likelihood language only: “Likely AI”, “Unclear”, “Likely Human”.


## Accuracy Upgrades (Hybrid Scoring)
- The backend now scores long documents in multiple text segments and aggregates section-level results.
- Additional linguistic signals are used (line repetition, paragraph uniformity, stopword balance, punctuation entropy).
- Optional **LLM calibration** is supported via DeepSeek API, blended with local heuristics for a more stable final score.

### Optional DeepSeek setup
Set these environment variables before running the backend:

```bash
export DEEPSEEK_API_KEY=your_key_here
# optional
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
```

If `DEEPSEEK_API_KEY` is not set, the app uses local scoring only.

## Quick Tests
- TXT: upload a small `.txt` to see preview and heuristic score.
- PDF/DOCX: upload to verify text extraction and preview (first 500–1000 chars).
- Image: upload `.jpg` to confirm score=50, label=Unclear, and reasoning notes missing text.

## Repo Structure
```
/backend
  main.py
  requirements.txt
/frontend
  index.html
  vite.config.ts
  package.json
  tsconfig.json
  tsconfig.node.json
  tailwind.config.ts
  postcss.config.js
  /src
    main.tsx
    App.tsx
    /components
      UploadCard.tsx
      ResultCard.tsx
    /lib
      api.ts
```

## API Contract
`POST http://localhost:8000/analyze` with `multipart/form-data` field `file`
```json
{
  "ai_likelihood_score": 0,
  "label": "Likely AI | Unclear | Likely Human",
  "reasoning": ["..."],
  "extracted_preview": "...",
  "metadata": {
    "local_score": 0,
    "llm_used": false
  }
}
```

Errors:
- 400 unsupported type → `{"error": "Unsupported file type. Please upload txt, md, pdf, docx, png, jpg, jpeg."}`
- 413 size limit → `{"error": "File too large. Max 10MB."}`
Single-page app to upload a file and get a heuristic AI-likelihood score. No data is stored; everything runs locally.

## Stack
- Frontend: React + TypeScript (Vite) with TailwindCSS
- Backend: FastAPI + Uvicorn (Python)
- No database; files processed in memory only.

## Prerequisites
- Python 3.10+ (virtualenv recommended)
- Node.js 18+ and npm

## Setup & Run
### Backend
```bash
cd backend
python -m venv .venv
./.venv/Scripts/activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev   # serves at http://localhost:5173
```

## Using the App
1. Start the backend server (port 8000).
2. Start the frontend dev server (port 5173).
3. Open http://localhost:5173 in your browser.
4. Upload a file (txt, md, csv, json, pdf, docx, png, jpg, jpeg). Max size 10MB.
5. Click **Analyze** to see the AI-likelihood score, label, reasoning bullets, and extracted preview.

## Acceptance Test Checklist
- Backend runs: `cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`
- Frontend runs: `cd frontend && npm install && npm run dev`
- Upload `.txt` → shows score/label/reasoning and text preview.
- Upload `.pdf` / `.docx` → text is extracted and previewed.
- Upload image (`.png/.jpg/.jpeg`) → returns score 50, label *Unclear*, reasoning notes that text is not read from images.

## Notes
- CORS is restricted to `http://localhost:5173` for local development.
- The scoring uses simple heuristics (no external AI). Results express likelihood, not certainty.
