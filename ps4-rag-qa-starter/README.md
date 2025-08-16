# PS-4: Retrieval Augmented Generation (RAG) based Question & Answering System

A production-ready starter repo for a **RAG-based Q&A** system using **FastAPI** + **FAISS** + **Sentence-Transformers** + **FLAN‑T5** (Hugging Face).
It ingests local documents, builds an embedding index, retrieves top passages, and generates grounded answers with citations.

---

## ✨ Features
- 📄 Ingest plain text/markdown/PDF (via `pypdf`) from `./data`
- 🔎 FAISS vector store with `all-MiniLM-L6-v2` embeddings
- 🧠 Generator: `google/flan-t5-base` (works on CPU)
- 🧪 Minimal tests
- 🚀 FastAPI endpoints: `/health`, `/ingest`, `/ask`
- 🐳 Dockerfile + Makefile for one-command run
- 🔐 `.env` driven configuration

---

## 🗂️ Project Structure
```
ps4-rag-qa/
├── app/
│   └── main.py                # FastAPI app (ingest & ask)
├── rag/
│   ├── config.py              # Settings
│   ├── ingest.py              # Build FAISS index from ./data
│   ├── retriever.py           # Retrieve top-k chunks
│   ├── generator.py           # HF text-generation (FLAN‑T5)
│   └── rag_pipeline.py        # Orchestrates retrieve → generate
├── data/                      # Put your docs here
│   └── sample.txt
├── artifacts/                 # Saved FAISS index + metadata
├── tests/
│   └── test_rag.py
├── .vscode/settings.json      # Dev UX helpers
├── .env.example               # Copy to .env and adjust
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── LICENSE
└── README.md
```

---

## ⚙️ Setup (Local)

```bash
# 1) Python env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Create .env from template
cp .env.example .env

# 4) Ingest your data (reads ./data)
python -m rag.ingest

# 5) Run the API
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for Swagger UI.

---

## 🐳 Run with Docker

```bash
# Build
docker build -t ps4-rag-qa .

# Ingest (mount your local data dir)
docker run --rm -v "$PWD/data":/app/data -v "$PWD/artifacts":/app/artifacts ps4-rag-qa python -m rag.ingest

# Serve API
docker run --rm -p 8000:8000 -v "$PWD/artifacts":/app/artifacts ps4-rag-qa
```

Or use `docker-compose up --build` for auto-ingest + serve (see compose file).

---

## 🔌 API

### `GET /health`
- Returns basic health info.

### `POST /ingest`
- Rebuilds the FAISS index from files in `./data`.
- Body: none

### `POST /ask`
- Body:
```json
{ "question": "What is RAG?", "top_k": 4 }
```
- Returns generated answer + retrieved chunks (citations).

---

## 🧪 Tests
```bash
pytest -q
```

---

## 🔧 Config (.env)
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `GEN_MODEL=google/flan-t5-base`
- `CHUNK_SIZE=750`
- `CHUNK_OVERLAP=120`
- `TOP_K=4`

---

## 📌 Notes
- FLAN‑T5 is a seq2seq model; for larger contexts, switch to a chat LLM (e.g., `mistralai/Mistral-7B-Instruct`) and use a lightweight API or local model.
- For PDFs, ensure `pypdf` is installed and your documents are text-extractable (scanned PDFs require OCR like `pytesseract`).

---

## 📄 License
MIT
