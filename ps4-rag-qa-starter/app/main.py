from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag.rag_pipeline import RAGPipeline

app = FastAPI(title="PS-4: RAG Q&A", version="1.0.0")
rag = RAGPipeline()  # lazy loads models on first use

class AskRequest(BaseModel):
    question: str
    top_k: int | None = None

@app.get("/health")
def health():
    return {"status": "ok", "has_index": rag.index_exists(), "embedding_model": rag.cfg.embedding_model, "gen_model": rag.cfg.gen_model}

@app.post("/ingest")
def ingest():
    try:
        rag.build_or_rebuild_index()
        return {"status": "ingested", "num_chunks": rag.store_size()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(req: AskRequest):
    if not rag.index_exists():
        raise HTTPException(status_code=400, detail="Index not found. Run /ingest first.")
    try:
        result = rag.answer(req.question, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
