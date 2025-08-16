import faiss, os, json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rag.config import Settings

def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def build_faiss_index(chunks: List[str], meta: List[Dict], cfg: Settings) -> None:
    embedder = load_embedder(cfg.embedding_model)
    embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, cfg.faiss_index_path)
    with open(cfg.meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "chunks": chunks}, f, ensure_ascii=False)

def load_index(cfg: Settings) -> Tuple[faiss.IndexFlatIP, List[str], List[Dict], SentenceTransformer]:
    if not os.path.exists(cfg.faiss_index_path):
        raise FileNotFoundError("FAISS index not found. Run ingest.")
    index = faiss.read_index(cfg.faiss_index_path)
    with open(cfg.meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    meta = data["meta"]
    embedder = load_embedder(cfg.embedding_model)
    return index, chunks, meta, embedder

def search(query: str, top_k: int, cfg: Settings) -> List[Dict]:
    index, chunks, meta, embedder = load_index(cfg)
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q, top_k)
    hits = []
    for i, score in zip(idxs[0], scores[0]):
        i = int(i)
        hits.append({
            "score": float(score),
            "text": chunks[i],
            "meta": meta[i]
        })
    return hits
