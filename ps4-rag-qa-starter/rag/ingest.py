import os, json, glob
from typing import List, Dict
from pypdf import PdfReader
from rag.config import Settings
from rag.retriever import build_faiss_index
from rag.utils import chunk_text

def read_textlike(path: str) -> str:
    if path.endswith(('.md', '.txt')):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    if path.endswith('.pdf'):
        text = []
        try:
            pdf = PdfReader(path)
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                text.append(extracted)
        except Exception as e:
            print(f"[WARN] Could not parse PDF {path}: {e}")
        return "\n".join(text)
    return ""

def collect_docs(data_dir: str) -> Dict[str, str]:
    paths = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    keep = [p for p in paths if p.lower().endswith(('.txt','.md','.pdf'))]
    store = {}
    for p in keep:
        store[p] = read_textlike(p)
    return store

def main():
    cfg = Settings()
    os.makedirs(cfg.art_dir, exist_ok=True)
    docs = collect_docs(cfg.data_dir)

    # chunk
    chunks = []
    meta = []
    for path, text in docs.items():
        for i, chunk in enumerate(chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)):
            chunks.append(chunk)
            meta.append({"source": path, "chunk_id": i})

    build_faiss_index(chunks, meta, cfg)

    print(f"[INGEST] {len(chunks)} chunks indexed → {cfg.faiss_index_path}")

if __name__ == "__main__":
    main()
