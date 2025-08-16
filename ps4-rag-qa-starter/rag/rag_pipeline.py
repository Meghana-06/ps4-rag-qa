import os, json
from typing import Optional, Dict, Any, List
from rag.config import Settings
from rag.retriever import search, build_faiss_index
from rag.generator import Generator
from rag.utils import chunk_text

class RAGPipeline:
    def __init__(self):
        self.cfg = Settings()
        self._gen = None

    def _lazy_gen(self):
        if self._gen is None:
            self._gen = Generator(self.cfg)
        return self._gen

    def index_exists(self) -> bool:
        return os.path.exists(self.cfg.faiss_index_path) and os.path.exists(self.cfg.meta_path)

    def build_or_rebuild_index(self):
        # simple ingest from data dir
        from rag.ingest import collect_docs
        docs = collect_docs(self.cfg.data_dir)
        chunks, meta = [], []
        for path, text in docs.items():
            for i, c in enumerate(chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)):
                chunks.append(c)
                meta.append({"source": path, "chunk_id": i})
        build_faiss_index(chunks, meta, self.cfg)

    def store_size(self) -> int:
        if not self.index_exists():
            return 0
        with open(self.cfg.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("chunks", []))

    def answer(self, question: str, top_k: Optional[int]=None) -> Dict[str, Any]:
        k = top_k or self.cfg.top_k
        hits = search(question, k, self.cfg)
        contexts = [h["text"] for h in hits]
        gen = self._lazy_gen()
        answer = gen.generate(question, contexts)
        return {
            "question": question,
            "answer": answer,
            "citations": [
                {"source": h["meta"]["source"], "chunk_id": h["meta"]["chunk_id"], "score": h["score"]}
                for h in hits
            ]
        }
