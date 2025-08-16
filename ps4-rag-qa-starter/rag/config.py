from dataclasses import dataclass
import os

@dataclass
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    gen_model: str = os.getenv("GEN_MODEL", "google/flan-t5-base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 750))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    top_k: int = int(os.getenv("TOP_K", 4))
    data_dir: str = os.getenv("DATA_DIR", "data")
    art_dir: str = os.getenv("ART_DIR", "artifacts")
    faiss_index_path: str = os.path.join(art_dir, "index.faiss")
    meta_path: str = os.path.join(art_dir, "meta.json")
