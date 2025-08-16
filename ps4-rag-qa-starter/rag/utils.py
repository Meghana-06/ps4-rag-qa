from typing import List

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c.strip()]
