from rag.utils import chunk_text

def test_chunk_text():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)
