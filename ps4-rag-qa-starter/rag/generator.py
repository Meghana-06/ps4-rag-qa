from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
from rag.config import Settings

class Generator:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.gen_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.gen_model)

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = self._build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        output = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        intro = "Answer the question using ONLY the context. If the answer is not in the context, say you don't know.\n\n"
        ctx = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(contexts)])
        q = f"\n\nQuestion: {question}\nAnswer:"
        return intro + ctx + q
