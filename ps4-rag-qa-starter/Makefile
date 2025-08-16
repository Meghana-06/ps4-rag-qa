# Quality of life
.PHONY: install ingest run test clean

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ingest:
	python -m rag.ingest

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

clean:
	rm -rf artifacts/* __pycache__ .pytest_cache
