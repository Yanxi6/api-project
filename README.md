# Titanic Model API (FastAPI)

## What it does
A small FastAPI service that loads a trained Titanic model and exposes a `/predict` endpoint.

## Local run
```bash
python3 -m pip install -r requirements.txt
uvicorn src.app:app --reload --port 8000
