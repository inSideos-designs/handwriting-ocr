# Handwriting OCR

A full-stack application for recognizing handwritten text from scanned documents.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development

```bash
# Run tests
pytest tests/ -v

# Start backend
uvicorn backend.main:app --reload

# Start frontend
cd frontend && npm run dev

# Start Dagster
dagster dev -m orchestration.definitions
```

## Docker

```bash
docker compose up --build
```
