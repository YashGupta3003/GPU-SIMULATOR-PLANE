# Scheduler Duration Logger + Flask API

This is the foundation for your AI GPU Control Plane project. It continuously logs duration,qtable metrics into SQLite and exposes a simple Flask API that the dashboard uses.

## Files
- `app.py` — Flask app + SQLite persistence
- `requirements.txt` — Python dependencies

## Run Locally
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python app.py
# App listens on 0.0.0.0:7860 by default
```
