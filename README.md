# Scheduler Duration Logger + Flask API

This is the foundation for your AI GPU Control Plane project. It continuously logs GPU metrics
(using NVML if available, otherwise simulated metrics) into SQLite and exposes a simple Flask API
that the dashboard will use later.

## Files
- `app.py` — Flask app + background telemetry logger + SQLite persistence
- `requirements.txt` — Python dependencies (installs NVML bindings)
- `telemetry.db` — created at runtime

## Run Locally
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python app.py
# App listens on 0.0.0.0:7860 by default
```
