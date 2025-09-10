# Phase 1 — GPU Telemetry Logger + Flask API

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

## Run on Lightning.ai
Launch a Python workstation with a GPU (T4 is fine), then:
```bash
pip install -r requirements.txt
python app.py
```

If NVML is available, real GPU metrics will be logged. If not, the app falls back to a simulator.

## Env Vars
- `DB_PATH` (default: `telemetry.db`)
- `LOG_INTERVAL_SEC` (default: `5` seconds)
- `PORT` (default: `7860`)
- `HOST` (default: `0.0.0.0`)

## API
- `GET /health`
- `GET /metrics/latest` — latest row per GPU
- `GET /metrics/series?minutes=15&gpu_index=0&limit=500`

## Next Phases
- **Phase 2:** Job queue + schedulers (Random, Round Robin, FCFS)
- **Phase 3:** Q-learning scheduler
- **Phase 4:** Chart.js dashboard (live telemetry + performance comparison)
