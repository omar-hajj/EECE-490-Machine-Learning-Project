# Inference API

Start from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Interactive API documentation: `http://127.0.0.1:8000/docs`. The dashboard is served at `http://localhost:8501`. Run `docker compose up --build` for both services if Docker is installed. Models are loaded once, and inference never retrains them.

| Endpoint | Purpose |
|---|---|
| GET /health | Confirm model loading and expected measurement count |
| GET /schema | Exact required names and units |
| GET /cases?limit=50&offset=0 | Page through retained held-out replay cases; limit ≤200 |
| GET /replay/{SampleID} | Diagnose one retained final-test case and show actual labels |
| POST /predict | Diagnose an uploaded measurement dictionary |

`POST /predict` accepts `{"measurements": {"exact released column name": numeric_value, ...}}`, with all 3,604 electrical columns. The `...` denotes the remaining required columns, not literal valid JSON. Request field names other than `measurements` are rejected. Do not send fault labels, IDs, resistance or scenario metadata as predictors. The dashboard accepts one-row CSV files (optional `SampleID` is stripped) or equivalent JSON, and provides a header template.

Voltage magnitudes are V, currents A, angles degrees in [-180,180]. Pre- and post-event phasors must refer to the same network, physical measurement locations, terminal direction and phase reference as the release. Missing/extra channels, nonfinite data, negative magnitudes and zero voltages outside the training domain produce validation errors. Numeric inputs outside observed operating conditions may still be accepted; this is schema validation, not a guarantee of distributional validity.

Response objects include `lightgbm`, `b0`, `b1`, `inference_ms`, and confidence/scope notes. Normal predictions suppress type and bus outputs. Type/location confidence uses validation temperature scaling; detection is an uncalibrated score. B1's relative residual measures circuit fit and is **not** probability. The circuit picture is a verified source-topology schematic, not a geographic map or exact recreation of study-specific tap settings.

Replay requires read-only access to `results/cache/raw.npy` and `metadata.csv`. Upload inference needs only saved model/network artifacts and configuration. The API intentionally does not accept arbitrary filesystem paths. The full cache is excluded from Docker image context and mounted read-only for replay.

This prototype is local research decision support. The model is closed-set over 164 fault buses and is not suitable for autonomous switching or field protection decisions. Low-FPR reliability, transient response and sensor-noise robustness were not established by this release.
