# Run the project locally

From the project folder, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Open [the dashboard](http://localhost:8501). Streamlit loads the saved models directly. This is one local Python application, with no separate backend to start.

## Replay a case

Select a DER configuration, condition and held-out SampleID, then click **Diagnose selected case**. The app displays detection, fault type, faulted bus, model scores, actual labels, B0/B1 comparison and a source-network schematic.

## Upload measurements

Use the **Upload measurements** tab. Download its CSV header template, then provide one row with all 3,604 released electrical columns. An optional `SampleID` column is stripped. JSON can be a measurement dictionary or an object containing a `measurements` dictionary.

Use volts, amperes and angles in degrees [-180,180], with both pre-event and post-event phasors. Measurements must match the released buses, line terminals and phase reference. Missing/extra columns, nonfinite data, negative magnitudes and zero voltages outside the verified training domain are rejected. Fault labels, resistance and scenario metadata are not predictors.

Normal predictions suppress fault type and bus. Type/location scores use validation temperature scaling; detection is an uncalibrated score. B1's network residual is a fit measure, not probability. The network diagram is a schematic of the verified source topology.

## Accuracy, time and reductions

The **Accuracy, time & reductions** tab shows the saved, executed single-case benchmark alongside the full held-out accuracy and error counts. LightGBM makes 74.0% fewer fault-type errors (50 to 13) and 55.6% fewer exact-bus errors (1,627 to 723) than B1. Its localization-stage median is 1.7066 ms versus B1's 9.5902 ms: 82.2% less computation time. B1 computes detection and classification faster. The table shows both median and p95 times for each task, plus the separately measured shared feature preparation cost.

The benchmark uses 110 distinct test cases, three passes, warm-up and four threads. Localization timing excludes predicted-type computation, which is reported separately. Stage timings exclude shared feature preparation, validation, model loading, acquisition and communication. They cannot establish physical fault-to-detection or breaker-clearing delay from steady-state phasors. See `reports/EFFICIENCY_AND_COST.md` for the protocol.

The same tab shows the sensor-count reduction and resulting accuracy/distance tradeoff. These are hardware counts, not measured financial savings. All rates and confidence scores in the dashboard appear as percentages; computation times remain in milliseconds and network distances in branch hops. Error reduction percentages and accuracy gains in percentage points are labeled separately.

## Required local files

Keep `artifacts/`, `config/` and `results/metrics.csv`. The comparison tab also reads `results/relative_error_reduction.csv`, `results/efficiency_timing_summary.csv`, `results/efficiency_benchmark_protocol.json` and `results/sensor_budget_tradeoff.csv`. Replay also needs `results/cache/raw.npy` and `results/cache/metadata.csv`. The existing `.venv` already contains the libraries. On a fresh machine, install `requirements.txt` into a Python 3.12 environment. The historical `requirements-lock.txt` records the original research environment and remains unchanged for the frozen experiment's provenance.

Stop Streamlit with **Ctrl+C** in its terminal. The app reuses saved models and does not retrain them.

This is a simulated research prototype. Only ten distinct normal test cases are available, and localization confidence remains imperfect. It is not certified electrical protection.
