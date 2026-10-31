# Solar-integrated distribution fault diagnosis — EECE 490

Reproducible comparison of LightGBM and conventional electrical methods for fault detection, ten-class fault classification and 164-bus localization on DER-Fault. Built for a third-year Electrical and Computer Engineering project at the American University of Beirut.

**Start with [the roadmap](PROJECT_ROADMAP.md) and [the executed feasibility audit](reports/DATASET_FEASIBILITY.md).** The audit found important restrictions on independent normal events, operating-state diversity and smaller PMU placements. Do not infer broad protection reliability from high simulation accuracy.

## Executed outcome

The [final performance report](reports/FINAL_PERFORMANCE_REPORT.md) contains actual held-out results, paired uncertainty, resistance/DER/sensor experiments and error cases.

| Held-out metric | B0 | B1 | LightGBM |
|---|---:|---:|---:|
| Detection recall | 66.08% | 100% | 100% |
| Fault-type accuracy | N/A | 98.61% | 99.64% |
| Exact-bus accuracy | N/A | 54.81% | 79.92% |
| Top-3 bus accuracy | N/A | 95.94% | 98.72% |
| Top-5 bus accuracy | N/A | 99.58% | 99.47% |

This is not uniform ML superiority: B1 has better top-5 location accuracy and smaller mean network-distance errors in the reduced-sensor experiments. Only ten distinct normal test cases remain after deduplication, so low-FPR claims are unsupported. B1 localization uses an approximate source circuit and has transformer-related ambiguities. The API contract checks and actual Streamlit replay test passed. Docker image build/run remains unverified because Docker is absent on this computer.

## Dataset and provenance

- [DER-Fault release 1.0](https://zenodo.org/records/22904220), by Fathima Razeeya Mohamed Razick and Petr Musilek.
- [Associated paper](https://arxiv.org/abs/2609.36247).
- [Original Iowa circuit source](https://wzy.ece.iastate.edu/Testsystem.html), associated with F. Bu et al., “A Time-series Distribution Test System based on Real Utility Data,” NAPS 2019.

The dataset README declares CC BY 4.0 for the authors' released materials. The circuit is a separately sourced model; attribute its authors and consult their reuse terms. Neither original dataset nor circuit archive is committed here. The 3.05 GB `dataverse_files.zip` initially in the folder is an unrelated IEEE 34-node HIF dataset and is not used.

Verified release: 19,800 cases; 3,604 electrical columns plus `SampleID`; 93 measured voltage buses; 60 labeled load conditions; six PV/EV/V2G scenarios; 11 classes including normal; fault resistance 0.05–24.999 Ω. Fault-only labels contain 164 buses. The expanded original OpenDSS model has 436 named buses, including secondary/source buses. All measured voltage nodes/current lines/fault buses match the source model.

## Environment

Use Python 3.12. Create a local virtual environment, then install the executed lock file:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The current workspace already has `.venv`. Its executable is the working interpreter; the system `python` alias pointed to the Windows Store and did not work. `requirements.txt` documents supported dependency families; `requirements-lock.txt` records the actual executed environment. OpenDSSDirect is needed for source-network construction, not for ordinary saved-model inference.

## Reproduce in order

Run from this repository root. Downloads need internet access and roughly 2.5 GB additional disk space for originals and caches; preserve the unrelated archive separately if desired.

The delivered directory already contains a frozen completed experiment. Use its saved models directly for inference. To **independently reproduce** the reported study, first make a separate checkout/copy, then remove `results/final_freeze.json` **only in that replica** before running the commands below. Preserve this original directory and its report. Reusing the documented split is appropriate for replication; further model development requires a new independent holdout and must not be presented as an untouched-test result.

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/audit_der.py
.\.venv\Scripts\python.exe scripts/topology.py
.\.venv\Scripts\python.exe scripts/prepare.py
.\.venv\Scripts\python.exe scripts/eda_der.py
.\.venv\Scripts\python.exe scripts/conventional.py
.\.venv\Scripts\python.exe scripts/train_der.py --experiment full
.\.venv\Scripts\python.exe scripts/experiment_baselines.py --experiment physical
.\.venv\Scripts\python.exe scripts/train_der.py --experiment physical
.\.venv\Scripts\python.exe scripts/experiment_baselines.py --experiment sensors
.\.venv\Scripts\python.exe scripts/train_der.py --experiment sensors
.\.venv\Scripts\python.exe scripts/experiment_baselines.py --experiment unseen
.\.venv\Scripts\python.exe scripts/train_der.py --experiment unseen
.\.venv\Scripts\python.exe scripts/evaluate_der.py
.\.venv\Scripts\python.exe scripts/explain_der.py
.\.venv\Scripts\python.exe scripts/plot_results.py
.\.venv\Scripts\python.exe scripts/verify_project.py
.\.venv\Scripts\python.exe scripts/verify_dashboard.py
.\.venv\Scripts\python.exe scripts/build_report.py
```

The split is frozen before electrical EDA so EDA does not guide choices using final test measurements. Training-only ANOVA screening is fitted separately for each model; deterministic phasor transformation itself has no learned parameters. Baselines are implemented and selected before ML. After all experiment variants are selected, evaluation hashes artifacts and opens the final test. Retuning is then refused. Run follow-up research in a new experiment directory with a new holdout; do not overwrite the completed frozen experiment to improve its reported scores.

See [preprocessing and baseline reasoning](reports/PREPROCESSING_AND_BASELINES.md) and [EDA interpretation](reports/EDA_DER.md). The final report is generated at `reports/FINAL_PERFORMANCE_REPORT.md`; all rows originate from executed predictions.

## Project structure

| Path | Purpose |
|---|---|
| `data/raw/der_fault/` | Verified downloaded release; ignored by Git |
| `data/raw/iowa/` | Original source circuit; ignored by Git |
| `scripts/` | Audit, split/features, baselines, training, evaluation, explanations and checks |
| `config/` | Feature schema, topology mapping, frozen sample split and sensor definitions |
| `artifacts/` | Saved model/selector bundles, B1 network, thresholds and temperatures |
| `results/` | Tuning history, metrics, paired uncertainty, predictions, error cases and confusion matrices |
| `results/cache/` | Original numeric and engineered caches; ignored by Git |
| `reports/` | Feasibility, engineering notes, EDA, plots and final results |
| `app/` | Shared inference engine, FastAPI backend and Streamlit dashboard |

## Run the prototype

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
# In another terminal:
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Open [API docs](http://127.0.0.1:8000/docs) and [dashboard](http://localhost:8501). Replay retained test cases, compare B0/B1/LightGBM, view actual labels and a source-topology schematic, or upload one complete measurement case. [API contract and units](reports/INFERENCE_API.md) explain the schema and confidence interpretation.

For Docker, run `docker compose up --build`. The replay cache is mounted read-only; raw datasets never enter the image. Docker availability/build verification is recorded separately in the roadmap, not inferred from the presence of a Dockerfile.

## Research scope and honest comparison

The split groups all scenarios of each load index and removes exact measurement duplicates. Final test: 3,600 faults and **only ten distinct normal events**. Shared pre-fault electrical states remain across different load labels. Normal pre/post phasors are exactly identical, so low-FPR detection and broad unseen-state generalization cannot be established.

B0 supports detection only. B1 uses voltage/current deviations, physical phase mapping and sequence rules, plus a source-circuit current-injection residual location method covering all 164 labeled buses. Its circuit is approximate and its free-current model can confuse transformer-related buses. Any ML localization advantage must be limited to this implementation. Scenario changes mix PV, EV and V2G, and resistance bins mix fault types/locations. Smaller published PMU lists are not nested in released measurements; experiments use available intersections of 36, 21 and 9 PMUs.

This is steady-state simulated decision support, **not certified protection**. It does not establish clearing time, noisy-event reliability, performance on another feeder or unseen fault buses. Probability scores do not represent field risk.

## Git checkpoints

Review and commit coherent milestones: (1) provenance/audit/split/EDA, (2) baselines/training/evaluation and actual results, (3) tested inference application/docs. Use explicit file paths when staging: the initial working tree had unrelated tracked deletions, which must not be staged automatically. Do not use `git add .` or `git add -A` in this inherited state. Large public data, archives, caches and binary model/network artifacts are ignored. Keep artifact hashes and settings in Git; distribute reproducible binary artifacts separately when needed. No remote push is required to run this project.
