# EECE 490 project roadmap

Machine Learning-Based Fault Detection, Classification, and Localization in Solar-Integrated Electrical Distribution Networks.

Status as of 2026-10-31. Every completion claim must reference executed outputs. No training is permitted before the Phase 1 audit passes. Final test predictions must be generated only after model and threshold selection is frozen.

| Phase | Objectives and tasks | Expected outputs | Status |
|---|---|---|---|
| 1 — Inspection and feasibility | Inventory local files; verify dataset provenance/checksums; join by SampleID; count classes/scenarios; audit missing, invalid and duplicate measurements; inspect PMUs, topology and independent simulation groups; assess all tasks and leakage | Reproducible audit, inventory, feasibility report, machine-readable counts | Complete: see reports/DATASET_FEASIBILITY.md, der_audit.json and topology_audit.json |
| 2 — EDA | Plot classes, detection balance, DER configurations, resistance and electrical signatures; inspect resistance and scenario dependence using development data | Figures, summary tables, engineering interpretation | Complete: reports/EDA_DER.md, five figures, results/eda_*.csv |
| 3 — Preprocessing | Parse phasors; verify phase/angle conventions; construct electrical features; exclude metadata leakage; mask normal location; group splits by load condition; reserve test data | Feature schema, feature cache, split manifest, preprocessing documentation | Complete: executed prepare.py, 5,132 features, grouped/deduplicated manifest |
| 4 — Conventional baselines | Implement B0 thresholds; tune strong B1 voltage/current/phase and sequence rules on development data; establish physically defensible localization and state its coverage | Saved baseline settings, validation metrics and confusion matrices, engineering derivations | Complete: settings/network artifacts, validation and test predictions, confusion matrices and engineering notes |
| 5 — LightGBM | Train separate detection/type/location models; modest validation tuning; retain best settings and artifacts; avoid test-dependent changes | Three trained models, tuning history, preprocessing bundle | Complete: 18 saved best models across six protocols; two-candidate searches; all selection before test freeze |
| 6 — Evaluation | Freeze candidates; evaluate validation and held-out test; report detection/FPR, type macro-F1, location top-k, uncertainty and timing; compare paired predictions fairly | Metrics, predictions, confusion matrices, final comparison report | Complete: results/metrics.csv, predictions/confusions, 2,000 paired load-cluster bootstrap samples; final report |
| 7 — DER analysis | Evaluate all tasks per scenario; distinguish PV from EV/V2G effects; test unseen DER configurations without reusing final test for selection | Scenario tables and figures, unseen-configuration experiment | Complete within release limitations: per-scenario metrics/plot and S6-excluded train/selection models tested at held-out S6 loads |
| 8 — Resistance and limited sensors | Predefine resistance bins; use documented PMU subsets; retrain and compare under identical information | Resistance and sensor tables/figures, coverage audit | Complete: resistance bins and local 93/36/21/9-PMU protocols, matched baseline/ML information, exact-bus versus distance tradeoffs documented. Exact published smaller placements unavailable; intersections explicitly labeled |
| 9 — Explainability and errors | Explain electrical features; analyze false alarms and type/location errors; compare B1-only and ML-only successes | Feature attribution, paired error cases, physical interpretation | Complete: native TreeSHAP on 100 held-out cases per task, additivity checks <2e-14, paired error cases and engineering interpretation |
| 10 — Deployment | Run one local Streamlit app; validate input; replay held-out examples; display confidence and baseline comparison; show verified topology | Tested dashboard, local inference, saved models and one-command launch guide | Scope simplified at the user's request: local Streamlit calls models directly. Separate API/container setup removed. Direct inference and Streamlit replay checks verify saved predictions |
| 11 — Documentation and GitHub | Maintain README, dependencies, provenance and reproduction commands; exclude data/binaries; propose coherent commits without staging user deletions | Reproducible clean project, documented limitations and final report | Complete: README, lock file, provenance, mentor notes, executed final report and explicit-path Git checkpoint workflow. Datasets/caches/binary artifacts excluded. No remote upload requested |

## Research safeguards

- Distinguish source documentation from findings verified in local files.
- Use fault type, bus, resistance, scenario, timestamp and IDs only as labels/audit strata, never predictive inputs.
- Normal rows have arbitrary bus metadata: normal cases have no physical fault location.
- Group shared load conditions across all DER scenarios; audit repeated pre-fault signatures and exact measurement duplicates.
- Do not equate PMU count with network observability, or bus-number differences with network distance.
- Conventional localization without coverage of uninstrumented buses is not a fair full-network comparator; report coverage and comparable subsets explicitly.
- Detection/type/location conditional metrics and end-to-end metrics answer different questions and must be labeled.
- Only actual executed results belong in the performance report. Unsupported metrics are N/A with a reason.

## Working tree preservation

The session began with tracked files deleted (from an earlier GPVS-related project) and only `dataverse_files.zip` present. These deletions are user state. Do not reset, restore, stage, or commit them implicitly. Stage only explicit existing files created/rewritten for this project; preserve unrelated deleted paths.

## Actual research outcome

Dashboard update (2026-10-31): added the executed single-case timing benchmark, relative error reductions, accuracy/error counts and sensor-count tradeoffs to a dedicated Streamlit tab. Rates and model scores display as percentages. Physical fault-to-detection delay remains unavailable; the measured localization-stage computation reduction excludes shared preprocessing. Verification covers the comparison tables, percentage formatting, fault replay and unchanged frozen artifacts.

The all-released-channel held-out comparison gives B1/LightGBM type accuracy 98.6111%/99.6389% and exact-bus accuracy 54.8056%/79.9167%. Detection ties under exact-no-change normals, and ten normal events do not establish low-FPR performance. B1 top-5 location accuracy is slightly higher. With 36/21/9 available PMUs, LightGBM has more exact matches but larger mean branch-hop errors. See the final report for actual scenario/resistance results, paired uncertainty and limitations; do not summarize the study as uniform ML superiority.
