# Fault Detection and Diagnosis in Grid-Connected Photovoltaic Systems

AUB EECE 490, Path A: investigate whether ML improves PV fault detection over
credible rules across MPPT and IPPT control modes.

**Current stage: Stage 5 feature preparation completed; interval review pending; no models trained.**

Provisional [B0/B1 rule baselines](reports/BASELINES.md) are now implemented and
calibrated on documented normal training windows only. No supervised classifier
has been trained and no held-out performance has been measured.

Next practical task: complete the [label review worksheet](reports/LABEL_REVIEW_WORKSHEET.md).
It links all 14 fault recordings to the existing waveform plots and leaves review
decisions blank. The [readiness check](reports/LABEL_READINESS.md) reports current
class coverage. Regenerate both with
`.\.venv\Scripts\python.exe scripts/prepare_label_review.py` (save completed notes
separately, because regeneration replaces the template).

The [feature report](reports/FEATURES.md) documents 15 causal predictors and 10,654
windows in the frozen partitions. Run
`.\.venv\Scripts\python.exe scripts/extract_features.py` to regenerate them.
Unreviewed labels remain missing; no accepted fault labels are available yet.

The requested **60/20/20 holdout** is now fixed at 10 training, 3 validation and
3 test recordings (62.5/18.75/18.75%, the closest feasible whole-recording split).
See [the split design](reports/SPLIT.md) and [assignments](config/recording_split.json).
Run `.\.venv\Scripts\python.exe scripts/split_gpvs.py` to reproduce/check it.

Stage 4 primary-paper review is complete: [verified fault mapping and timing limits](reports/LABELING.md).
The [first interval review](reports/INTERVAL_REVIEW.md) adds six recordings with
proposed regions and leaves eight ambiguous fault recordings unresolved.
`annotations/intervals.csv` separates proposals from accepted labels; the validator
keeps proposals out of training. Stage 5 feature preparation may proceed with
missing labels, while independent interval review remains pending.
The [recording register](annotations/recordings.csv) now identifies all F1-F7
mechanisms. The paper confirms nominal 10 kHz sampling but explicitly lacks
precise physical fault-onset times. F0 normal intervals are documented; positive
intervals still need independent annotation before supervised training.

## Start here

Open [the EDA notebook](notebooks/01_eda.ipynb). It contains executed tables,
seven figures, explanations and the initial input-column decision.

- **Keep as inputs:** Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc.
- **Keep separately:** Time, for window timing and evaluation, never as a predictor.
- **Defer:** Iabc, If, Vabc, Vf until their estimator initialization and causal delay
  are verified. This is provisional, not proof that these signals are useless.

The [column policy](config/input_columns.json) is applied by
`scripts/gpvs_data.py:model_input_view`. It returns a copy with nine columns and
all original rows. Raw data and the full EDA tables retain all 14 columns.
No imputation, outlier clipping or learned feature selection has been applied.

Read [the feasibility report](reports/FEASIBILITY.md) for provenance, unresolved
labels and the proposed evaluation. Whole recordings must stay together in splits.
Full-corpus EDA is descriptive and must be disclosed; future learned preprocessing
and feature selection must use training data only.

## Setup and run

Keep `Data for EECE 490.zip` in the repository root. Use Python 3.12 and PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_eda.py
```

Alternatively, open `notebooks/01_eda.ipynb` in VS Code, select
`.venv\Scripts\python.exe` as the notebook kernel and choose **Run All**.
The command-line runner saves the executed notebook with embedded plots.
It takes roughly a minute or two depending on the computer and loads the
recordings into memory. It does not export a second large dataset.

For the full MAT/CSV integrity audit:

```powershell
.\.venv\Scripts\python.exe scripts/audit_gpvs.py
```

On this workstation, the Windows `python` alias was unavailable. The existing
`.venv` was created using bundled Python 3.12 and shared base packages. To recreate
that environment here, use the following instead of the setup commands above:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv --without-pip --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --target '.venv\Lib\site-packages' -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_eda.py
```

## Submission files

| File | Purpose |
| --- | --- |
| `notebooks/01_eda.ipynb` | Main EDA narrative, code, plots and results |
| `scripts/gpvs_data.py` | Read-only data loader and provisional input view |
| `scripts/run_eda.py` | Re-execute the notebook with the current interpreter |
| `config/input_columns.json` | Explicit, reversible input-column policy |
| `annotations/recordings.csv` | Recording bounds and evidence-backed annotation status |
| `reports/LABELING.md` | Label verification findings and completion procedure |
| `scripts/audit_gpvs.py` | Reproduce data-quality and MAT/CSV consistency checks |
| `reports/recording_inventory.csv` | Timing, dimensions, numerical quality and MAT comparison |
| `reports/channel_statistics.csv` | Per-channel ranges, repetition and missing values |
| `reports/checksums.json` | SHA-256 fingerprints of supplied raw files |
| `reports/audit_environment.json` | Versions used to produce the audit |
| `reports/FEASIBILITY.md` | Dataset limitations, course alignment and evaluation plan |
| `requirements*.txt`, `.gitignore` | Reproducible setup and raw/private file exclusions |

The duplicate extracted dataset README and separate MAT inventory were removed;
MAT comparison evidence is now in the recording inventory. PNG previews, temporary
files, virtual environments, raw ZIP/MAT files and private `data/` are ignored by Git.
The official dataset README remains intact inside the raw archive.

## Next steps

- [x] Audit raw data and preserve local checksums.
- [x] EDA and a reversible initial input-column policy.
- [x] Verify F1-F7 descriptions and nominal sampling against the original paper.
- [x] Record the paper's support for independent experiments and its timing limits.
- [ ] Finish interval annotation and remaining calibration/latency checks.
- [x] Freeze a recording-level approximate 60/20/20 split before model features.
- [ ] Validate interval labels and session independence before fitting models.
- [x] Implement and verify causal feature extraction within frozen partitions.
- [x] Prepare provisional B0/B1 rules from accepted normal training windows.
- [ ] Compare B0/B1 rules, a simple ML comparator and justified LightGBM.
- [ ] Analyze errors and mode shift, then build the Dockerized API and replay UI.

## Data attribution

Bakdi, A.; Guichi, A.; Mekhilef, S.; Bounoua, W. (2020), GPVS-Faults,
Mendeley Data V1, [DOI 10.17632/n76t439f65.1](https://data.mendeley.com/datasets/n76t439f65/1),
CC BY 4.0. Results here are descriptive summaries of the supplied copy, not
model performance. Raw files are unchanged; remote checksum equality remains
unverified. Original proposal/poster claims describe earlier intentions, not results.
