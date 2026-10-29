# GPVS data audit and feasibility checkpoint

**Stage 4 update:** the original paper is now available locally and reviewed.
Table 1 (p.7) verifies F1-F7; sections 2.2/4.1 (pp.5/10) confirm 100 microsecond
sampling and describe independent experiments. Exact physical onsets remain
unknown according to pp.10/13. See `LABELING.md` and the annotation register for
the current evidence. The audit below records the earlier access-limited state;
its statements that the full paper or mechanism mapping are unavailable are
superseded by this update, not current blockers.

Audit and EDA date: 29 October 2026. **Decision: suitable for a small exploratory binary
fault-detection study, but not ready for supervised training.** Exact onset
annotations, primary-paper fault mappings and experiment independence remain
unresolved. EDA is available in `notebooks/01_eda.ipynb`; no labels, model training
or application have been implemented. The provisional nine-input schema is in
`config/input_columns.json`. Time remains separate and estimated channels are
deferred, not permanently deleted.

## Scope and evidence inspected

The repository initially contained a two-line README and the untracked
203,108,810-byte `Data for EECE 490.zip`. No applicable AGENTS.md was found in
the repository or its ancestor directories. Existing raw data were preserved.
No Lebanese monitoring data were found in this repository or archive.

The following existing PDFs were read from `../../` without modification:

- `EECE490_Fall_2026_27_Project_Announcement.pdf` (6 pages): requirements.
- `EECE490_Project_Proposal.pdf`: previous intentions.
- `POSTER.pdf`: previous intentions, including a proposed simulation study.

Revised question: **Does a simple ML detector improve on credible voltage/current
rules for faults in a grid-connected PV laboratory system, and how does behavior
change between MPPT and IPPT?** Intended user: a PV maintenance/operator team
reviewing supplementary alarms. This dataset does not establish field deployment
performance in Lebanese generator networks.

| Earlier proposal/poster claim | Required revision |
| --- | --- |
| Generator versus solar/battery operation | MPPT versus IPPT are PV control modes, not energy-source labels. |
| LG, LL, LLG, LLL and HIF classification | F1-F7 are scenario identifiers; these electrical-fault mappings are not established. |
| Feeder section localization/network map | No branch topology, section labels or branch sensor array are present. Remove localization. |
| Approximately 3,000 simulated runs | Local data are 16 experimental recordings. No such simulation corpus was found. |
| B0 reproduces installed breakers; universal 3x/1.1x current claims | Treat B0 as a limited current heuristic. These claims are not demonstrated by GPVS. |
| B0 misses faults and B1 recovers most | Hypotheses to test, not completed results. |
| 60/20/20 with all fault classes | Not feasible with one recording per fault/mode. |
| Detection within 100 ms at <=1% event false trips | Conditional goals, currently unsupported by onset precision and event counts. |
| Combining Lebanese monitoring with waveform corpora | Keep separate; five-minute/half-hour data cannot supply waveform fault labels. |

## Provenance and access limits

[Mendeley's current dataset URL](https://data.mendeley.com/datasets/n76t439f65)
redirected to version 1, published 28 August 2020, DOI 10.17632/n76t439f65.1,
with CC BY 4.0 on the source page. Authors: Azzeddine Bakdi, Amar Guichi,
Saad Mekhilef and Wahiba Bounoua. The source describes 16 scenarios supplied
in MAT and CSV formats. The local archive agrees with that structure.

Local downloadable content: F0L/F0M through F7L/F7M as 16 `.mat` files,
`ReadMe.pdf`, and `CSV_Files.zip` containing the same 16 identifiers as CSVs.
The nested archive also has macOS metadata entries, which are not recordings.
`checksums.json` records all local file sizes and SHA-256 values, including
these metadata entries. No replacement data were downloaded.

The public web reader did not expose the dynamic remote file manifest; direct
page/API requests returned HTTP 403. Thus **current individual download URLs
and remote-to-local checksum equality are not verified**. Local checksums prove
repeatability of this copy, not authenticity against the remote release.

Associated paper: Bakdi et al., *Real-time fault detection in PV systems under
MPPT using PMU and high-frequency multi-sensor data through online PCA-KDE-based
multivariate KL divergence*, IJEPES 125 (2021), 106457,
[DOI 10.1016/j.ijepes.2020.106457](https://doi.org/10.1016/j.ijepes.2020.106457).
The dataset cites it as 2020; the journal volume is February 2021. Publisher
preview and institutional abstract were accessible, but full text was not:
publisher access was restricted and the Boumerdes repository timed out.
[University of Malaya's record](https://knova.um.edu.my/research_publications_2021_2025/11351/)
also says full text is unavailable there. The accessible abstract concerns PV,
grid, inverter, sensor and controller faults, but does not establish numbered
F1-F7 mappings. **Full-paper review is incomplete; no mapping is invented.**

## Inventory and measurement quality

There are **16 recordings, not 32 independent experiments**. Each MATLAB v7.3
HDF5 table has a corresponding CSV export. Every numerical entry was compared
with `rtol=1e-12, atol=1e-12`; all passed. Maximum absolute difference across
all pairs is approximately 5.12e-13, consistent with text export precision.
The columns are float64 in the parsed CSV and in the MATLAB measurement arrays.
The archive contains 2,163,480 waveform rows, totaling 216.313 seconds of recorded
time across files. Rows are strongly dependent measurements, not experiments.

| Scenario | IPPT recording: rows / duration (s) | MPPT recording: rows / duration (s) | Documented meaning |
| --- | ---: | ---: | --- |
| F0 | 143,715 / 14.369947 | 141,014 / 14.099888 | Fault-free |
| F1 | 129,013 / 12.899921 | 139,014 / 13.899897 | Fault scenario; description unresolved |
| F2 | 142,128 / 14.199899 | 144,015 / 14.399960 | Fault scenario; description unresolved |
| F3 | 103,497 / 10.348573 | 69,967 / 6.995897 | Fault scenario; description unresolved |
| F4 | 144,014 / 14.399894 | 144,014 / 14.399866 | Fault scenario; description unresolved |
| F5 | 143,015 / 14.299936 | 144,014 / 14.399863 | Fault scenario; description unresolved |
| F6 | 144,015 / 14.399938 | 144,015 / 14.399917 | Fault scenario; description unresolved |
| F7 | 144,015 / 14.399890 | 144,015 / 14.399846 | Fault scenario; description unresolved |

Every file has 14 columns:

| Columns | Meaning | Unit status |
| --- | --- | --- |
| Time | Recorded time | Seconds explicitly documented |
| Ipv; Vpv; Vdc | PV current; PV voltage; DC voltage | A / V / V expected physically; not explicitly stored as units |
| ia, ib, ic | Three-phase currents | A expected; calibration/scale needs confirmation |
| va, vb, vc | Three-phase voltages | V expected; line/phase reference and calibration need confirmation |
| Iabc, Vabc | Estimated positive-sequence magnitudes | Current/voltage units expected; RMS versus peak convention unresolved |
| If, Vf | Estimated current/voltage frequency | Hz expected; estimator delay and implementation unresolved |

MATLAB VariableUnits fields are empty. The local README has a typo in the Vf
description; the public page identifies it as voltage frequency. No explicit
onset channel, fault-state channel, irradiance, temperature, diesel source flag,
branch ID or experiment/session metadata appears among the 14 columns.

**Timing discrepancy:** the README reports 9.9989 microseconds (about 100 kHz),
but every recording has a median step near 99.9966-99.9969 microseconds
(about 10 kHz). Average rates are 10,000.976-10,001.080 Hz except F2L at
10,009.015 Hz. Use timestamps as supplied; do not silently multiply time by ten
or assume why the documentation differs. Confirm the intended acquisition/export
rate with authors before finalizing latency claims.

All times increase strictly; no missing/non-finite values, duplicate full rows,
duplicate complete 13-channel measurement rows or consecutive identical complete
measurement rows were found within any recording. Every complete signal-array
hash differs across recordings. This does not prove statistical independence
or rule out shared subsequences or shared experimental sessions.

Timing is not perfectly uniform: each recording has 1-9 steps differing from
its median by more than 1%. F5M and F7M each have one gap greater than 1.5 times
their median. Across all files the minimum step is 36.656 microseconds (F7M),
and maximum is 163.338 microseconds (F5M). The per-file minimum, maximum, mean,
median, standard deviation and counts are in `recording_inventory.csv`.
The gap criterion is a diagnostic, not proof of a missing sample.

Repeated individual channel values are present, as expected with quantization,
and are counted in `channel_statistics.csv`. Specific review flags:

- All recordings contain runs of 201 identical magnitude values (Iabc/Vabc = 1)
  and 204 identical frequency values (If/Vf = 50). These suggest estimator
  initialization; verify warm-up and causality before using these channels.
- F3L and F5M have slightly negative minimum If values (approximately -0.0222),
  suggesting the frequency estimate may become unreliable at very low current.
  Do not interpret those estimates as literal negative grid frequency.
- Vdc reaches zero and has a longest identical-value run of 839 samples somewhere
  in the corpus. Negative Ipv/Vpv and large current excursions also occur.
  These are inspection flags, not automatic evidence of corrupt data: transients,
  faults, noise or shutdown may explain them. Do not delete or clip them blindly.
- F3 recordings end much earlier than most others. A shutdown is possible, but
  file length alone does not establish the mechanism or onset time.

## Labels and independence

There is one supplied recording for each of the 16 scenario/mode combinations.
The README describes each file as an experiment. It provides no run-repeat count,
session IDs, acquisition dates or pairing metadata, so **one recording per cell
is observed; one statistically independent trial per cell is not established**.
The two formats must always share one recording ID. Related sessions, if later
identified, must share a larger group in any split.

F0L and F0M are documented fault-free experiments (28.469835 seconds combined,
before any warm-up exclusion). F1-F7 reportedly include operation before manual
injection. Exact normal intervals and sustained-fault intervals have not been
validated. The documentation says injection occurred "halfway"; it gives no
sample index, tolerance or per-run timestamp. Inventory midpoints are geometric
references only and MUST NOT become labels. No whole fault file should be
labelled faulty; even the first half should not automatically become exact normal
ground truth. Later recovery, intermittent faults and shutdown need review.

Proposed annotation procedure, independent of detector outputs:

1. Obtain the full original paper and, preferably, author injection logs or
   sample-level annotations. Record the primary source/page for each F mapping.
2. Two human reviewers inspect the raw multichannel trace and available logs,
   blinded to predictions. Record earliest/latest plausible onset `[t_lo,t_hi]`,
   confirmed-normal and confirmed-fault intervals, recovery/shutdown, evidence,
   uncertainty, reviewer and adjudication. Observable signal change is not
   automatically the physical injection time.
3. Mark ambiguous intervals unknown and exclude them from supervised scoring.
   Exclude causal windows overlapping uncertainty or warm-up. Freeze annotations
   before selecting features, rules or models. Keep unresolvable runs unlabeled.
4. A detection at `t_alarm` gives latency interval
   `[t_alarm-t_hi, t_alarm-t_lo]`, not an exact latency. Do not claim physical
   100 ms detection from waveform-derived onset without adequate precision.

## Proposed evaluation for review

**Update:** the user selected a fixed approximate 60/20/20 holdout. The active
design is now in `SPLIT.md` and `config/recording_split.json` (10/3/3 recordings).
The scenario-pair cross-validation proposal below is retained as an alternative,
not the active protocol. The fixed split does not provide every class in every
partition or resolve onset/session uncertainty.

**Initial task: binary detection on confirmed intervals in unseen recordings.**
Multiclass diagnosis is deferred. Two recordings per scenario across both modes
cannot support train/validation/test membership for every scenario. Within each
mode there is only one. Random-row or random-window splitting would measure
recognition of the same experiment and is prohibited.

Recommended exploratory design, contingent on labels and independence:

- Group the L/M pair for each scenario together for an outer leave-one-fault-
  scenario-pair-out study (seven folds). This asks whether binary detectors
  generalize to a fault scenario absent from development, not whether they can
  identify every known class. Preserve whole recordings and related sessions.
- In each outer fold, select thresholds/hyperparameters using inner grouped
  folds over the remaining six fault scenarios. Both binary classes must actually
  exist in each usable partition after annotation; otherwise reduce scope.
  Pre-fault intervals of the held-out pair remain strictly test data.
- F0L/F0M can be declared development-only normal references, never advertised
  as independent held-out normal tests in that design. Report held-out confirmed
  pre-fault false alarms separately. With only two all-normal recordings, there
  is no large independent normal test set. Additional benign-event recordings
  are necessary for a reliable false-trip claim.
- Allocate group IDs before window extraction. Use trailing causal windows;
  consider 10/20/40 ms only after validating sample rate and electrical reference.
  At about 10 kHz these correspond roughly to 100/200/400 samples; use elapsed
  time when handling irregular steps. 10 ms is only half of a nominal 50 Hz cycle.
- Fit scaling, imputation if needed, feature selection, calibration and all
  thresholds using permitted development data only. No filename, absolute
  experiment time, row number, file length or midpoint indicators as predictors.
  Mode is grouping metadata initially; any mode-aware rule must be available
  equally to competitors and declared explicitly.
- Cross-mode transfer (train L, evaluate M and reverse) is optional exploratory
  analysis: shared scenario identity and possible related sessions limit it.
  It does not establish generator-to-inverter transfer or unseen-fault transfer.

There is no untouched large final test set in this plan. Nested grouped results
are descriptive estimates with very few groups, not a production guarantee.
The full dataset has already been inspected for audit; document that fact and
avoid tuning against patterns in future outer-fold outcomes.

Compare B0 current threshold, B1 stronger voltage deviation/imbalance/persistence
rules, a simple ML comparator (e.g. logistic regression), and only then LightGBM.
B0 may be unsuitable for reduced-current faults; assess relevance after mapping
mechanisms. B1 should use validated waveform measurements and physically justified
references. Exclude supplied PMU estimates initially unless their causal delay
and initialization are established. Do not call B0 an exact circuit-breaker model.

All methods receive identical permitted signals and use the same causal window,
alarm persistence, reset/debounce and event definition for each comparison.
Compare justified window lengths on development folds, including their buffering
delay. Report measured computation latency separately from observation delay.
Keep the same allowed tuning budget/data and operating-point criterion.

Report per-recording event detection counts, missed events, time-to-alarm bounds,
confirmed-normal exposure and false alarms per recording and per second/minute.
Define alarm merging and event matching before evaluation. Window-level precision,
recall or PR-AUC may be secondary and must not be called event false-trip rates.
Show raw paired B1/ML outcomes and variability by mode/scenario. If resampling is
used, resample independent groups, never windows; with seven scenario groups,
intervals and significance claims will be fragile. Do not force a conclusion
that ML beats rules.

A <=1% event false-trip claim is not feasible here. For illustration, even with
zero false alarms, a one-sided exact 95% binomial upper bound falls to <=1% only
after 299 independent harmless events (`1 - 0.05**(1/n)`). This corpus has two
documented all-normal recordings, not 299 independent harmless events. Slicing
them into hundreds of windows does not create those events.

## Course alignment and checklist

The actual announcement requires a baseline and ML comparison, justified metrics,
leakage control, reproducibility, failure analysis, a Dockerized API, functional
interface, documented contributions and a final approximately five-minute
presentation. The realistic constraint requires approval during project review.
Propose limited labeled recordings and MPPT/IPPT shift; do not assume approval.

Responsible-ML topics tied to this system: (1) leakage and honest evaluation,
(2) robustness to operating-mode shift and unreliable sensor estimates,
(3) missed-fault/false-alarm consequences and human oversight, and
(4) interpretability through rule explanations and later SHAP error analysis.
SHAP does not establish causality. The eventual prototype provides supplementary
maintenance alerts, with no automatic protection action inferred from these data.

- [x] Inspect repository and available course materials.
- [x] Locate and preserve raw files; add Git exclusions.
- [x] Read supplied dataset documentation and verify current version/license.
- [x] Inspect all MAT/CSV pairs, timestamps, numeric quality and local checksums.
- [x] Document obsolete scope and propose grouped evaluation.
- [x] Perform descriptive EDA and apply a reversible initial input schema.
- [ ] Verify remote individual file manifest and checksum provenance.
- [ ] Obtain/read full primary paper; verify numbered fault descriptions and units.
- [ ] Resolve timing discrepancy and recording/session independence.
- [ ] Review annotation procedure and create validated interval labels.
- [ ] Review/approve revised task, metrics and evaluation at this checkpoint.
- [ ] Then implement loaders/labels, causal features, B0/B1, simple ML and LightGBM
  in small explainable stages, followed by error analysis and uncertainty.
- [ ] After credible experiments: FastAPI, Docker, Streamlit replay, documentation
  and actual team contributions (Omar El Hajj, Sami Marouni, Houssein Karnib).

Before training, review these unresolved evidence gaps; they are not permission to
invent annotations or compensate with a random-window split.
