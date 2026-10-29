# Fixed train / validation / test split

Requested design: 60% training, 20% validation, 20% testing using existing data.
With 16 indivisible recordings, the closest counts are **10 / 3 / 3**, or
**62.5% / 18.75% / 18.75%**. These are recording percentages, not sample or duration
percentages. Exact 60/20/20 would require dividing recordings or omitting data.

The frozen assignments are in `config/recording_split.json`. Reproduce/check them:

```powershell
.\.venv\Scripts\python.exe scripts/split_gpvs.py
```

The fixed seed is 490. Fault scenario IDs are ranked with SHA-256 using the seed;
assignments use only scenario/mode metadata, never signal patterns or model scores.
Training has five recordings per mode. Validation has one IPPT and two MPPT;
test has two IPPT and one MPPT. Training covers F0-F7. Each held-out fault
recording has the other operating mode of that scenario in training.

## How to use the split

`gpvs_data.load_partitioned_recordings()` returns dictionaries of full recordings
under `train`, `validation` and `test`. Split first, then extract windows inside
each recording. A MAT file and its CSV export share the same recording ID; they
cannot be assigned separately. Do not shuffle rows into new partitions.

- Training: fit model parameters, scalers, imputers and learned feature selection.
- Validation: choose model settings, feature/window choices and alarm thresholds.
- Testing: evaluate the frozen pipeline once; do not adjust it after seeing results.

Stage 5 now generates recording-local windows under this split; see FEATURES.md.
Unreviewed labels remain missing. Raw files are unchanged.

## Limits to report

Both F0 recordings stay in training to provide documented normal data in both
modes. Validation and test contain fault-scenario recordings whose pre-injection
normal intervals still need confirmation. Until these and positive intervals are
annotated, binary metrics and alarm-threshold selection are not ready. Never
label whole files faulty or treat their first halves as exact normal labels.

Not every fault scenario appears in validation or test, so this split does not
support an all-class diagnosis benchmark. Its initial purpose is exploratory
binary detection on held-out recordings, with both modes represented. It is not
an unseen-fault or pure cross-mode evaluation. There are only three test recordings;
no reliable <=1% event-level false-trip claim follows from this design.

The reviewed primary paper (pp.5/10) describes independent experiments, supporting
recording-level grouping. Recording IDs are the grouping unit because session IDs are unavailable. If L/M
recordings share a session or other dependence, record-level separation alone
does not guarantee independence: revise grouping before training if that is found.
Whole-corpus EDA was already performed; disclose this rather than calling the test
set completely unseen. Do not reroll the seed to obtain better results.

This fixed holdout design replaces the earlier proposed leave-one-scenario-out
plan as the primary experiment, at the user's request. That earlier plan remains
an optional future sensitivity study, not an additional required step.
