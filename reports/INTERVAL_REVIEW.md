# Stage 4 interval review — 30 October 2026

The first visual review covers all 14 fault recordings. It uses raw-measurement
summaries, not model predictions or the paper's detector alarm times. No exact
physical injection timestamp has been inferred. The frozen 10/3/3 split remains.

## Results

| Recording | Proposed normal region (s) | Proposed changed/fault region (s) |
| --- | --- | --- |
| F1L | 0.1–8.4 | 8.9–12.8 |
| F1M | 0.1–8.2 | 8.7–10.0 |
| F2L | 0.1–8.5 | 9.2–14.1 |
| F2M | 0.1–8.3 | 9.0–14.3 |
| F5L | 0.1–6.8 | 8.0–10.4 |
| F5M | 0.1–7.0 | 8.1–10.3 |

These are deliberately interior review regions, not estimated injection bounds
or measured confidence intervals. They are **assistant proposals only**. A stable
pre-change trace is not proof of normality, and a changed trace can include fault
consequences or shutdown. They require independent review before becoming labels.
The 0.1-second starting point is a conservative review boundary, not a validated
warm-up duration. No general warm-up rule has been fitted or applied.

F1 shows current-pattern changes; F1M also has DC collapse and a later current
collapse excluded from its proposed changed region. F2 shows phase-current
separation with a DC-voltage shift. F5 shows PV current/voltage collapse and later
changes, so its entire suffix is not assigned one state.

F3L/F3M remain unknown: a late current collapse and DC rise are visible, but these
do not locate each intermittent sag or distinguish shutdown from fault onset.
F4L/F4M, F6L/F6M and F7L/F7M remain unknown because the reviewed summaries do not
establish sufficiently clear boundaries. Selecting only the clear cases would
favor easier faults, so any later restricted evaluation must disclose this bias.

F0L/F0M retain documented normal status. Their startup handling is separate from
their fault labels. All omitted times and gaps between proposed regions are unknown.

## Files and review procedure

`annotations/intervals.csv` holds 22 rows: two documented normal intervals,
12 proposed intervals across six recordings, and eight unresolved full-recording
entries. The physical-onset columns in `annotations/recordings.csv` remain blank.

Reproduce the seven paired review plots:

```powershell
.\.venv\Scripts\python.exe scripts/review_intervals.py
.\.venv\Scripts\python.exe scripts/review_intervals.py --scenario 1 --start 8 --end 9.5
.\.venv\Scripts\python.exe scripts/review_intervals.py --scenario 2 --start 8 --end 9.5
.\.venv\Scripts\python.exe scripts/review_intervals.py --scenario 5 --start 6.8 --end 8.2
```

Plots are in `reports/figures/label_review/`. They use 20 ms disjoint display bins:
DC median/minimum/maximum and phase RMS including the DC component. This is
offline visualization, not the causal feature implementation. Short transients
can be obscured by summaries; human reviewers must inspect raw-waveform detail
where needed. Plot generation does not write or approve annotations.

Two team reviewers should independently assess the proposed intervals and
unresolved recordings using the measurements and paper, without model predictions.
Record their identities, evidence and adjudication; adjust boundaries or retain
unknown status when disagreement remains. `reviewed` denotes team-reviewed
waveform labels, not exact physical-injection ground truth. Do not change status
merely to make training run. Full-corpus review for labeling must be disclosed.

## Safe handoff to stage 5

`scripts/interval_labels.py` validates bounds and rejects overlaps, unsupported
documented labels and missing reviewer information. Only `documented` or
`reviewed` intervals can supply 0/1 labels. A window must fit completely inside
one accepted interval; its endpoint alone is insufficient. Proposals, ambiguous
regions and uncovered times return missing labels, never an implicit zero.

**Ready for stage 5 feature preparation, not supervised training:** features can
be extracted separately inside each frozen recording partition while unknown
labels stay missing. The first-pass review is complete; independent annotation
review is still pending. Do not claim stage 4's ground-truth work is fully done.
