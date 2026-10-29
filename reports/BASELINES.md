# Baseline preparation

Two provisional rule detectors are implemented in `scripts/baselines.py`. Their
reference ranges were fitted exclusively to 1,403 accepted normal training windows
from F0L and F0M. No validation/test predictions or performance metrics were produced.
These rules detect deviations from the available normal reference, not fault types.

- **B0:** a candidate alarm when any phase-current RMS exceeds that phase's maximum
  in the normal training windows. It cannot detect reduced-current faults reliably.
- **B1:** a candidate alarm when Vdc mean or any phase-voltage RMS leaves its normal
  training minimum/maximum range, or when current/voltage RMS spread exceeds its
  normal training maximum. This is a separate voltage/imbalance comparator, not a
  union with B0. DC PV features remain available for subsequent models.
- **Both:** require three consecutive candidate windows. Any noncandidate window
  immediately clears the alarm; missing window steps and recording changes reset
  the counter. There is no additional latch, hysteresis or event merging yet.

These empirical ranges pool the two modes. They are initial engineering choices,
not equipment protection settings, physical operating limits or validated alarm
thresholds. Min/max ranges are sensitive to outliers and the limited operating
conditions represented by two recordings. A mode-specific reference would need a
justified deployment assumption and validation. Zero candidates on calibration
normals follows from using their extrema and is not evidence of generalization.

Three overlapping windows at a nominal 20 ms step add about 40 ms after the first
candidate window; their evidence is correlated. Together with the 100 ms observation
window, this does not establish a detection-delay bound. Exact onset is still unknown.

## Run

```powershell
.\.venv\Scripts\python.exe scripts/baselines.py
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p test_baselines.py
```

Outputs are local, regenerable files: `data/baselines/rules.json` and
`data/baselines/train_predictions.csv`. The latter contains exploratory training
alarm traces, not fault ground truth. Predictions never promote annotations.
The script checks extraction input fingerprints before calibration and rejects
recordings outside the training assignment. Five tests verify the partition guard,
exclusion of unknown calibration labels, rule/persistence behavior, and resets at
gaps and recording boundaries.

## What still requires evidence

The baseline implementation is ready for validation after interval review. The
team must review the candidate regions in `annotations/intervals.csv` using the
paper and waveform evidence, recording reviewers and decisions as described in
INTERVAL_REVIEW.md. Unknown intervals should stay unknown. The two-reviewer process
is this project's annotation quality convention, not a claim about course rules.
There are still no accepted positive windows or accepted held-out normal windows.
Consequently, fault recall, precision/F1, held-out false-alarm rates and model
comparisons cannot yet be reported. Event definitions and scoring must also be
specified before evaluation. Do not tune thresholds to the existing test plots.

Next: independent label review, then a simple supervised comparator and validation
of thresholds/persistence using only the designated development data. Final test
scoring follows frozen choices; unresolved labels remain excluded and their
coverage and selection bias must be reported.
