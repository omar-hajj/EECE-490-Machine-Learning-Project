# Stage 5: causal feature preparation

Feature extraction is implemented and run on all 16 recordings using the frozen
10/3/3 recording split. No model, scaler, threshold or learned feature selection
has been fitted. Independent interval review remains a prerequisite for supervised
training and evaluation.

## Features and timing

The 15 predictors are the mean and population standard deviation of each of Ipv,
Vpv and Vdc (6), mean instantaneous Ipv * Vpv (1), individual phase current and
voltage RMS values (6), and maximum minus minimum phase RMS for current and
voltage (2). RMS spread is an absolute magnitude proxy, not a calibrated imbalance
percentage or a sequence-component estimate. Units follow the source measurements;
sensor calibration and power-sign conventions remain unverified.

Each window uses 1,000 samples and advances 200 samples: nominally 100 ms and
20 ms at 10 kHz. Output is available at the last sample timestamp; there are no
centered filters, future samples, padding or windows crossing recording boundaries.
The saved actual first-to-last sample span is about 99.9 ms and varies with timing
jitter. Incomplete tails are omitted. These choices are provisional and have not
been optimized on held-out performance. Window buffering, step timing, computation,
sensor delay and alarm persistence all matter to eventual detection latency;
this implementation does not establish a <=100 ms detection claim.

Windows beginning within the first 0.1 seconds of each recording are excluded as
a provisional startup precaution. This is not proof that settling has finished.
No normalization, clipping, imputation or further input removal is applied. The four
estimated channels remain deferred under the existing input policy.

Only a window contained completely in a documented or reviewed interval receives
a binary label. All proposed, unresolved and boundary-crossing windows stay missing.
Recording ID, timing, sample indices and labels are metadata, never predictors.
Use the explicit predictor allowlist in the output manifest when fitting later.

## Generated inventory

| Partition | Windows | Accepted normal | Accepted fault | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,632 | 1,403 | 0 | 5,229 |
| Validation | 2,095 | 0 | 0 | 2,095 |
| Test | 1,927 | 0 | 0 | 1,927 |

Overlapping windows are correlated; 10,654 windows do not constitute independent
experiments. Whole recordings remain in their assigned partition. Full-corpus EDA
and interval inspection were already performed and must be disclosed. Extracting
held-out features here fits nothing and is not a model evaluation.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p test_features.py
```

Local generated CSVs are `data/features/train.csv`, `validation.csv`, `test.csv`.
They are ignored by Git and can be regenerated; rerunning replaces them. The
manifest stores settings, the predictor allowlist and input/source fingerprints.
`reports/feature_inventory.csv` is the compact per-recording submission summary.
Four automated checks cover known feature values, past-output invariance to future
changes, missing proposed labels/startup exclusion, and rejection of nonfinite data.

Next: resolve independent annotation review and settling/calibration questions,
then fit preprocessing and baseline rules using training data only. Validation
will select settings; test scoring waits until those settings are frozen. Currently
there are no accepted positive labels and no accepted validation/test labels, so
supervised fitting and meaningful held-out scoring cannot yet proceed.
