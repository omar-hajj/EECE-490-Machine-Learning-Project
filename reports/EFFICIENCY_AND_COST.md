# Additional comparison — errors, computation time and sensor budget

This analysis uses the already frozen models. No retraining or final-test-driven selection was performed.

## What was compared

B0 is a basic fixed voltage-sag/current-threshold detector. It does not classify or locate faults. B1 combines electrical phase and zero-sequence rules with an approximate source-network current-injection residual location method. LightGBM uses separate learned detection/type/bus models. All methods receive the same available measurements within each experiment and are scored against actual held-out labels.

## A meaningful additional percentage: relative error reduction

| task | B1_accuracy | LightGBM_accuracy | accuracy_gain_percentage_points | relative_error_reduction_percent |
| --- | --- | --- | --- | --- |
| classification | 0.9861 | 0.9964 | 1.0278 | 74.0 |
| localization | 0.5481 | 0.7992 | 25.1111 | 55.5624 |

Accuracy gain in percentage points is 100 × (accuracy_ML − accuracy_B1). Relative error reduction is 100 × (errors_B1 − errors_ML) / errors_B1. These answer different questions. Do not call a percentage-point gain the same thing as a relative percentage improvement. Detection ties, so no error-reduction ratio is defined when both make zero observed errors. All conclusions retain the original dataset and B1 limitations.

## Actual computation benchmark

| component | n | mean_ms | median_ms | p95_ms |
| --- | --- | --- | --- | --- |
| B0_detection | 330 | 0.013822 | 0.012 | 0.026265 |
| B1_classification | 330 | 0.217463 | 0.1898 | 0.43351 |
| B1_detection | 330 | 0.015939 | 0.0133 | 0.03171 |
| B1_localization | 330 | 10.711498 | 9.5902 | 18.073635 |
| LightGBM_classification | 330 | 1.403525 | 1.2437 | 2.52335 |
| LightGBM_detection | 330 | 1.035128 | 0.90475 | 1.92357 |
| LightGBM_localization | 330 | 1.867379 | 1.7066 | 3.198835 |
| shared_phasor_and_feature_transform | 330 | 27.656808 | 24.53345 | 51.343195 |

Units are milliseconds. The benchmark uses 110 distinct test cases, up to ten per class, three repeated passes, ten warm-up cases, shuffled method order and four threads. Median and 95th percentile describe computing time across the sampled runs, not a confidence interval or a worst-case guarantee. CPU/background conditions can affect results. This is a new warm single-case benchmark; do not mix it with the earlier batch-throughput timings.

| task | B1_median_ms | LightGBM_median_ms | ML_time_reduction_percent |
| --- | --- | --- | --- |
| detection | 0.0133 | 0.9048 | -6702.6316 |
| classification | 0.1898 | 1.2437 | -555.2687 |
| localization | 9.5902 | 1.7066 | 82.2048 |

Time reduction = 100 × (time_B1 − time_ML) / time_B1. Positive means less computing time for ML; negative means more. These compare the current prototype implementations on this computer, not optimized relay hardware. Localization improves computationally while ML detection and classification are slower in this implementation. The shared feature transformation is excluded from this percentage and can dominate the complete prediction path; the model-stage reduction must not be presented as an end-to-end speedup.

The shared transformation is measured separately and constructs all electrical features for the current implementation. B0/B1 rules do not intrinsically need all ML features, so the common transform is not an optimized minimum baseline preprocessing cost. B1 localization uses its predicted type; its classification cost must be added when accounting for that dependency. Model-only timings exclude loading, validation, communication and acquisition. The prototype's earlier 41.12 ms average includes validation, feature construction, all three ML models and both baselines on five replay cases; it is not a detection-only latency.

**We cannot measure physical fault detection delay from this release.** Steady-state pre/post phasors lack a fault-inception-aligned waveform stream and decision windows. End-to-end delay would include measurement/phasor estimation, communications, algorithm computation and (for clearing time) breaker action. Computing faster does not by itself establish faster fault clearing or reduced outage duration.

## Sensor-count and cost tradeoff

| PMUs | model | PMU_count_reduction_percent | exact_bus_accuracy_percent | mean_branch_hops |
| --- | --- | --- | --- | --- |
| 93 | B1 | 0.0 | 54.6944 | 0.5589 |
| 93 | LightGBM | 0.0 | 79.8889 | 0.35 |
| 36 | B1 | 61.2903 | 28.25 | 2.0897 |
| 36 | LightGBM | 61.2903 | 38.3889 | 2.4031 |
| 21 | B1 | 77.4194 | 24.1667 | 2.4267 |
| 21 | LightGBM | 77.4194 | 31.5 | 3.095 |
| 9 | B1 | 90.3226 | 17.9167 | 3.0692 |
| 9 | LightGBM | 90.3226 | 25.2222 | 3.3747 |

PMU-count reduction is a hardware-count proxy, not measured financial savings. These are the physically local current-terminal experiments. Placements differ and are not nested, so count and instrument locations change together. Reduced sensors significantly reduce localization accuracy. Do not claim an equivalent-accuracy cheaper system from these results.

A financial model would require explicit prices/assumptions: total cost = number of PMUs × per-PMU purchase/installation cost + communications + computing + maintenance. Annualized cost, power consumption and avoided outage cost have not been measured. Unit-cost scenarios can be added later and labeled estimates, but currency savings and reduced outage costs are currently unsupported.

## Suggested presentation statement

Report the measured type/localization error reductions alongside absolute accuracy, computing-time medians/p95, and sensor-budget tradeoffs. The contribution is an accuracy–time–measurement tradeoff study; it does not assume ML is faster or cheaper. Preserve cases and metrics where B1 is better.
