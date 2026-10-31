# Engineering defense notes

## What each phase establishes

1. **Inspection establishes what is measurable.** Labels alone are insufficient: we verified phasor channels, phase identity, checksums, duplicate states and source topology before fitting anything.
2. **EDA connects the labels to circuit behavior.** Phase-selective voltage disturbance is consistent with affected phases; higher resistance usually reduces current and voltage disturbance. Zero sequence helps distinguish a two-phase ground fault from a two-phase line fault.
3. **Preprocessing makes the comparison reproducible.** Complex conversion handles angle wrap. Physical phase mapping prevents calling conductor 1 “phase A” on a single-phase B or C branch. No simulation metadata becomes a predictor.
4. **Conventional methods are the reference engineering solution.** B1 has strong type rules and a complete candidate-bus network method. Improving B1 before testing is part of a fair study, not an obstacle to ML.
5. **LightGBM learns interactions across instruments.** Separate detection, type and bus models allow each task's errors to be measured. Training-only feature screening reduces the search space without consulting held-out labels.
6. **Evaluation distinguishes apparent accuracy from reliable evidence.** Fault-only type/location metrics measure diagnosis given an event; detection-gated scores include misses. Ten distinct normals cannot establish low-FPR protection reliability.
7. **Scenario analysis measures DER-configuration dependence.** Configurations change PV, EV, V2G and sampled faults together. They do not constitute a pure controlled PV sweep.
8. **Resistance and sensor experiments stress the information available.** Resistance weakens many signatures. Sensor placements alter both locations and counts, while remote terminal currents must be removed to claim local PMU availability.
9. **Explainability supports diagnosis, not causality.** TreeSHAP shows which correlated electrical features moved a model's margin; it cannot prove a physical cause or prescribe a relay threshold.
10. **Deployment verifies reuse of frozen artifacts.** Inference uses the same schema and transform, validates units/channels, and reports honest score limitations. A dashboard does not certify protection safety.
11. **Documentation and Git preserve the evidence.** Keep original data out of Git, retain provenance and frozen IDs, and make every reported score traceable to predictions.

## Decisions to defend clearly

- **Normal fault-bus fields are ignored.** A bus label attached to a normal simulation does not mean there is a fault there. Including these labels would create an invalid localization task.
- **The split is grouped by load index across scenarios.** A random row split would reuse operating conditions and exact normal vectors. Deduplication removes exact overlap, but repeated pre-fault states still limit unseen-state claims. We state that limitation rather than assuming metadata groups are electrically independent.
- **Accuracy is not the only detection metric.** The study's post-deduplication test set is overwhelmingly faults. Fault recall, false alarms, normal counts and their uncertainty matter more than an isolated accuracy value. A zero observed FPR with ten normal events is weak evidence.
- **All released channels and local PMUs are different experiments.** The release has 726 current channels; only 387 are at the listed 93 PMU buses. The local-93 experiment isolates that distinction before reducing to 36/21/9 available PMUs.
- **B1 localization covers all 164 fault buses.** It ranks circuit-based candidate residuals, not just instrumented buses. However, approximate source settings and unconstrained fault currents can make transformer-related locations ambiguous. ML outperforming it does not establish superiority over all conventional fault-location methods.
- **Network error is branch hops.** This is a transparent source-graph distance. It is not an impedance-weighted electrical distance or geographic metres, and the expanded graph has 436 named buses because transformer secondary/source buses are explicit.
- **Temperature scaling has a narrow meaning.** Type/location confidence is adjusted using validation, then tested once. It describes this simulation benchmark's probabilities, not a utility's event prevalence or unseen networks. Detection has too few independent normals for a calibration claim.
- **The final test is frozen.** Do not change settings to fix a displayed final-test failure. Analyze it, record it, then design a new experiment with a new independent holdout if further development is necessary.

## Stronger next experiment

Obtain the exact dataset-generation code and modified OpenDSS files. Verify that applying the 60 load profiles changes solved electrical states as intended; repeat the audit with electrical-state fingerprints. Generate independent normal events with realistic noise and benign switching, then compare all methods at a meaningful false-alarm budget. Add a resistance-constrained conventional candidate-circuit location method to resolve transformer ambiguity, and validate the complete pipeline on a second feeder.

The final performance report, rather than these notes, is the source for numerical conclusions. Use its actual error counts and paired intervals when explaining whether the observed improvement is practically meaningful.
