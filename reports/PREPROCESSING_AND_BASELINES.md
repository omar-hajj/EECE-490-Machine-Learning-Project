# Preprocessing and conventional methods

## Partition and feature protocol

Seed 490 selects 36/12/12 training/validation/test representative load indices, stratified by load category. All six DER scenarios of each selected load index stay together. Seed 491 separates validation from test. Exact complete measurement vectors are retained once with train, then validation, then test priority. This removes 1,743 redundant normal rows without dropping a fault. The result is 10,839 training rows (39 normal), 3,608 validation rows (8 normal), and 3,610 test rows (10 normal). Each fault type has 1,080/360/360 rows. See `config/split.json` and `config/split_manifest.csv` for exact IDs.

Deduplication uses electrical values only. Labels are used for the inventory and split composition checks, not to select measurement features. Test measurement values are transformed by a deterministic transform but not used for fitted statistics, feature screening, model selection, threshold tuning or calibration. Shared pre-fault operating states remain a documented limitation. Test groups are held-out metadata loading groups, not proven independent electrical states.

The reusable `PhasorFeatures` transform calculates complex quantities as magnitude × exp(j × angle × π/180). It retains pre/post magnitudes, post/pre magnitude ratios and real/imaginary pre/post increments. Voltage and current normalization floors are 1 V and 1 A. It creates incremental sequence magnitudes at complete physical ABC triplets only, after mapping conductor order with OpenDSS NodeOrder. There are 5,132 full-placement features. There are no metadata predictors or bus-label-derived features. Absolute voltage level is meaningful across instrumented buses; no numerical bus ID is used as a regression target. Normal fault-bus values are masked by restricting localization to fault rows.

No missing-value imputation is required for this verified release. Nonfinite uploaded data is rejected rather than silently imputed. Fitted univariate feature screening for LightGBM is learned on training rows only and is stored with each model; all candidate selection uses validation.

## B0 — simple fixed protection rules

Trigger on a voltage magnitude sag of at least 10% at any measured phase OR on maximum measured current exceeding 72.4992 A (twice the training-normal 99th-percentile maximum current). The current scale is estimated from training only and then fixed. Actual CT ratios, thermal ratings and relay settings are not supplied. B0 is an intentionally simple reference, not a commissioned relay. Classification and localization are N/A.

## B1 — detection and phase/type rules

Detection uses the maximum normalized complex voltage increment, optionally combined with incremental current. Validation selects the weight and threshold from a prespecified grid. In this noiseless release it selects voltage alone at 10^-6 per unit. This very small setting exploits exact no-change normals and is **not a field protection threshold**. Real sensor-noise/transient events are absent, so the project cannot establish a usable relay pickup.

Classification aggregates maximum voltage/current changes for each physical phase, normalizes each three-phase response, and compares second-largest/largest and third-largest/largest ratios to select one, two or three affected phases. Two-phase faults require an additional ground discriminator: incremental voltage zero sequence relative to maximum phase disturbance and/or incremental residual current over valid ABC line triplets. Voltage/current weight, phase ratios, ground source and ground threshold are selected by validation macro-F1. All are electrical rules; no statistical classifier is fitted for B1.

The initial current-only ground discriminator was inadequate in development data. Adding the available voltage zero-sequence discriminator raised validation macro-F1 from 0.7360 to 0.9908. This improvement was made before any final test prediction. Retaining the stronger baseline is essential to the research question.

## B1 — network localization

The independently downloaded source circuit is solved in OpenDSS. The nodal admittance matrix Y includes the supplied source, load-admittance representation, transformer connections and default regulator settings. Its inverse Z has max |YZ−I| = 3.01×10^-8. The model is approximate because the dataset-specific modified circuit and solved tap/load states are not supplied.

For candidate bus b, incremental bus injection q produces ΔV ≈ Z[:,b] q. Monitored line current response follows ΔI = Yprim × ΔV_local. Combine measured voltage and current increments after scaling by training-median pre-fault magnitudes (floors 1 V and 5 A for network fitting). Candidate current phase support comes from B1's **predicted** type, never the true type. LL faults enforce equal-and-opposite phase injection; grounded multi-phase faults allow supported phase injections. QR/SVD subspace projection finds the least-squares residual for every candidate, and the smallest residual gives the bus. Rank the same residuals for top-3/top-5. Validation selects current weight among 0, 0.1 and 1; the chosen value is 0.1.

The candidate universe is the 164 source-model buses with all three phases, independently matching all 164 released fault labels. Coverage is therefore complete for the released task. This is a conventional model-based fault-location method, not a nearest-PMU rule or learned bus-template model. B1 receives the source circuit in addition to the same measurements; ML does not receive topology as features.

**Identifiability limitation:** fitted fault current is a free nuisance parameter. Transformer-related buses can have similar transfer subspaces, so this method can confuse primary and secondary fault locations. It does not impose the fault-resistance equation or recreate dataset-specific load/DER controls. Report its branch-hop errors, top-k performance, and transformer-pair confusion. A victory over this implementation does not prove superiority over every conventional impedance or state-estimation method. A future stronger study should reconstruct the exact generation model and add resistance-constrained candidate circuit solves.

## Evaluation safeguards

Conditional type/location metrics score actual fault rows, with predicted type used by B1 localization. End-to-end results additionally require detection to trigger. B0's unsupported tasks are N/A, never zero. Network distance is shortest branch hops in the verified source graph; it is not electrical impedance or geographic distance. Detection FPR has only ten distinct final normal cases, so a 1% FPR target is unsupported. Report uncertainty and counts rather than an empirical zero as evidence of field reliability.

Raw LightGBM probabilities are dataset-dependent scores. Type and location can use scalar validation temperature scaling; detection is too sparse in normal events for a reliable calibration claim. Confidence is not a certified protection probability.
