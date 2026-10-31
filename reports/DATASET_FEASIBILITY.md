# Phase 1 — DER-Fault feasibility

Audit executed on 2026-10-31 with `scripts/audit_der.py` and `scripts/topology.py`. No models were trained for this report. All numerical findings below are computed from the downloaded release, not assumed from its description.

## Dataset identity and provenance

The initial project folder contained only `.git` and `dataverse_files.zip` (3,047,947,556 bytes). Its six members are `train.zip`, `test.zip`, `IEEE_34_node_HIF.pdf`, `data_read.py`, `data_explanation.pdf`, and `MANIFEST.TXT`. This is an unrelated IEEE 34-node HIF dataset; it is preserved and excluded from this project.

The correct files were downloaded from [DER-Fault release 1.0](https://zenodo.org/records/22904220). Every file's MD5 matches the release API, recorded in `der_provenance.json`. Measurements occupy 1,343,926,563 bytes; metadata 1,714,901 bytes. The release has six flat CSV/Markdown files. The API response is preserved locally. The data README declares CC BY 4.0; the landing page license field is less specific. Attribute the authors and consult their README before redistribution. Raw data and the source circuit are excluded from Git.

## Verified data structure

- 19,800 measurement rows, with unique `SampleID`; one-to-one join to 19,800 metadata rows.
- 3,604 electrical measurement columns, plus `SampleID`; 16 metadata columns including the join key.
- 60 distinct representative load indices/timestamps; 360 load/scenario combinations; 55 cases per combination (five for each class).
- 1,800 rows for each class: `Normal`, `LGA`, `LGB`, `LGC`, `LLAB`, `LLBC`, `LLAC`, `LLABG`, `LLBCG`, `LLACG`, `LLLG`.
- 1,800 normal and 18,000 fault rows. Fault-only bus labels cover **164 distinct buses**, with 87–143 cases per bus. Each fault type covers all 164 buses. Normal bus metadata has 429 distinct values and has no physical localization meaning.
- Fault resistance: 0.050–24.999 ohm; median 12.4665 ohm. Do not describe this as a field high-impedance-fault dataset or extrapolate beyond that range.

| Scenario | Rows | PV % | EV % | V2G % |
|---|---:|---:|---:|---:|
| S1 | 3300 | 0 | 0 | 0 |
| S2 | 3300 | 30 | 0 | 0 |
| S3 | 3300 | 50 | 20 | 0 |
| S4 | 3300 | 70 | 30 | 0 |
| S5 | 3300 | 70 | 30 | 30 |
| S6 | 3300 | 30 | 50 | 50 |

These are installed/configured percentages; the metadata profile multipliers vary with the selected operating hour. Scenario comparisons are observational, with randomly different bus/resistance draws, and do not isolate a causal PV effect. S4/S5 differ in V2G, not PV.

## Measurements and physical interpretation

There are 93 measured voltage buses: **41 have complete A/B/C measurements and 52 have one phase**, giving 175 voltage phase channels. There are 726 current conductor/terminal channels. Each channel has magnitude and angle before and after the event. These are steady-state phasors, not time-series waveforms: timing, relay clearing time, transient detection latency and inverter control dynamics cannot be evaluated.

Magnitudes are consistent with volts and amperes in the OpenDSS source, and angles with degrees. For example bus3003 median pre-fault A/B/C angles are about -30.63, -150.54 and 89.44 degrees, consistent with positive phase sequence. Voltage magnitude range is 143.94–9593.06 V; current 0–4713.94 A. Angles span approximately ±180 degrees. Wrap angles through complex exponentials; never subtract wrapped angles directly.

**Current `PhN` is a conductor ordinal, not always electrical phase N.** Source OpenDSS `NodeOrder` identifies 187 mismatches among its 930 line terminal/conductor mappings; **148 of the 726 released current channels** have a physical phase different from their column ordinal. The preprocessing uses this mapping before phase aggregation or symmetrical components. Sequence components are restricted to complete, co-located physical A/B/C triplets. Near-zero current angles are numerically defined but physically uninformative; complex currents and a one-amp normalization floor prevent arbitrary angle effects.

## Data quality and dependence

- Missing electrical values: 0; infinities: 0; negative magnitudes: 0; missing metadata: 0. All measurement columns match the expected phasor schema.
- There are **18,057 unique complete electrical vectors**, giving 1,743 redundant rows. No identical fault vector has conflicting type or bus labels.
- The 1,800 normal rows contain only **57 unique vectors**. Every normal row has exactly zero pre/post difference. Twenty duplicate-vector groups cross load indices. This makes naive detection scores optimistic and prevents treating 1,800 normal rows as independent false-alarm trials.
- Only 1,033 unique complete pre-fault vectors exist across all 19,800 cases; 581 pre-fault fingerprints are shared across load indices. Even the no-DER normal cases have only 21 exact vectors across 60 labeled load conditions. The release therefore does not demonstrate 60 electrically independent pre-fault operating states. This may reflect generation/state/application details not supplied with the release; do not assert an unverified simulator bug.
- A row split would share both load conditions and exact normal measurements across partitions. Group by load index across all scenarios, eliminate exact full-vector overlap, and document the reduced number of independent normal events. Shared pre-fault states remain a limitation even after group splitting; this is generalization to held-out metadata load groups, not proof of unseen electrical states.
- Retain all 164 location classes in each partition where possible; never interpret normal bus metadata as a target. Fault type/bus/resistance, row order, IDs, time, scenario and profile metadata must not enter predictive features.

## Topology and limited sensors

The release does not distribute a circuit model. The [original Iowa source](https://wzy.ece.iastate.edu/Testsystem.html) supplies an OpenDSS archive, downloaded separately. It compiles and solves successfully. Its expanded model contains **436 buses, 915 non-ground nodes and 440 line/transformer elements**, rather than literally 240 total named OpenDSS buses: secondary and source/regulator buses are explicit.

All measured voltage nodes, monitored line names and faulted buses match this source model. This supports a source-topology diagram and shortest-path **branch-hop** error, counting line/transformer branches. Branch hops are a topological distance, not impedance distance or geographic metres. Study-specific tap settings, modified impedances and generation scripts are unavailable; source-model transfer-impedance localization is an approximate conventional comparator, not an exact reconstruction of the study circuit. Verify its development-set residuals and retain that limitation in comparisons.

Published PMU lists are not nested. The 56-, 36- and 20-PMU configurations contain only **36, 21 and 9 voltage buses**, respectively, among released voltage channels. Exact experiments on those published placements are impossible from this release. Use their intersections with the released placement and label them explicitly as 36/21/9 available PMUs. Current availability must also be limited to terminals physically located at selected PMU buses, rather than keeping all line currents by accident.

The released full matrix itself includes both terminal currents on monitored lines. Only **387 of 726 current channels** are physically at the published PMU_93 buses; 339 are at 147 additional terminal buses. Consequently, the all-released-channel experiment is not literally a local 93-PMU system. A separate `available_93` experiment retains only the 387 local currents and supplies the proper reference for 36/21/9-PMU reductions. Both baselines and ML use identical information in each experiment.

## Feasibility decision and scope

| Objective | Decision | Defensible evaluation |
|---|---|---|
| Detection | Supported with major realism limitations | Normal/fault metrics on deduplicated held-out groups; report normal count and uncertainty. Low-FPR claims unsupported if normal count is too small. No field false-alarm claim. |
| Classification | Supported | Ten fault classes, fault-only conditional metrics and detection-gated end-to-end results. Grounded vs ungrounded phase pairs require zero-sequence information. |
| Localization | Supported on 164 labeled buses | Exact/top-k accuracy and source-topology hops. A physics-based candidate-current-injection residual method can cover the same buses. Unknown/unlabeled buses remain unsupported. |

Proceed with controlled experiments, but constrain conclusions to this simulated release. The research question is answerable as an empirical comparison on the available cases; it cannot establish certified protection performance, causal PV impact, transient response, or broad loading generalization. A strong B1 result is a valid research outcome.

## Source references

- [Dataset README](https://zenodo.org/records/22904220/files/README.md?download=1)
- [Dataset generation paper](https://arxiv.org/abs/2609.36247)
- [Original circuit model](https://wzy.ece.iastate.edu/publication/OpenDSS%20Model.zip)
- [OpenDSSDirect API](https://dss-extensions.org/OpenDSSDirect.py/opendssdirect.html)

Next action: freeze the grouped/deduplicated manifest before EDA, inspect development-set electrical signatures, then implement and tune conventional baselines before LightGBM.
