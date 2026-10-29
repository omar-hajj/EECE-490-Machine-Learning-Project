# Stage 4: verified fault mechanisms and timing limits

Update 30 October: first-pass measurement review is recorded in `INTERVAL_REVIEW.md`
and `annotations/intervals.csv`. Six fault recordings have provisional interior
regions, eight remain unresolved, and no positive label has been accepted yet.

29 October 2026. The supplied `paper.pdf` is the original 16-page paper by Bakdi
et al., IJEPES 125 (2021), 106457, DOI 10.1016/j.ijepes.2020.106457. Its title,
authors and DOI match the dataset reference. Text was read and relevant pages
rendered to verify Table 1 and the timing statements. Its fingerprint and page
references are in `annotations/paper_provenance.json`.

**Scenario meanings are now verified. Exact physical onset labels are not supplied
by the paper.** Source review is complete; supervised interval annotation is not.

## Verified mapping

Table 1 on page 7 gives the mapping below. The dataset naming convention applies
these mechanisms to both L/IPPT and M/MPPT recordings.

| ID | Mechanism | Documented disturbance |
| --- | --- | --- |
| F0 | Fault-free experiment | Established by the dataset README |
| F1 | Inverter fault | Complete failure of one of six IGBTs |
| F2 | Current feedback sensor fault | 20% error in one phase sensor |
| F3 | Grid anomaly | Intermittent voltage sags |
| F4 | PV array mismatch | 10-20% nonuniform partial shading |
| F5 | PV array mismatch | 15% open circuit in the PV array |
| F6 | MPPT/IPPT controller fault | Boost-converter PI gain decreased by 20% |
| F7 | Boost converter controller fault | PI time constant increased by 20% |

F2 is identified as a current feedback sensor in the text on pages 5 and 12 and
Figure 10's caption on page 13. The phase and sign of its error are not specified.
All 14 fault-recording descriptions in `annotations/recordings.csv` are updated
with source references. F0 normal bounds are preserved. Blank onset fields mean
unknown, never zero or normal. A scenario ID is not a per-row fault label.

## Timing evidence

- Page 5 section 2.2 describes manual injections generally around seconds 7-8.
  This is approximate, not a guaranteed interval for every exported recording.
- Page 11 section 4.2 describes the F6 example around 11 seconds, without an
  explicit L/M filename. Do not assign 11 seconds to both F6 recordings.
- Pages 10 and 13 explicitly acknowledge imprecise physical onset timing and the
  resulting limitations on precise detection-rate and detection-delay assessment.
- FD/FA markers in Figures 9-12 are detector outputs, not independent injection
  timestamps. They have not been copied into ground-truth annotations.
- F3 is intermittent: a single onset cannot label every later sag/recovery interval.
  F6's persistence discussion on page 11 does not establish continuous faults for
  all other scenarios.
- Our F3M recording ends around 6.996 seconds, so a universal 7-8-second cutoff
  cannot safely be imposed on this corpus.

The 3.3-second normal reference period on page 10 describes the paper's training
procedure. It does not independently annotate the first 3.3 seconds of every
exported fault recording. No numerical onset bounds or faulty intervals have
been fabricated from these general statements.

## Other findings

**Sampling:** pages 5 and 10 state 100 microseconds, about 10 kHz, agreeing with
our timestamp audit. The README's 9.9989-microsecond value conflicts with both.
Use actual timestamps and nominal 10 kHz; preserve jitter and the F2L rate anomaly.

**Independence:** pages 5 and 10 describe independent experiments, supporting
recording-level grouping in our fixed 10/3/3 split. Session IDs and repeat metadata
are still absent, and all recordings come from one experimental setup. This is
not evidence of independent sites or broad deployment coverage.

**Estimated channels:** page 5 describes a virtual PLL positive-sequence PMU,
a one-cycle running Fourier window, reporting rate set to 64, and interpolation/
extrapolation to the acquisition grid. The original method uses seven signals:
Ipv, Vpv, Vdc, Iabc, If, Vabc and Vf. The four estimates are not established as
unnecessary; retain them for a later controlled comparison. Our nine measured
inputs remain provisional while startup, exported alignment and total latency
are verified. No raw columns have been deleted.

## Next annotation step

Use experiment logs if available, or have two team reviewers inspect raw
multichannel measurements independently of model predictions. Record confirmed
normal intervals, faulty intervals, unknown intervals, supporting channels,
reviewer names and disagreement. Separate recovery, intermittent faults and
shutdown. Use numerical onset ranges only where evidence supports them.

Waveform-based annotations identify observable signal changes, not necessarily
physical injection. Freeze them before training and report their provenance and
uncertainty. Do not claim verified 100 ms physical-injection detection from those
annotations. Ambiguous cases remain unresolved. If adequate annotations cannot
be established, narrow the task to normal-reference anomaly monitoring with
qualitative event evaluation rather than an unsupported supervised benchmark.

Causal features may be prepared within the frozen partitions without inventing
labels or fitting models. The fixed approximate 60/20/20 split is unchanged.

## Stage summary

- Done: primary-paper review, verified mechanisms, nominal sampling confirmation,
  independence and PMU evidence, updated annotation register and paper fingerprint.
- Remaining: human-reviewed intervals or author logs for supervised evaluation.
- Next: interval review; causal feature preparation can proceed independently.
- Unchanged: raw files, split assignments, no trained models or performance claims.

The user-supplied paper stays local and is excluded from Git. Cite its DOI in
submissions rather than redistributing the journal PDF.
