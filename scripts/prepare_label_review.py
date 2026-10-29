"""Build a review packet and label coverage check without modifying annotations."""
import pandas as pd
from gpvs_data import ROOT
from interval_labels import load_intervals, window_labels
from split_gpvs import load_split


def main():
    intervals = load_intervals()
    register = pd.read_csv(ROOT / 'annotations/recordings.csv').set_index('recording')
    sections = [
        '# Label review worksheet',
        'This is a review aid, not a source of accepted labels. No decisions have been filled in.',
        'Review measurements and the paper without looking at model alarms. Record your own '
        'interval boundaries and evidence first, then compare with the assistant proposals below. '
        'A visible change alone does not prove a fault, and a stable trace does not prove normality. '
        'Leave uncertain times unknown. The plots summarize 20 ms bins; inspect raw samples for '
        'short events or unclear transitions.',
        'For each recording, record two independent assessments and reconcile disagreements. '
        'These are waveform-based labels, not exact physical injection timestamps. '
        'The two-reviewer convention is our project quality procedure, not a stated course requirement.',
        '## Decisions to record',
        'For each proposed or new interval, provide: start/end seconds; normal, fault or unknown; '
        'reviewer names; evidence with waveform/channel details and source page; and whether '
        'the decision agrees with, revises or rejects the proposal. Split intermittent events '
        'into separate intervals. Never fill unknown gaps automatically.',
    ]
    coverage = []
    for partition, names in load_split().items():
        table = pd.read_csv(ROOT / f'data/features/{partition}.csv')
        if set(table.recording) != set(names):
            raise ValueError('Feature recording assignments differ from frozen split')
        for name in names:
            windows = table.loc[table.recording.eq(name)]
            labels = window_labels(name, windows.start_s, windows.end_s, intervals)
            coverage.append(dict(partition=partition, recording=name, windows=len(labels),
                                 normal=int(labels.eq(0).sum()), fault=int(labels.eq(1).sum()),
                                 unknown=int(labels.isna().sum())))
            if name.startswith('F0'):
                continue
            scenario = name[:2]
            sections.extend([
                f'## {name} — {register.loc[name, "mode"]}',
                str(register.loc[name, 'scenario_description']),
                f'[Paired overview](figures/label_review/{scenario}.png). '
                'L is IPPT; M is MPPT. Check plot titles to identify the intended recording.',
                'Current suggestions (all remain unaccepted):',
                '| Start (s) | End (s) | Suggested state | Status |\n'
                '| ---: | ---: | --- | --- |',
                '\n'.join(f'| {r.start_s:.6f} | {r.end_s:.6f} | {r.state} | {r.status} |'
                          for r in intervals.loc[intervals.recording.eq(name)].itertuples()),
                '**Reviewer 1:** pending. **Reviewer 2:** pending.\n\n'
                '**Independent boundaries and evidence:** pending.\n\n'
                '**Adjudicated decision and remaining uncertainty:** pending.',
            ])
    sections.extend([
        '## Applying completed reviews',
        'Keep this generated worksheet as a template; save completed reviewer notes separately '
        'because regeneration replaces this file. Only after evidence-backed review, transfer '
        'agreed decisions into annotations/intervals.csv, using reviewed status and distinct '
        'reviewer identities. Preserve evidence and keep unresolved intervals unknown. '
        'Rerun interval_labels.py, extract_features.py, then baselines.py. '
        'Do not approve regions simply to make training possible.',
        'Independent session verification, startup settling and calibration limitations remain '
        'separate from this annotation check. Reviewing only visibly obvious faults introduces '
        'selection bias. Retain the frozen split and do not inspect test alarm performance to tune choices.',
    ])
    (ROOT / 'reports/LABEL_REVIEW_WORKSHEET.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    report = pd.DataFrame(coverage)
    totals = report.groupby('partition')[['windows', 'normal', 'fault', 'unknown']].sum()
    lines = ['# Current label readiness',
             'Recomputed from current interval annotations and existing feature-window bounds. '
             'Class presence is necessary but does not establish label quality or adequate coverage.',
             '| Partition | Windows | Normal | Fault | Unknown | Both classes present? |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for part, row in totals.iterrows():
        ready = row.normal > 0 and row.fault > 0
        lines.append(f'| {part} | {row.windows} | {row.normal} | {row.fault} | {row.unknown} | {"Yes" if ready else "No"} |')
    lines.append('\nNo positive labels are invented by this check. Unknown windows must not be '
                 'treated as normal. Use LABEL_REVIEW_WORKSHEET.md to record independent evidence. '
                 'Regenerate feature tables after any accepted annotation changes.')
    (ROOT / 'reports/LABEL_READINESS.md').write_text('\n\n'.join(lines[:2]) + '\n\n' + '\n'.join(lines[2:]) + '\n', encoding='utf-8')
    print(totals.to_string())
    print('Review packet and readiness report saved. Annotations unchanged.')


if __name__ == '__main__':
    main()
