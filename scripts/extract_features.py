"""Past-only recording-local features; no fitting, imputation or inferred labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpvs_data import ROOT, load_partitioned_recordings, model_input_view
from interval_labels import window_labels


def extract_recording(frame, recording, settings):
    size, stride = settings['window_samples'], settings['stride_samples']
    if not isinstance(size, int) or not isinstance(stride, int) or size < 2 or stride < 1:
        raise ValueError('Window and stride must be positive integer sample counts')
    warmup = settings['startup_exclusion_s']
    if not np.isfinite(warmup) or warmup < 0:
        raise ValueError('Invalid startup exclusion')
    times = frame.Time.to_numpy(dtype=float)
    values = model_input_view(frame)
    if (len(times) < size or not np.isfinite(times).all() or
            not (np.diff(times) > 0).all() or not np.isfinite(values.to_numpy()).all()):
        raise ValueError('Need finite measurements, increasing times and a complete window')
    rows = []
    for end in range(size, len(frame) + 1, stride):
        begin = end - size
        if times[begin] < times[0] + warmup:
            continue
        block = values.iloc[begin:end]
        row = dict(recording=recording, start_sample=begin, end_sample=end - 1,
                   start_s=times[begin], end_s=times[end - 1],
                   observed_span_s=times[end - 1] - times[begin])
        for channel in ('Ipv', 'Vpv', 'Vdc'):
            x = block[channel].to_numpy()
            row[channel + '_mean'] = float(x.mean())
            row[channel + '_std'] = float(x.std(ddof=0))
        row['pv_power_mean'] = float((block.Ipv * block.Vpv).mean())
        for group, channels in [('current', ['ia', 'ib', 'ic']),
                                ('voltage', ['va', 'vb', 'vc'])]:
            rms = np.sqrt(np.mean(block[channels].to_numpy() ** 2, axis=0))
            for channel, value in zip(channels, rms):
                row[channel + '_rms'] = float(value)
            # Absolute spread stays defined when all phases are zero.
            row[group + '_rms_spread'] = float(rms.max() - rms.min())
        rows.append(row)
    if not rows:
        raise ValueError('No complete windows after startup exclusion')
    result = pd.DataFrame(rows)
    result['label'] = window_labels(recording, result.start_s, result.end_s)
    return result


METADATA = ['recording', 'start_sample', 'end_sample', 'start_s', 'end_s',
            'observed_span_s', 'label']


def main():
    settings = json.loads((ROOT / 'config/features.json').read_text())
    destination = ROOT / 'data/features'
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    feature_columns = None
    for partition, recordings in load_partitioned_recordings().items():
        frames = []
        for name, frame in recordings.items():
            result = extract_recording(frame, name, settings)
            frames.append(result)
            summary.append(dict(partition=partition, recording=name, windows=len(result),
                                normal=int(result.label.eq(0).sum()),
                                fault=int(result.label.eq(1).sum()),
                                unknown=int(result.label.isna().sum())))
        table = pd.concat(frames, ignore_index=True)
        feature_columns = [c for c in table if c not in METADATA]
        if not np.isfinite(table[feature_columns].to_numpy()).all():
            raise ValueError('Nonfinite feature output')
        table.to_csv(destination / f'{partition}.csv', index=False)
    manifest = {'settings': settings, 'predictor_columns': feature_columns,
                'metadata_columns': METADATA, 'input_sha256': {}}
    for relative in ['Data for EECE 490.zip', 'config/recording_split.json',
                     'config/input_columns.json', 'config/features.json',
                     'annotations/intervals.csv', 'scripts/extract_features.py']:
        with (ROOT / relative).open('rb') as source:
            manifest['input_sha256'][relative] = hashlib.file_digest(source, 'sha256').hexdigest()
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = pd.DataFrame(summary)
    report.to_csv(ROOT / 'reports/feature_inventory.csv', index=False)
    print(report.groupby('partition')[['windows', 'normal', 'fault', 'unknown']].sum().to_string())
    print(f'{len(feature_columns)} predictors; no model fitted. Outputs: {destination}')


if __name__ == '__main__':
    main()
