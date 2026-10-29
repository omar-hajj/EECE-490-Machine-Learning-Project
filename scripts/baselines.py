"""Provisional normal-reference rules, fitted only on accepted training normals."""
import json
import hashlib
import numpy as np
import pandas as pd
from gpvs_data import ROOT
from split_gpvs import load_split

CURRENT = ['ia_rms', 'ib_rms', 'ic_rms']
VOLTAGE = ['Vdc_mean', 'va_rms', 'vb_rms', 'vc_rms']
SPREAD = ['current_rms_spread', 'voltage_rms_spread']


def fit_rules(train, allowed_recordings):
    if not set(train.recording) <= set(allowed_recordings):
        raise ValueError('Non-training recording supplied for calibration')
    normal = train.loc[train.label.eq(0)]
    if normal.empty:
        raise ValueError('Accepted training normal windows required')
    if not np.isfinite(normal[CURRENT + VOLTAGE + SPREAD].to_numpy()).all():
        raise ValueError('Nonfinite calibration input')
    return {
        'status': 'provisional empirical normal envelope; not protection ratings',
        'consecutive_windows': 3,
        'calibration_recordings': sorted(normal.recording.unique()),
        'calibration_windows': len(normal),
        'B0_upper': {c: float(normal[c].max()) for c in CURRENT},
        'B1_lower': {c: float(normal[c].min()) for c in VOLTAGE},
        'B1_upper': {c: float(normal[c].max()) for c in VOLTAGE + SPREAD},
    }


def persistent_alarm(flags, end_samples, stride, count):
    """Reset on any false flag or missing step; no state crosses recordings."""
    run, previous, output = 0, None, []
    for flag, sample in zip(flags, end_samples):
        if previous is not None and sample - previous != stride:
            run = 0
        run = run + 1 if flag else 0
        output.append(run >= count)
        previous = sample
    return np.asarray(output, dtype=bool)


def predict_rules(table, model, stride):
    if not np.isfinite(table[CURRENT + VOLTAGE + SPREAD].to_numpy()).all():
        raise ValueError('Nonfinite rule input')
    outputs = []
    for name, group in table.groupby('recording', sort=True):
        group = group.sort_values('end_sample')
        if group.end_sample.duplicated().any():
            raise ValueError('Duplicate window endpoint')
        b0 = np.zeros(len(group), dtype=bool)
        b1 = np.zeros(len(group), dtype=bool)
        for channel, bound in model['B0_upper'].items():
            b0 |= group[channel].to_numpy() > bound
        for channel, bound in model['B1_lower'].items():
            b1 |= group[channel].to_numpy() < bound
        for channel, bound in model['B1_upper'].items():
            b1 |= group[channel].to_numpy() > bound
        result = group[['recording', 'start_s', 'end_s', 'end_sample', 'label']].copy()
        for rule, flags in [('B0', b0), ('B1', b1)]:
            result[rule + '_candidate'] = flags
            result[rule + '_alarm'] = persistent_alarm(
                flags, group.end_sample, stride, model['consecutive_windows'])
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def main():
    directory = ROOT / 'data/features'
    manifest = json.loads((directory / 'manifest.json').read_text())
    for relative, expected in manifest['input_sha256'].items():
        with (ROOT / relative).open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
                raise ValueError('Stale features: rerun extraction before calibration')
    train = pd.read_csv(directory / 'train.csv')
    model = fit_rules(train, load_split()['train'])
    model['feature_manifest'] = manifest
    destination = ROOT / 'data/baselines'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'rules.json').write_text(json.dumps(model, indent=2) + '\n')
    # Deliberately only produce training diagnostics at this stage.
    predictions = predict_rules(train, model, manifest['settings']['stride_samples'])
    predictions.to_csv(destination / 'train_predictions.csv', index=False)
    print(f"Calibrated on {model['calibration_windows']} accepted normal training windows")
    print('Recordings: ' + ', '.join(model['calibration_recordings']))
    print('Saved provisional rules and training diagnostics; no held-out evaluation.')


if __name__ == '__main__':
    main()
