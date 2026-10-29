"""Read the supplied archive without extracting or changing raw recordings.

This is an audit only: no labels, windows, splits, or models are generated.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd
import h5py


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('Data for EECE 490.zip'))
    parser.add_argument('--output', type=Path, default=Path('reports'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.archive.read_bytes()
    manifest = [{'path': args.archive.name, 'bytes': len(raw), 'sha256': sha(raw)}]
    inventory, channels, mats = [], [], []
    mat_arrays = {}
    expected = 'Time Ipv Vpv Vdc ia ib ic va vb vc Iabc If Vabc Vf'.split()
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        for name in outer.namelist():
            if outer.getinfo(name).is_dir():
                continue
            blob = outer.read(name)
            manifest.append({'path': name, 'bytes': len(blob), 'sha256': sha(blob)})
            if name.endswith('.mat'):
                with h5py.File(io.BytesIO(blob), 'r') as obj:
                    cells = [v for v in obj['#refs#'].values()
                             if isinstance(v, h5py.Dataset) and v.shape == (14, 1)
                             and h5py.check_dtype(ref=v.dtype) is not None]
                    names, arrays = None, None
                    for cell in cells:
                        targets = [obj[r] for r in cell[:, 0]]
                        if all(v.attrs.get('MATLAB_class') == b'char' for v in targets):
                            names = [''.join(chr(int(c)) for c in v[()].ravel()) for v in targets]
                        elif all(v.dtype == np.dtype('float64') and v.size > 100 for v in targets):
                            arrays = np.column_stack([v[()].ravel() for v in targets])
                    assert names == expected and arrays is not None, name
                    mat_arrays[Path(name).stem] = arrays
                    unit_fields = []
                    obj.visititems(lambda path, item: unit_fields.append(
                        {'path': path, 'empty': bool(item.attrs.get('MATLAB_empty', 0))})
                        if path.endswith('/VariableUnits') else None)
                    mats.append({'file': name, 'format': 'MATLAB v7.3 HDF5 table',
                                 'shape': list(arrays.shape), 'columns': names, 'dtype': str(arrays.dtype),
                                 'nonfinite': int((~np.isfinite(arrays)).sum()),
                                 'units_metadata': unit_fields,
                                 'header': blob[:116].decode('ascii', errors='replace').strip()})
            if name != 'CSV_Files.zip':
                continue
            with zipfile.ZipFile(io.BytesIO(blob)) as inner:
                for member in sorted(inner.namelist()):
                    if inner.getinfo(member).is_dir():
                        continue
                    csvraw = inner.read(member)
                    manifest.append({'path': name + '/' + member, 'bytes': len(csvraw), 'sha256': sha(csvraw)})
                    if not member.endswith('.csv'):
                        continue
                    frame = pd.read_csv(io.BytesIO(csvraw))
                    assert list(frame.columns) == expected, (member, list(frame.columns))
                    values = frame.to_numpy(dtype=float)
                    t = values[:, 0]
                    dt = np.diff(t)
                    signals = values[:, 1:]
                    scenario = Path(member).stem
                    mat = mat_arrays[scenario]
                    assert mat.shape == values.shape
                    delta = np.abs(mat-values)
                    assert np.allclose(mat, values, rtol=1e-12, atol=1e-12), scenario
                    next(m for m in mats if m['file'] == scenario+'.mat').update(
                        csv_max_absolute_difference=float(delta.max()),
                        csv_equal_within_rtol_atol_1e_12=True)
                    inventory.append(dict(
                        recording=scenario, mode='IPPT' if scenario[-1] == 'L' else 'MPPT',
                        mat_format='MATLAB v7.3 HDF5 table',
                        mat_csv_max_abs_difference=float(delta.max()),
                        mat_csv_equal_rtol_atol_1e_12=True,
                        mat_units_empty=all(u['empty'] for u in next(m for m in mats if m['file'] == scenario+'.mat')['units_metadata']),
                        scenario=scenario[:2], rows=len(frame), columns=len(frame.columns),
                        start_s=t[0], end_s=t[-1], duration_s=t[-1]-t[0],
                        dt_min_s=dt.min(), dt_median_s=np.median(dt), dt_mean_s=dt.mean(), dt_max_s=dt.max(),
                        dt_std_s=dt.std(), fs_mean_hz=1/dt.mean(), nonpositive_steps=int((dt <= 0).sum()),
                        gaps_gt_1_5_median=int((dt > 1.5*np.median(dt)).sum()),
                        steps_deviating_over_1pct=int((np.abs(dt/np.median(dt)-1) > .01).sum()),
                        missing=int(frame.isna().sum().sum()), nonfinite=int((~np.isfinite(values)).sum()),
                        duplicate_rows=int(frame.duplicated().sum()),
                        duplicate_signal_rows=int(frame.iloc[:, 1:].duplicated().sum()),
                        consecutive_equal_signal_rows=int(np.all(signals[1:] == signals[:-1], axis=1).sum()),
                        signal_array_sha256=sha(np.ascontiguousarray(signals).tobytes()),
                        midpoint_reference_only_s=(t[0]+t[-1])/2))
                    for col in frame:
                        x = frame[col].to_numpy()
                        counts = frame[col].value_counts()
                        equal = x[1:] == x[:-1]
                        boundaries = np.r_[0, np.flatnonzero(~equal)+1, len(x)]
                        channels.append(dict(recording=scenario, channel=col, dtype=str(x.dtype),
                                             min=float(np.nanmin(x)), max=float(np.nanmax(x)),
                                             mean=float(np.nanmean(x)), std=float(np.nanstd(x)),
                                             unique=int(frame[col].nunique()), zero_count=int((x == 0).sum()),
                                             missing=int(frame[col].isna().sum()), nonfinite=int((~np.isfinite(x)).sum()),
                                             consecutive_repeats=int(equal.sum()), longest_equal_run=int(np.diff(boundaries).max()),
                                             most_common_value=float(counts.index[0]), most_common_count=int(counts.iloc[0])))
                    print(scenario, len(frame), 'rows,', round(1/dt.mean(), 3), 'Hz', flush=True)
    pd.DataFrame(inventory).to_csv(args.output / 'recording_inventory.csv', index=False)
    pd.DataFrame(channels).to_csv(args.output / 'channel_statistics.csv', index=False)
    (args.output / 'checksums.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output / 'audit_environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        h5py=h5py.__version__), indent=2), encoding='utf-8')
    assert len(inventory) == 16 and len(mats) == 16, 'Unexpected recording count'
    assert sha(args.archive.read_bytes()) == manifest[0]['sha256'], 'Raw archive changed'


if __name__ == '__main__':
    main()
