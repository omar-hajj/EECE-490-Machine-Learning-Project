"""Small read-only loader shared by EDA and future experiment code."""
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'Time Ipv Vpv Vdc ia ib ic va vb vc Iabc If Vabc Vf'.split()


def load_recordings(archive=None):
    """Return {recording_id: DataFrame}; never combine runs or invent row labels."""
    import pandas as pd

    archive = Path(archive) if archive is not None else ROOT / 'Data for EECE 490.zip'
    recordings = {}
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('CSV_Files.zip'))) as inner:
            for name in sorted(inner.namelist()):
                if not name.endswith('.csv'):
                    continue
                frame = pd.read_csv(inner.open(name))
                if list(frame.columns) != EXPECTED:
                    raise ValueError(f'Unexpected columns in {name}')
                recording_id = Path(name).stem
                if recording_id in recordings:
                    raise ValueError(f'Duplicate recording ID: {recording_id}')
                recordings[recording_id] = frame
    if set(recordings) != {f'F{i}{mode}' for i in range(8) for mode in 'LM'}:
        raise ValueError('Expected F0-F7 in both L/M modes')
    return recordings


def model_input_view(frame):
    """Return a COPY of nine provisional measurements; raw frame is untouched.

    Keep the original frame's Time separately for causal windows and latency.
    This function does not label, normalize, impute, split, or remove rows.
    """
    policy = json.loads((ROOT / 'config/input_columns.json').read_text(encoding='utf-8'))
    return frame.loc[:, policy['measurement_inputs']].copy()


def load_partitioned_recordings(archive=None):
    """Load whole recordings into the frozen train/validation/test partitions.

    Call before extracting windows. Each returned frame still has Time and all
    raw channels; it has no inferred fault labels or fitted preprocessing.
    """
    from split_gpvs import load_split

    assignments = load_split()
    recordings = load_recordings(archive)
    return {part: {name: recordings[name] for name in names}
            for part, names in assignments.items()}
