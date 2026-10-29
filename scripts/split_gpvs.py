"""Create/check a fixed recording-level approximation to 60/20/20.

Uses identifiers only, never waveform values, labels or performance.
Existing assignments are never silently overwritten.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'config/recording_split.json'
SEED = 490


def proposed_split():
    # Six different scenarios supply one held-out recording each. Keeping the
    # opposite mode in training preserves scenario coverage there. F0 stays in
    # training because only two fully normal recordings exist.
    faults = sorted(range(1, 8), key=lambda i: hashlib.sha256(
        f'{SEED}:F{i}'.encode('ascii')).hexdigest())
    validation = [f'F{faults[0]}L', f'F{faults[3]}M', f'F{faults[4]}M']
    test = [f'F{faults[1]}L', f'F{faults[2]}L', f'F{faults[5]}M']
    all_ids = {f'F{i}{m}' for i in range(8) for m in 'LM'}
    train = sorted(all_ids - set(validation) - set(test))
    return {'seed': SEED, 'unit': 'recording',
            'target_fractions': {'train': .6, 'validation': .2, 'test': .2},
            'actual_fractions': {'train': 10/16, 'validation': 3/16, 'test': 3/16},
            'partitions': {'train': train, 'validation': sorted(validation), 'test': sorted(test)},
            'notes': ['No sample labels are assigned by this manifest.',
                      'F0L and F0M are training-only; held-out normal intervals need annotation.',
                      'Scenario modes may cross partitions; session independence is unverified.',
                      'Full-corpus EDA preceded this split; it is not a blind benchmark.']}


def validate_split(manifest):
    partitions = manifest['partitions']
    if set(partitions) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation and test partitions')
    flat = [name for names in partitions.values() for name in names]
    expected = {f'F{i}{mode}' for i in range(8) for mode in 'LM'}
    if len(flat) != 16 or len(set(flat)) != 16 or set(flat) != expected:
        raise ValueError('Recordings must appear exactly once: missing, duplicate or unknown ID')
    if [len(partitions[s]) for s in ('train', 'validation', 'test')] != [10, 3, 3]:
        raise ValueError('Expected counts 10/3/3')
    if {name[:2] for name in partitions['train']} != {f'F{i}' for i in range(8)}:
        raise ValueError('Training must contain all scenario IDs')
    if not {'F0L', 'F0M'} <= set(partitions['train']):
        raise ValueError('Documented normal recordings must stay in training for this design')
    for names in partitions.values():
        if {name[-1] for name in names} != {'L', 'M'}:
            raise ValueError('Each partition must contain both modes')
    return partitions


def load_split():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    return validate_split(manifest)


if __name__ == '__main__':
    proposed = proposed_split()
    validate_split(proposed)
    if MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text(encoding='utf-8'))
        validate_split(saved)
        if saved != proposed:
            raise ValueError('Saved split differs; preserving it. Review changes explicitly.')
    else:
        MANIFEST.parent.mkdir(exist_ok=True)
        MANIFEST.write_text(json.dumps(proposed, indent=2)+'\n', encoding='utf-8')
    for split, ids in proposed['partitions'].items():
        print(f'{split}: {len(ids)}/16 ({100*len(ids)/16:.2f}%): {", ".join(ids)}')
