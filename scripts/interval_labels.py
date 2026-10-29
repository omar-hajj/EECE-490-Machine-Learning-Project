"""Validate interval annotations; proposals never become training labels.

Bounds are inclusive. Uncovered and unresolved times remain missing, not normal.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_intervals(intervals, inventory):
    required = {'recording','start_s','end_s','state','status','evidence','reviewer','second_reviewer'}
    if not required <= set(intervals.columns):
        raise ValueError('Missing interval metadata')
    bounds = inventory.set_index('recording')
    def present(value):
        return pd.notna(value) and bool(str(value).strip())
    for name, group in intervals.groupby('recording', dropna=False):
        if name not in bounds.index:
            raise ValueError(f'Unknown recording {name}')
        previous_end = -np.inf
        for row in group.sort_values('start_s').itertuples(index=False):
            if not np.isfinite([row.start_s,row.end_s]).all() or row.start_s > row.end_s:
                raise ValueError('Invalid interval bounds')
            if row.start_s < bounds.loc[name,'start_s'] or row.end_s > bounds.loc[name,'end_s']:
                raise ValueError('Interval outside recording')
            if row.start_s <= previous_end:
                raise ValueError('Overlapping inclusive intervals')
            previous_end = row.end_s
            if row.state not in {'normal','fault','unknown'}:
                raise ValueError('Invalid state')
            if row.status not in {'documented','reviewed','proposed','unresolved'}:
                raise ValueError('Invalid review status')
            if not present(row.evidence) or not present(row.reviewer):
                raise ValueError('Evidence and reviewer required')
            if row.status == 'documented' and (not name.startswith('F0') or row.state != 'normal'):
                raise ValueError('Only F0 normal status is documented by the supplied sources')
            if row.status == 'reviewed' and (not present(row.second_reviewer)
                                             or row.second_reviewer == row.reviewer):
                raise ValueError('Reviewed intervals require a distinct second reviewer')
    return intervals


def load_intervals():
    return validate_intervals(
        pd.read_csv(ROOT/'annotations/intervals.csv', keep_default_na=False),
        pd.read_csv(ROOT/'reports/recording_inventory.csv'))


def window_labels(recording, starts, ends, intervals=None):
    """Label only windows wholly contained in one accepted interval.

    Does not enforce warm-up eligibility or verify the truth of reviewer entries.
    Unknown labels must be excluded from supervised fitting and scoring.
    """
    inventory = pd.read_csv(ROOT/'reports/recording_inventory.csv')
    intervals = load_intervals() if intervals is None else validate_intervals(intervals, inventory)
    if recording not in set(inventory.recording):
        raise ValueError('Unknown recording')
    start, end = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    if (start.ndim != 1 or start.shape != end.shape or
            not np.isfinite(start).all() or not np.isfinite(end).all() or (start > end).any()):
        raise ValueError('Expected finite paired window bounds with start <= end')
    labels = pd.Series(pd.NA, index=range(len(start)), dtype='Int8')
    accepted = intervals[(intervals.recording == recording) &
                         intervals.status.isin(['documented','reviewed'])]
    for row in accepted.itertuples(index=False):
        if row.state in {'normal','fault'}:
            labels.loc[(start >= row.start_s) & (end <= row.end_s)] = int(row.state == 'fault')
    return labels


if __name__ == '__main__':
    intervals = load_intervals()
    print(intervals.groupby(['status','state']).size().to_string())
    print('Proposals and uncovered times remain unlabeled. No accepted positive labels yet.')
