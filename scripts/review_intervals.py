"""Render measurements for independent interval review; no detector or labels.

20 ms disjoint display bins summarize raw measurements (not model features).
Optional --start/--end permits closer inspection of measurement summaries.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gpvs_data import load_recordings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=int, choices=range(8))
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    args = parser.parse_args()
    recordings = load_recordings()
    out = ROOT / 'reports/figures/label_review'
    out.mkdir(parents=True, exist_ok=True)
    for scenario in ([args.scenario] if args.scenario is not None else range(1, 8)):
        fig, axes = plt.subplots(5, 2, figsize=(14, 12), layout='constrained')
        for j, mode in enumerate('LM'):
            name = f'F{scenario}{mode}'
            d = recordings[name]
            if args.start is not None:
                d = d[d.Time >= args.start]
            if args.end is not None:
                d = d[d.Time <= args.end]
            if d.empty:
                raise ValueError(f'No samples in requested interval for {name}')
            groups = np.floor(d.Time.to_numpy()/.02).astype(int)
            t = d.Time.groupby(groups).mean()
            for i, c in enumerate(['Ipv', 'Vpv', 'Vdc']):
                g = d[c].groupby(groups)
                axes[i,j].fill_between(t, g.min(), g.max(), color='#087f8c', alpha=.25)
                axes[i,j].plot(t, g.median(), color='#087f8c', linewidth=.8)
                axes[i,j].set_ylabel(c+' (native units)')
            for i, channels in [(3,['ia','ib','ic']), (4,['va','vb','vc'])]:
                for c in channels:
                    rms = np.sqrt(d[c].pow(2).groupby(groups).mean())
                    axes[i,j].plot(t, rms, label=c, linewidth=.8)
                axes[i,j].set_ylabel('Phase current RMS' if i==3 else 'Phase voltage RMS')
                axes[i,j].legend(ncol=3, fontsize=8, loc='upper right')
            axes[0,j].set_title(name+(' · IPPT' if mode=='L' else ' · MPPT'))
            for ax in axes[:,j]:
                ax.set_xlabel('Recorded time (s)')
                ax.grid(alpha=.2)
                ax.set_xlim(d.Time.iloc[0],d.Time.iloc[-1])
        fig.suptitle(f'F{scenario}: measurement review only — no inferred onset or model outputs\n'
                     '20 ms display bins; DC median + min/max; AC RMS includes DC component', fontsize=13)
        suffix='' if args.start is None and args.end is None else f'_{args.start}_{args.end}'
        path=out/f'F{scenario}{suffix}.png'
        fig.savefig(path,dpi=125)
        plt.close(fig)
        print(path.name,flush=True)


if __name__=='__main__':
    main()
