"""Reusable phasor features. No labels, IDs or DER metadata enter this transform."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAULT_TYPES = ["Normal", "LGA", "LGB", "LGC", "LLAB", "LLBC", "LLAC", "LLABG", "LLBCG", "LLACG", "LLLG"]


class PhasorFeatures:
    def __init__(self, columns, topology, sensors=None):
        self.columns = list(columns)
        self.topology = topology
        self.sensors = set(sensors) if sensors is not None else None
        self.lookup = {c:i for i,c in enumerate(columns)}
        self.v = [c[:-len("_Vmag_pre")] for c in columns if c.endswith("_Vmag_pre") and (sensors is None or c.split(".")[0] in self.sensors)]
        self.i = []
        for c in columns:
            if not c.endswith("_Imag_pre"):
                continue
            base = c[:-len("_Imag_pre")]
            # Terminal currents are available only at selected PMU buses.
            info = topology["current_phase_map"][base.lower()]
            if sensors is None or info["bus"] in self.sensors:
                self.i.append(base)
        self.vphase = [int(c.rsplit(".",1)[1]) for c in self.v]
        self.iphase = [topology["current_phase_map"][c.lower()]["phase"] for c in self.i]

    def phasors(self, a, bases, quantity, state):
        mags = a[:, [self.lookup[f"{c}_{quantity}mag_{state}"] for c in bases]]
        angles = a[:, [self.lookup[f"{c}_{quantity}ang_{state}"] for c in bases]]
        return mags * np.exp(1j*np.deg2rad(angles))

    @staticmethod
    def sequence(a):
        alpha = np.exp(2j*np.pi/3)
        return np.stack([(a[:,0]+a[:,1]+a[:,2])/3, (a[:,0]+alpha*a[:,1]+alpha**2*a[:,2])/3, (a[:,0]+alpha**2*a[:,1]+alpha*a[:,2])/3], axis=1)

    def transform(self, a, return_aux=False):
        a = np.asarray(a)
        vp = self.phasors(a, self.v, "V", "pre")
        vq = self.phasors(a, self.v, "V", "post")
        ip = self.phasors(a, self.i, "I", "pre")
        iq = self.phasors(a, self.i, "I", "post")
        vs = np.maximum(np.abs(vp), 1.0)
        # A one-amp floor prevents nearly unloaded lines dominating relative changes.
        isc = np.maximum(np.abs(ip), 1.0)
        dv = (vq-vp)/vs
        di = (iq-ip)/isc
        arrays, names = [], []
        for bases, p, q, scale, delta, prefix in [(self.v,vp,vq,vs,dv,"V"), (self.i,ip,iq,isc,di,"I")]:
            metrics = {"pre_mag":np.abs(p), "post_mag":np.abs(q), "mag_ratio":np.abs(q)/scale, "delta_re":delta.real, "delta_im":delta.imag}
            for metric, values in metrics.items():
                arrays.append(values)
                names.extend(f"{b}_{prefix}_{metric}" for b in bases)
        groups = {}
        for j,b in enumerate(self.v):
            groups.setdefault(b.rsplit(".",1)[0], {})[self.vphase[j]]=j
        three_v = [g for g in groups.values() if set(g)=={1,2,3}]
        igroups = {}
        for j,b in enumerate(self.i):
            group = b.rsplit("_Ph",1)[0]
            igroups.setdefault(group,{})[self.iphase[j]]=j
        three_i = [g for g in igroups.values() if set(g)=={1,2,3}]
        vseq0, vseq2, iseq0 = [], [], []
        # Only calculate sequence components for co-located physical A/B/C triplets.
        for prefix, bases, p, q, groups3 in [("V",self.v,vp,vq,three_v),("I",self.i,ip,iq,three_i)]:
            for g in groups3:
                ix = [g[k] for k in [1,2,3]]
                sp, sq = self.sequence(p[:,ix]), self.sequence(q[:,ix])
                norm = np.maximum(np.abs(sp[:,1]), 1.0)
                ds = np.abs(sq-sp)/norm[:,None]
                arrays.append(ds)
                stem = bases[ix[0]].rsplit(".",1)[0] if prefix=="V" else bases[ix[0]].rsplit("_Ph",1)[0]
                names.extend(f"{stem}_{prefix}_delta_seq{k}" for k in range(3))
                if prefix == "V":
                    vseq0.append(ds[:,0]); vseq2.append(ds[:,2])
                else:
                    iseq0.append(np.abs((sq-sp)[:,0]) / np.maximum(np.linalg.norm((sq-sp),axis=1),1e-6))
        x = np.concatenate(arrays, axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Feature transform produced invalid values")
        self.feature_names_ = names
        if not return_aux:
            return x
        phase_dv = np.column_stack([np.max(np.abs(dv[:,np.array(self.vphase)==p]),axis=1) if p in self.vphase else np.zeros(len(a)) for p in [1,2,3]])
        phase_di = np.column_stack([np.max(np.abs(di[:,np.array(self.iphase)==p]),axis=1) if p in self.iphase else np.zeros(len(a)) for p in [1,2,3]])
        aux = {"dv":dv, "di":di, "vp":vp, "ip":ip, "vq":vq, "iq":iq, "phase_dv":phase_dv, "phase_di":phase_di,
               "voltage_disturbance":np.max(np.abs(dv),axis=1), "current_disturbance":np.max(np.abs(di),axis=1),
               "max_sag":np.max(1-np.abs(vq)/vs,axis=1), "max_current":np.max(np.abs(iq),axis=1),
               "v0":np.max(vseq0,axis=0) if vseq0 else np.zeros(len(a)), "v2":np.max(vseq2,axis=0) if vseq2 else np.zeros(len(a)),
               "i0_fraction":np.max(iseq0,axis=0) if iseq0 else np.zeros(len(a))}
        return x, aux


def load_cache():
    cache = ROOT / "results/cache"
    a = np.load(cache / "raw.npy", mmap_mode="r")
    m = pd.read_csv(cache / "metadata.csv")
    cols = json.loads((cache / "columns.json").read_text())
    topology = json.loads((ROOT / "config/topology.json").read_text())
    return a, m, cols, topology
