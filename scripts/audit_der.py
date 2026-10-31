"""Phase 1: inspect originals, verify provenance, and audit every measurement row.

Does not train models. Numeric cache retains original column order and SampleID.
Run from the repository root: python scripts/audit_der.py
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/der_fault"
OUT = ROOT / "reports"
CACHE = ROOT / "results/cache"


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def digest(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    record = json.loads((RAW / "zenodo_record.json").read_text(encoding="utf-8-sig"))
    inventory = []
    for f in record["files"]:
        p = RAW / f["key"]
        expected = f["checksum"].split(":", 1)[1]
        actual = digest(p)
        inventory.append(dict(file=p.name, bytes=p.stat().st_size, md5=actual, expected_md5=expected, verified=actual == expected))
        print("Verified", p.name, actual == expected, flush=True)
    dump(OUT / "der_provenance.json", {"record": "https://zenodo.org/records/22904220", "files": inventory})
    if not all(f["verified"] for f in inventory):
        raise ValueError("Original-file checksum mismatch: do not proceed")

    m = pd.read_csv(RAW / "labels_metadata_DERfaults.csv")
    assert m.SampleID.is_unique and not m.isna().any().any()
    cols = next(csv.reader((RAW / "DERfaults_measurements.csv").open()))
    assert len(cols) == len(set(cols)) and cols[0] == "SampleID"
    n = len(m)
    # Float64 audit avoids hiding invalid or near-duplicate values through rounding.
    raw = np.lib.format.open_memmap(CACHE / "raw.npy", mode="w+", dtype="float64", shape=(n, len(cols) - 1))
    ids, hashes, prehashes, posthashes = [], [], [], []
    nan = np.zeros(len(cols) - 1, dtype=int)
    inf = nan.copy()
    negative = nan.copy()
    pre = np.array([i for i, c in enumerate(cols[1:]) if c.endswith("_pre")])
    post = np.array([i for i, c in enumerate(cols[1:]) if c.endswith("_post")])
    offset = 0
    for chunk in pd.read_csv(RAW / "DERfaults_measurements.csv", chunksize=500):
        a = chunk.iloc[:, 1:].to_numpy(dtype=float)
        size = len(a)
        raw[offset:offset + size] = a
        ids.extend(chunk.SampleID.to_list())
        nan += np.isnan(a).sum(axis=0)
        inf += np.isinf(a).sum(axis=0)
        negative += (a < 0).sum(axis=0)
        for row in a:
            hashes.append(hashlib.sha256(row.tobytes()).hexdigest())
            prehashes.append(hashlib.sha256(row[pre].tobytes()).hexdigest())
            posthashes.append(hashlib.sha256(row[post].tobytes()).hexdigest())
        offset += size
        if offset % 3000 == 0:
            print("Audited rows", offset, flush=True)
    raw.flush()
    assert offset == n
    assert len(set(ids)) == n and set(ids) == set(m.SampleID)
    m = m.set_index("SampleID").loc[ids].reset_index()
    m["measurement_hash"] = hashes
    m["prefault_hash"] = prehashes
    m["postfault_hash"] = posthashes
    m.to_csv(CACHE / "metadata.csv", index=False)
    dump(CACHE / "columns.json", cols[1:])
    feature_inventory = pd.DataFrame({"column": cols[1:], "missing": nan, "infinite": inf, "negative": negative})
    feature_inventory["quantity"] = feature_inventory.column.str.extract(r"_(Vmag|Vang|Imag|Iang)_", expand=False)
    feature_inventory.to_csv(OUT / "der_feature_inventory.csv", index=False)
    voltages = [c for c in cols[1:] if "_Vmag_pre" in c]
    vbuses = sorted(set(c.split(".")[0] for c in voltages))
    currents = [c for c in cols[1:] if "_Imag_pre" in c]
    unknown = [c for c in cols[1:] if not re.fullmatch(r"(?:[^,]+)_(?:Vmag|Vang|Imag|Iang)_(?:pre|post)", c)]
    normal = m.Fault_Label.eq(0).to_numpy()
    dupe = m.groupby("measurement_hash").agg(n=("SampleID", "size"), labels=("Fault_Label", "nunique"), buses=("Fault_Bus", "nunique"), loads=("Representative_Load_Index", "nunique"))
    dupe[dupe.n > 1].to_csv(OUT / "der_duplicate_groups.csv")
    m.groupby("Fault_Type").size().rename("count").to_csv(OUT / "der_class_counts.csv")
    m.groupby(["Scenario_ID", "PV_Penetration", "EV_Penetration", "V2G_Participation"]).size().rename("count").to_csv(OUT / "der_scenario_counts.csv")
    m.groupby(["Representative_Load_Index", "Scenario_ID", "Fault_Type"]).size().rename("count").to_csv(OUT / "der_group_counts.csv")
    qstats = {}
    for q in ["Vmag", "Vang", "Imag", "Iang"]:
        ix = [i for i,c in enumerate(cols[1:]) if f"_{q}_" in c]
        a = np.asarray(raw[:, ix])
        qstats[q] = dict(zip(["min", "p01", "median", "p99", "max"], np.nanpercentile(a, [0, 1, 50, 99, 100]).tolist()))
    normal_change = np.abs(raw[normal][:, pre] - raw[normal][:, post])
    p = pd.read_csv(RAW / "pmu_placement_configurations.csv")
    pmus = {c: p[c].dropna().astype(str).tolist() for c in p}
    faults = m.loc[~normal]
    summary = {
        "samples": n, "measurement_columns": len(cols)-1, "metadata_columns": 16,
        "class_counts": m.Fault_Type.value_counts().to_dict(), "scenario_counts": m.Scenario_ID.value_counts().to_dict(),
        "load_conditions": int(m.Representative_Load_Index.nunique()), "load_scenario_groups": int(m.groupby(["Representative_Load_Index", "Scenario_ID"]).ngroups),
        "timestamp_unique": int(m.Timestamp.nunique()), "normal_rows": int(normal.sum()), "fault_rows": len(faults),
        "fault_buses": int(faults.Fault_Bus.nunique()), "fault_bus_counts_range": [int(faults.Fault_Bus.value_counts().min()), int(faults.Fault_Bus.value_counts().max())],
        "normal_bus_metadata_must_be_masked": True, "fault_resistance_ohm": faults.Resistance.describe().to_dict(),
        "missing_measurements": int(nan.sum()), "infinite_measurements": int(inf.sum()),
        "negative_magnitude_measurements": int(feature_inventory.loc[feature_inventory.quantity.isin(["Vmag", "Imag"]), "negative"].sum()),
        "unknown_measurement_columns": unknown, "voltage_phase_channels": len(voltages), "voltage_buses": len(vbuses), "current_phase_terminal_channels": len(currents),
        "pmu_lists": {k: len(v) for k,v in pmus.items()}, "unmeasured_requested_voltage_buses": {k: sorted(set(v)-set(vbuses)) for k,v in pmus.items()},
        "full_measurement_unique": len(set(hashes)), "full_measurement_duplicate_excess_rows": n-len(set(hashes)),
        "duplicate_measurements_conflicting_types": int(((dupe.n > 1) & (dupe.labels > 1)).sum()),
        "duplicate_measurements_conflicting_fault_buses": int(m.loc[~normal].groupby("measurement_hash").Fault_Bus.nunique().gt(1).sum()),
        "prefault_unique": len(set(prehashes)), "postfault_unique": len(set(posthashes)),
        "normal_max_raw_pre_post_difference": float(normal_change.max()),
        "measurement_ranges": qstats,
        "python": sys.version,
    }
    dump(OUT / "der_audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
