"""Freeze grouped development/test partitions, then create measurement-only features."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from features import ROOT, PhasorFeatures, load_cache


def split_manifest(m):
    loads = m[["Representative_Load_Index", "Load_Category"]].drop_duplicates()
    tr, rem = train_test_split(loads, train_size=.6, stratify=loads.Load_Category, random_state=490)
    va, te = train_test_split(rem, test_size=.5, stratify=rem.Load_Category, random_state=491)
    split = pd.Series("excluded_duplicate", index=m.index)
    seen = set()
    for name, groups in [("train", tr), ("validation", va), ("test", te)]:
        ix = m.index[m.Representative_Load_Index.isin(groups.Representative_Load_Index)]
        for j in ix:
            h = m.at[j, "measurement_hash"]
            if h not in seen:
                split.at[j] = name
                seen.add(h)
    result = m[["SampleID", "Representative_Load_Index", "Scenario_ID", "Fault_Type", "Fault_Bus", "measurement_hash"]].copy()
    result["split"] = split
    result.to_csv(ROOT / "config/split_manifest.csv", index=False)
    info = {"seed": 490, "grouping": "Representative_Load_Index across every DER scenario", "deduplication": "Exact complete measurement hash, one occurrence retained; priority train then validation then test; redundant normal cases excluded. No faults removed by this rule in this release.", "groups": {name: sorted(g.Representative_Load_Index.tolist()) for name,g in [("train",tr),("validation",va),("test",te)]}, "rows": result.groupby("split").size().to_dict(), "composition": result.groupby(["split","Fault_Type"]).size().unstack(fill_value=0).to_dict(orient="index")}
    (ROOT / "config/split.json").write_text(json.dumps(info,indent=2))
    assert not (set(tr.Representative_Load_Index)&set(te.Representative_Load_Index))
    assert result[result.split.ne("excluded_duplicate")].measurement_hash.is_unique
    assert all(result[result.split.eq(s)].Fault_Type.eq("Normal").any() for s in ["train","validation","test"])
    return split


def main():
    a,m,cols,topo = load_cache()
    split_manifest(m)
    f = PhasorFeatures(cols,topo)
    x = None
    aux = {}
    for start in range(0,len(m),500):
        z,u=f.transform(a[start:start+500],return_aux=True)
        if x is None:
            x=np.lib.format.open_memmap(ROOT / "results/cache/features.npy",mode="w+",dtype="float32",shape=(len(m),z.shape[1]))
        x[start:start+len(z)] = z
        for k,v in u.items():
            if k not in aux:
                aux[k]=np.lib.format.open_memmap(ROOT / f"results/cache/aux_{k}.npy",mode="w+",dtype=v.dtype,shape=(len(m),*v.shape[1:]))
            aux[k][start:start+len(z)] = v
    x.flush()
    for v in aux.values(): v.flush()
    (ROOT / "config/features.json").write_text(json.dumps({"feature_names":f.feature_names_, "raw_columns":cols, "angle_units":"degrees", "voltage_units":"V", "current_units":"A", "current_phase_mapping":"OpenDSS conductor NodeOrder, not PhN column ordinal", "metadata_features":[], "ratio_floors":{"voltage":1.0,"current":1.0}},indent=2))
    print("Features",x.shape,flush=True)
    print((ROOT / "config/split.json").read_text(),flush=True)


if __name__ == "__main__": main()
