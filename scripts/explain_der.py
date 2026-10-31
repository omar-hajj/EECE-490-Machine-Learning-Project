"""Native LightGBM TreeSHAP contributions and paired test error analysis; no refitting."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import ROOT


def main():
    assert (ROOT / "results/final_freeze.json").exists()
    m=pd.read_csv(ROOT / "results/cache/metadata.csv")
    split=pd.read_csv(ROOT / "config/split_manifest.csv").split
    x=np.load(ROOT / "results/cache/features.npy",mmap_mode="r")
    rng=np.random.default_rng(490)
    notes=[]
    for task in ["detection","classification","localization"]:
        b=joblib.load(ROOT / f"artifacts/full_{task}.joblib")
        ids=np.flatnonzero(split.eq("test") & (m.Fault_Label.gt(0) if task!="detection" else True))
        if task=="detection":
            normal=ids[m.iloc[ids].Fault_Label.eq(0).to_numpy()]
            fault=ids[m.iloc[ids].Fault_Label.gt(0).to_numpy()]
            ids=np.concatenate([normal,rng.choice(fault,size=100-len(normal),replace=False)])
        else: ids=rng.choice(ids,size=min(100,len(ids)),replace=False)
        z=np.asarray(x[np.ix_(ids,b["indices"])])
        contrib=b["model"].booster_.predict(z,pred_contrib=True,num_threads=4)
        raw=b["model"].booster_.predict(z,raw_score=True,num_threads=4)
        nfeatures=len(b["indices"])
        if np.asarray(contrib).ndim==2 and contrib.shape[1] != nfeatures+1:
            contrib=contrib.reshape(len(ids),len(b["model"].classes_),nfeatures+1)
            chosen=np.argmax(raw,axis=1)
            error=float(np.max(np.abs(contrib.sum(axis=2)-raw)))
            selected=contrib[np.arange(len(ids)),chosen,:-1]
        else:
            error=float(np.max(np.abs(contrib.sum(axis=1)-raw)))
            selected=contrib[:,:-1]
        importance=np.abs(selected).mean(axis=0)
        frame=pd.DataFrame({"feature":b["feature_names"],"mean_abs_TreeSHAP_raw_margin":importance}).sort_values("mean_abs_TreeSHAP_raw_margin",ascending=False)
        frame.to_csv(ROOT / f"results/{task}_treeshap.csv",index=False)
        fig,ax=plt.subplots(figsize=(10,6)); top=frame.head(12).iloc[::-1]; ax.barh(top.feature,top.mean_abs_TreeSHAP_raw_margin,color="#087f8c"); ax.set(xlabel="Mean absolute contribution to predicted class raw margin",title=f"{task.title()} — native TreeSHAP, 100 held-out cases"); fig.tight_layout(); fig.savefig(ROOT / f"reports/figures/{task}_treeshap.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        notes.append({"task":task,"sample_size":len(ids),"additivity_max_abs_error":error,"sample_ids":m.iloc[ids].SampleID.tolist(),"scope":"Raw-margin tree contributions before temperature scaling. Multiclass importance averages predicted-class contributions, not every class. Fixed deterministic test sample; no retraining."})
        print(task,"TreeSHAP additivity error",error,flush=True)
    (ROOT / "results/explainability_audit.json").write_text(json.dumps(notes,indent=2))
    d=pd.read_csv(ROOT / "results/full_test_predictions.csv")
    faults=d[d.Fault_Label.gt(0)].copy()
    records=[]
    for task in ["classification","localization"]:
        if task=="classification": a=faults.B1_type.eq(faults.Fault_Label); b=faults.LightGBM_type.eq(faults.Fault_Label)
        else: a=faults.B1_top1_correct; b=faults.LightGBM_top1_correct
        for category,mask in [("B1_only_correct",a&~b),("ML_only_correct",b&~a),("both_wrong",~a&~b),("both_correct",a&b)]:
            g=faults[mask]; records.append({"task":task,"category":category,"n":len(g),"median_resistance_ohm":g.Resistance.median() if len(g) else None,"example_sample_ids":g.SampleID.head(10).tolist()})
            g.head(20).to_csv(ROOT / f"results/{task}_{category}_examples.csv",index=False)
    (ROOT / "results/error_summary.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__": main()
