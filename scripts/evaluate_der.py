"""Freeze all selected artifacts, then generate held-out predictions and honest comparisons."""
import json
import hashlib
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss
from features import ROOT, FAULT_TYPES, PhasorFeatures, load_cache
from conventional import load_aux, rule_predictions, locate
from experiment_baselines import subset_aux
from train_der import probabilities


def freeze():
    required=[ROOT / f"artifacts/{tag}_{task}.joblib" for tag in ["full","available_93","available_36","available_21","available_9","unseen_s6"] for task in ["detection","classification","localization"]]
    assert all(p.exists() for p in required),"Finish every planned model selection before opening final test"
    files=list((ROOT / "artifacts").glob("*.joblib"))+list((ROOT / "artifacts").glob("*.json"))+list((ROOT / "config").glob("*.json"))+[ROOT / "config/split_manifest.csv",ROOT / "artifacts/network_transfer.npz",ROOT / "requirements-lock.txt"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=ROOT / "results/final_freeze.json"
    if path.exists():
        old=json.loads(path.read_text())
        assert old["sha256"]==hashes,"Model/selection artifacts changed after test freeze"
    else:
        path.write_text(json.dumps({"protocol":"All selection finished before final predictions; no final-test-based tuning permitted.","sha256":hashes},indent=2))


def graph_distances(topo):
    buses=topo["buses"]; ix={b:i for i,b in enumerate(buses)}
    graph=np.zeros((len(buses),len(buses)))
    for e in topo["edges"]: graph[ix[e["u"]],ix[e["v"]]]=graph[ix[e["v"]],ix[e["u"]]]=1
    return shortest_path(csr_matrix(graph),directed=False,unweighted=True),ix


def prediction_frame(m,aux,x,tag):
    prefix="" if tag=="full" else f"{tag}_"
    settings=json.loads((ROOT / f"artifacts/{prefix}baseline_settings.json").read_text())
    net=joblib.load(ROOT / f"artifacts/{prefix}b1_network.joblib")
    start=time.perf_counter(); b0,b1,typ,score=rule_predictions(aux,settings)
    rank,res=locate(aux,typ,net)
    b1seconds=time.perf_counter()-start
    bundles={task:joblib.load(ROOT / f"artifacts/{tag}_{task}.joblib") for task in ["detection","classification","localization"]}
    start=time.perf_counter()
    ps={task:probabilities(b,x) for task,b in bundles.items()}
    mlseconds=time.perf_counter()-start
    d=m.copy()
    d["B0_detection"]=b0.astype(int); d["B1_detection"]=b1.astype(int); d["B1_type"]=typ; d["B1_location"]=rank[:,0]; d["B1_residual"]=res
    d["LightGBM_score"]=ps["detection"][:,1]
    d["LightGBM_detection"]=(ps["detection"][:,1]>=bundles["detection"].get("threshold",.5)).astype(int)
    d["LightGBM_type"]=bundles["classification"]["model"].classes_[ps["classification"].argmax(axis=1)]
    order=np.argsort(-ps["localization"],axis=1)[:,:5]
    mlrank=bundles["localization"]["model"].classes_[order]
    d["LightGBM_location"]=mlrank[:,0]
    for model,r in [("B1",rank),("LightGBM",mlrank)]:
        for k in [1,3,5]: d[f"{model}_top{k}_correct"]=np.any(r[:,:k]==m.Fault_Bus.to_numpy()[:,None],axis=1)
        for k in range(5): d[f"{model}_rank{k+1}"]=r[:,k]
    d["LightGBM_type_confidence"]=ps["classification"].max(axis=1)
    d["LightGBM_location_confidence"]=ps["localization"].max(axis=1)
    cal=[]
    fault=m.Fault_Label.gt(0).to_numpy()
    for task,col in [("classification","Fault_Label"),("localization","Fault_Bus")]:
        p=ps[task][fault]; y=m.loc[fault,col].to_numpy(); classes=bundles[task]["model"].classes_
        confidence=p.max(axis=1); correct=classes[p.argmax(axis=1)]==y
        ece=0.
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            k=(confidence>lo)&(confidence<=hi)
            if k.any(): ece+=k.mean()*abs(correct[k].mean()-confidence[k].mean())
        cal.append({"task":task,"log_loss":log_loss(y,p,labels=classes),"ECE_10_bins":ece,"temperature":bundles[task]["temperature"]})
    return d,{"rows":len(m),"baseline_batch_seconds":b1seconds,"ml_batch_seconds":mlseconds,"baseline_batch_ms_per_case":1000*b1seconds/len(m),"ml_batch_ms_per_case":1000*mlseconds/len(m),"timing_scope":"prediction from precomputed features/phasors; excludes disk/model loading and feature extraction"},cal


def metric_rows(d,tag,partition,stratum="all",value="all"):
    result=[]
    truth=d.Fault_Label.gt(0).to_numpy(); fault=truth
    def add(model,task,metric,v,n): result.append(dict(experiment=tag,partition=partition,stratum=stratum,value=str(value),model=model,task=task,metric=metric,result=float(v) if v is not None else None,n=int(n)))
    for model in ["B0","B1","LightGBM"]:
        p=d[f"{model}_detection"].to_numpy()
        for metric,fn in [("accuracy",accuracy_score),("precision",precision_score),("recall",recall_score),("f1",f1_score)]:
            if metric=="accuracy": v=fn(truth,p)
            else: v=fn(truth,p,zero_division=0)
            add(model,"detection",metric,v,len(d))
        nnormal=int((~truth).sum()); fp=int(((~truth)&(p>0)).sum())
        add(model,"detection","false_positive_rate",fp/nnormal if nnormal else None,nnormal)
        if nnormal:
            ci=binomtest(fp,nnormal).proportion_ci(confidence_level=.95,method="exact")
            add(model,"detection","FPR_95_lower",ci.low,nnormal); add(model,"detection","FPR_95_upper",ci.high,nnormal)
        add(model,"detection","recall_at_1pct_FPR",None,nnormal)
        if model=="B0":
            for task,metric in [("classification","accuracy"),("classification","macro_f1"),("localization","exact_bus_accuracy"),("localization","top3_accuracy"),("localization","top5_accuracy"),("localization","mean_branch_hops")]: add(model,task,metric,None,fault.sum())
            continue
        if not fault.any(): continue
        actual=d.loc[fault,"Fault_Label"]; pred=d.loc[fault,f"{model}_type"]
        add(model,"classification","accuracy",accuracy_score(actual,pred),fault.sum())
        add(model,"classification","macro_f1",f1_score(actual,pred,labels=list(range(1,11)),average="macro",zero_division=0),fault.sum())
        add(model,"classification","end_to_end_accuracy",np.mean((actual.to_numpy()==pred.to_numpy())&(p[fault]>0)),fault.sum())
        for k in [1,3,5]: add(model,"localization","exact_bus_accuracy" if k==1 else f"top{k}_accuracy",d.loc[fault,f"{model}_top{k}_correct"].mean(),fault.sum())
        add(model,"localization","end_to_end_exact_bus_accuracy",np.mean(d.loc[fault,f"{model}_top1_correct"].to_numpy()&(p[fault]>0)),fault.sum())
        add(model,"localization","mean_branch_hops",d.loc[fault,f"{model}_branch_hops"].mean(),fault.sum())
        add(model,"localization","median_branch_hops",d.loc[fault,f"{model}_branch_hops"].median(),fault.sum())
    return result


def confusion_outputs(d,tag,partition):
    out=ROOT / "results/confusion"; out.mkdir(exist_ok=True)
    figs=ROOT / "reports/figures"
    for model in ["B0","B1","LightGBM"]:
        tasks=[("detection",d.Fault_Label.gt(0),d[f"{model}_detection"],[0,1],["Normal","Fault"])]
        if model!="B0":
            f=d.Fault_Label.gt(0)
            tasks.append(("classification",d.loc[f,"Fault_Label"],d.loc[f,f"{model}_type"],list(range(1,11)),FAULT_TYPES[1:]))
        for task,y,p,labels,names in tasks:
            cm=confusion_matrix(y,p,labels=labels)
            pd.DataFrame(cm,index=names,columns=names).to_csv(out / f"{tag}_{partition}_{model}_{task}.csv")
            pd.DataFrame(classification_report(y,p,labels=labels,target_names=names,output_dict=True,zero_division=0)).T.to_csv(out / f"{tag}_{partition}_{model}_{task}_report.csv")
            if tag=="full":
                fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(len(names)),names,rotation=65,ha="right"); ax.set_yticks(range(len(names)),names); ax.set(xlabel="Predicted",ylabel="Actual",title=f"{model} {task} — {partition}")
                for i in range(len(names)):
                    for j in range(len(names)): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8,color="white" if cm[i,j]>cm.max()/2 else "black")
                fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figs / f"{partition}_{model}_{task}_confusion.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        if model!="B0":
            f=d.Fault_Label.gt(0); buses=sorted(set(d.loc[f,"Fault_Bus"])|set(d.loc[f,f"{model}_location"]))
            pd.DataFrame(confusion_matrix(d.loc[f,"Fault_Bus"],d.loc[f,f"{model}_location"],labels=buses),index=buses,columns=buses).to_csv(out / f"{tag}_{partition}_{model}_localization.csv")


def paired_statistics(d):
    rng=np.random.default_rng(490)
    result=[]
    for task in ["detection","classification","localization"]:
        f=np.ones(len(d),dtype=bool) if task=="detection" else d.Fault_Label.gt(0).to_numpy()
        q=d.loc[f].copy()
        if task=="detection":
            y=q.Fault_Label.gt(0).to_numpy(); a=q.B1_detection.to_numpy()==y; b=q.LightGBM_detection.to_numpy()==y
        elif task=="classification":
            a=q.B1_type.to_numpy()==q.Fault_Label.to_numpy(); b=q.LightGBM_type.to_numpy()==q.Fault_Label.to_numpy()
        else: a=q.B1_top1_correct.to_numpy(); b=q.LightGBM_top1_correct.to_numpy()
        q["delta"]=b.astype(float)-a.astype(float)
        sums=q.groupby("Representative_Load_Index").delta.agg(["sum","count"])
        boot=[]
        for _ in range(2000):
            s=sums.iloc[rng.integers(0,len(sums),len(sums))]; boot.append(s["sum"].sum()/s["count"].sum())
        wins=int((b&~a).sum()); losses=int((a&~b).sum())
        result.append({"task":task,"accuracy_difference_ML_minus_B1":float(np.mean(b)-np.mean(a)),"cluster_bootstrap_95_lower":float(np.quantile(boot,.025)),"cluster_bootstrap_95_upper":float(np.quantile(boot,.975)),"ML_only_correct":wins,"B1_only_correct":losses,"discordant_binomial_p_descriptive":binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.,"clusters":len(sums),"caution":"12 metadata load clusters; shared electrical operating states remain. Binomial p ignores dependence and is descriptive only; use cluster interval as primary, not field-generalization evidence."})
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--validation-only",action="store_true"); args=parser.parse_args()
    if not args.validation_only: freeze()
    a,m,c,t=load_cache(); split=pd.read_csv(ROOT / "config/split_manifest.csv").split
    dist,ix=graph_distances(t)
    placements=pd.read_csv(ROOT / "data/raw/der_fault/pmu_placement_configurations.csv")
    available={v.split(".")[0] for v in PhasorFeatures(c,t).v}
    rows=[]; times=[]; calibration=[]
    variants=[("full",None)] if args.validation_only else [("full",None),("available_93","PMU_93"),("available_36","PMU_56"),("available_21","PMU_36"),("available_9","PMU_20"),("unseen_s6",None)]
    for tag,col in variants:
        for part in (["validation"] if args.validation_only else ["validation","test"]):
            mask=split.eq(part)
            if tag=="unseen_s6": mask &= m.Scenario_ID.eq("S6")
            ids=np.flatnonzero(mask)
            if col is None:
                x=np.load(ROOT / "results/cache/features.npy",mmap_mode="r")[ids]
                aux={k:v[ids] for k,v in load_aux().items()}
            else:
                sensors=sorted(set(placements[col].dropna())&available); f=PhasorFeatures(c,t,sensors)
                x=f.transform(a[ids]); aux=subset_aux(a[ids],f)
            d,timing,cal=prediction_frame(m.iloc[ids].reset_index(drop=True),aux,x,tag)
            d["partition"]=part
            for model in ["B1","LightGBM"]:
                d[f"{model}_branch_hops"]=[dist[ix[actual],ix[pred]] if label>0 else np.nan for actual,pred,label in zip(d.Fault_Bus,d[f"{model}_location"],d.Fault_Label)]
            d.to_csv(ROOT / f"results/{tag}_{part}_predictions.csv",index=False)
            rows.extend(metric_rows(d,tag,part))
            for stratum in ["Scenario_ID","Fault_Type","resistance_bin"]:
                if stratum=="resistance_bin":
                    d[stratum]=pd.cut(d.Resistance,[0,1,5,10,15,25],include_lowest=True).astype(str); strata=d[d.Fault_Label.gt(0)]
                else: strata=d
                for value,g in strata.groupby(stratum): rows.extend(metric_rows(g,tag,part,stratum,value))
            confusion_outputs(d,tag,part)
            times.append({"experiment":tag,"partition":part,**timing})
            calibration.extend([{"experiment":tag,"partition":part,**z} for z in cal])
            if tag=="full" and part=="test":
                (ROOT / "results/paired_statistics.json").write_text(json.dumps(paired_statistics(d),indent=2))
                fault=d.Fault_Label.gt(0)
                errors=d[fault & ((d.B1_type!=d.Fault_Label)|(d.LightGBM_type!=d.Fault_Label)|~d.B1_top1_correct|~d.LightGBM_top1_correct)]
                errors.to_csv(ROOT / "results/error_cases.csv",index=False)
            print("Evaluated",tag,part,len(d),flush=True)
    prefix="validation_smoke_" if args.validation_only else ""
    pd.DataFrame(rows).to_csv(ROOT / f"results/{prefix}metrics.csv",index=False)
    pd.DataFrame(times).to_csv(ROOT / f"results/{prefix}inference_timing.csv",index=False)
    pd.DataFrame(calibration).to_csv(ROOT / f"results/{prefix}calibration.csv",index=False)


if __name__=="__main__": main()
