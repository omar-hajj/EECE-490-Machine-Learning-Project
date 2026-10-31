"""Phase 5: validation-selected LightGBM; never consumes final test measurements."""
import argparse
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, log_loss
from features import ROOT, PhasorFeatures, load_cache


def probabilities(bundle,x):
    p=bundle["model"].predict_proba(np.asarray(x[:,bundle["indices"]]))
    if p.shape[1]>2:
        p=softmax(np.log(np.maximum(p,1e-12))/bundle.get("temperature",1.),axis=1)
    return p


def fit_task(x,m,split,task,names,tag="full",quick=False):
    tr=split.eq("train").to_numpy(copy=True); va=split.eq("validation").to_numpy(copy=True)
    if task=="detection": y=m.Fault_Label.gt(0).astype(int).to_numpy()
    elif task=="classification":
        y=m.Fault_Label.to_numpy(); tr &= y>0; va &= y>0
    else:
        y=m.Fault_Bus.to_numpy(); fault=m.Fault_Label.gt(0).to_numpy(); tr &= fault; va &= fault
    xt=np.asarray(x[tr]); xv=np.asarray(x[va]); yt=y[tr]; yv=y[va]
    # Selection is learned exclusively from training, inside every sensor/scenario experiment.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        warnings.simplefilter("ignore",RuntimeWarning)
        scores,_=f_classif(xt,yt)
    scores=np.nan_to_num(scores,nan=-np.inf,posinf=1e30)
    configs=[dict(k=256,num_leaves=7,n_estimators=100,min_child_samples=10),dict(k=512,num_leaves=15,n_estimators=160,min_child_samples=5)]
    if quick: configs=[dict(k=256,num_leaves=7,n_estimators=100,min_child_samples=10)]
    best=(-np.inf,None); history=[]
    for config in configs:
        k=min(config["k"],x.shape[1]); indices=np.argsort(-scores,kind="stable")[:k]
        args={k:v for k,v in config.items() if k!="k"}
        model=lgb.LGBMClassifier(**args,learning_rate=.08,max_bin=63,reg_lambda=1.,n_jobs=4,random_state=490,deterministic=True,force_col_wise=True,verbosity=-1,class_weight="balanced" if task=="detection" else None)
        start=time.perf_counter()
        model.fit(xt[:,indices],yt,eval_set=[(xv[:,indices],yv)],callbacks=[lgb.early_stopping(20,verbose=False)])
        p=model.predict_proba(xv[:,indices]); pred=model.classes_[p.argmax(axis=1)]
        if task=="detection": score=balanced_accuracy_score(yv,pred)
        elif task=="classification": score=f1_score(yv,pred,average="macro")
        else: score=accuracy_score(yv,pred)
        record={"task":task,"tag":tag,**config,"best_iteration":model.best_iteration_,"validation_selection_score":float(score),"training_seconds":time.perf_counter()-start}
        history.append(record); print(json.dumps(record),flush=True)
        if score>best[0]:
            best=(score,{"model":model,"indices":indices,"temperature":1.,"task":task,"tag":tag,"config":record,"feature_names":[names[i] for i in indices]})
    bundle=best[1]
    if task!="detection":
        p=bundle["model"].predict_proba(xv[:,bundle["indices"]])
        def loss(temp): return log_loss(yv,softmax(np.log(np.maximum(p,1e-12))/temp,axis=1),labels=bundle["model"].classes_)
        opt=minimize_scalar(loss,bounds=(.5,5),method="bounded")
        bundle["temperature"]=float(opt.x)
    else:
        p=probabilities(bundle,xv)[:,1]
        thresholds=[.5,.05,.1,.2,.3,.7,.8,.9,.95]
        bundle["threshold"]=max(thresholds,key=lambda t:balanced_accuracy_score(yv,p>=t))
    joblib.dump(bundle,ROOT / f"artifacts/{tag}_{task}.joblib",compress=3)
    pd.DataFrame(history).to_csv(ROOT / f"results/{tag}_{task}_tuning.csv",index=False)
    gain=bundle["model"].booster_.feature_importance(importance_type="gain")
    pd.DataFrame({"feature":bundle["feature_names"],"gain":gain}).sort_values("gain",ascending=False).to_csv(ROOT / f"results/{tag}_{task}_importance.csv",index=False)
    return bundle


def subset_cache(a,f,tag):
    path=ROOT / f"results/cache/{tag}_features.npy"
    x=None
    for start in range(0,len(a),500):
        z=f.transform(a[start:start+500])
        if x is None: x=np.lib.format.open_memmap(path,mode="w+",dtype="float32",shape=(len(a),z.shape[1]))
        x[start:start+len(z)]=z
    x.flush()
    return x


def main():
    if (ROOT / "results/final_freeze.json").exists():
        raise RuntimeError("Final test already opened. Use a new, explicitly documented experiment directory for further tuning.")
    parser=argparse.ArgumentParser(); parser.add_argument("--experiment",choices=["full","sensors","physical","unseen"],default="full"); args=parser.parse_args()
    assert (ROOT / "artifacts/b1_network.joblib").exists(),"Finish conventional baselines first"
    a,m,cols,topo=load_cache()
    split=pd.read_csv(ROOT / "config/split_manifest.csv").split
    if args.experiment=="full":
        x=np.load(ROOT / "results/cache/features.npy",mmap_mode="r")
        names=json.loads((ROOT / "config/features.json").read_text())["feature_names"]
        settings={}
        for task in ["detection","classification","localization"]:
            b=fit_task(x,m,split,task,names)
            settings[task]={"config":b["config"],"temperature":b["temperature"],"threshold":b.get("threshold"),"classes":b["model"].classes_.tolist()}
        (ROOT / "artifacts/ml_settings.json").write_text(json.dumps(settings,indent=2))
    elif args.experiment in ["sensors","physical"]:
        placement=pd.read_csv(ROOT / "data/raw/der_fault/pmu_placement_configurations.csv")
        available={v.split(".")[0] for v in PhasorFeatures(cols,topo).v}
        audit=[]
        for col in (["PMU_93"] if args.experiment=="physical" else ["PMU_56","PMU_36","PMU_20"]):
            sensors=sorted(set(placement[col].dropna())&available)
            tag=f"available_{len(sensors)}"
            f=PhasorFeatures(cols,topo,sensors)
            x=subset_cache(a,f,tag)
            audit.append({"published_list":col,"actual_bus_count":len(sensors),"voltage_channels":len(f.v),"current_channels":len(f.i),"sensor_buses":sensors,"feature_count":x.shape[1]})
            for task in ["detection","classification","localization"]: fit_task(x,m,split,task,f.feature_names_,tag)
        (ROOT / f"config/{args.experiment}_experiments.json").write_text(json.dumps(audit,indent=2))
    else:
        # A predeclared shift: S6 is absent from train/validation; evaluate it only
        # at held-out loads. Comparators will use the same S1-S5 development data.
        x=np.load(ROOT / "results/cache/features.npy",mmap_mode="r")
        names=json.loads((ROOT / "config/features.json").read_text())["feature_names"]
        unseen=split.copy(); unseen[(m.Scenario_ID=="S6") & unseen.isin(["train","validation"]) ]="excluded_scenario"
        for task in ["detection","classification","localization"]: fit_task(x,m,unseen,task,names,"unseen_s6")
    print("Experiment selection complete:",args.experiment,flush=True)


if __name__ == "__main__": main()
