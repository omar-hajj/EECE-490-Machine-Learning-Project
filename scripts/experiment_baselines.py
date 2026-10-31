"""Fit B1 independently under each reduced sensor and unseen-S6 protocol."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from features import ROOT, PhasorFeatures, load_cache
from conventional import tune_rules, rule_predictions, network_dictionary, locate, load_aux


def subset_aux(a,f):
    aux={}
    for start in range(0,len(a),500):
        _,u=f.transform(a[start:start+500],return_aux=True)
        for k,v in u.items():
            if k not in aux: aux[k]=np.empty((len(a),*v.shape[1:]),dtype=v.dtype)
            aux[k][start:start+len(v)]=v
    return aux


def main():
    if (ROOT / "results/final_freeze.json").exists():
        raise RuntimeError("Final test already opened; baseline retuning requires a new experiment.")
    p=argparse.ArgumentParser(); p.add_argument("--experiment",choices=["sensors","physical","unseen"],required=True); args=p.parse_args()
    a,m,c,t=load_cache(); split=pd.read_csv(ROOT / "config/split_manifest.csv").split
    variants=[]
    if args.experiment in ["sensors","physical"]:
        placement=pd.read_csv(ROOT / "data/raw/der_fault/pmu_placement_configurations.csv")
        available={v.split(".")[0] for v in PhasorFeatures(c,t).v}
        for col in (["PMU_93"] if args.experiment=="physical" else ["PMU_56","PMU_36","PMU_20"]):
            sensors=sorted(set(placement[col].dropna())&available)
            variants.append((f"available_{len(sensors)}",sensors,split))
    else:
        s=split.copy(); s[(m.Scenario_ID=="S6") & s.isin(["train","validation"]) ]="excluded_scenario"
        variants=[("unseen_s6",None,s)]
    for tag,sensors,s in variants:
        f=PhasorFeatures(c,t,sensors)
        aux=load_aux() if sensors is None else subset_aux(a,f)
        settings=tune_rules(aux,m,s,tag)
        _,_,typ,_=rule_predictions(aux,settings)
        tr=np.flatnonzero(s.eq("train")); va=np.flatnonzero(s.eq("validation") & m.Fault_Label.gt(0))
        best=(-1,None); tuning=[]
        for w in [0,.1,1]:
            bank=network_dictionary(f,t,{k:v[tr] for k,v in aux.items()},w)
            rank,res=locate({k:v[va] for k,v in aux.items()},typ[va],bank)
            score=accuracy_score(m.iloc[va].Fault_Bus,rank[:,0]); tuning.append({"weight":w,"validation_exact_bus_accuracy":score})
            if score>best[0]: best=(score,bank)
        settings["B1_localization"]={"current_weight":best[1]["current_weight"],"validation_exact_bus_accuracy":best[0],"candidates":len(best[1]["buses"])}
        (ROOT / f"artifacts/{tag}_baseline_settings.json").write_text(json.dumps(settings,indent=2))
        joblib.dump(best[1],ROOT / f"artifacts/{tag}_b1_network.joblib",compress=3)
        pd.DataFrame(tuning).to_csv(ROOT / f"results/{tag}_baseline_localization_tuning.csv",index=False)
        print(tag,json.dumps(settings),flush=True)
    print("Baseline variants complete",args.experiment,flush=True)


if __name__=="__main__": main()
