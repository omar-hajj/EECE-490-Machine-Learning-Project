"""B0 and B1 electrical rules plus network current-injection residual localization.

No learned bus templates or target-dependent distances are used by B1 localization.
The network dictionary comes only from the independently sourced OpenDSS model.
"""
import itertools
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from threadpoolctl import threadpool_limits
from features import ROOT, FAULT_TYPES, PhasorFeatures, load_cache

PHASES={1:[1],2:[2],3:[3],4:[1,2],5:[2,3],6:[1,3],7:[1,2],8:[2,3],9:[1,3],10:[1,2,3]}


def phase_rule(aux, single=.4, triple=.65, ground=.3, current_weight=0, ground_source="current"):
    v=aux["phase_dv"]
    i=aux["phase_di"]
    s=v/np.maximum(v.max(axis=1,keepdims=True),1e-9)+current_weight*i/np.maximum(i.max(axis=1,keepdims=True),1e-9)
    order=np.argsort(s,axis=1)[:,::-1]
    sorted_s=np.take_along_axis(s,order,axis=1)
    second=sorted_s[:,1]/np.maximum(sorted_s[:,0],1e-9)
    third=sorted_s[:,2]/np.maximum(sorted_s[:,0],1e-9)
    # LL currents sum to ~0; ground faults produce residual current. Incremental
    # components suppress normal load unbalance; a max over valid triplets keeps
    # faults downstream of different feeders visible.
    vg=aux["v0"]/np.maximum(aux["voltage_disturbance"],1e-9)
    ground_value={"current":aux["i0_fraction"], "voltage":vg, "combined":np.maximum(vg,aux["i0_fraction"])}[ground_source]
    grounded=ground_value>ground
    result=np.full(len(v),10,dtype=int)
    one=second<single
    result[one]=order[one,0]+1
    two=(~one)&(third<triple)
    pair=np.sort(order[:,:2],axis=1)
    for phases,label in [([0,1],4),([1,2],5),([0,2],6)]:
        mask=two & np.all(pair==phases,axis=1)
        result[mask]=label+3*grounded[mask]
    return result


def load_aux():
    return {p.stem[4:]:np.load(p,mmap_mode="r") for p in (ROOT / "results/cache").glob("aux_*.npy")}


def tune_rules(aux,m,split,tag="full"):
    tr=split.eq("train").to_numpy(); va=split.eq("validation").to_numpy()
    truth=m.Fault_Label.to_numpy()
    normal=tr & (truth==0)
    b0_current=float(2*np.quantile(aux["max_current"][normal],.99))
    records=[]
    best=(-1,None)
    faultva=va&(truth>0)
    for single,triple,ground,cw,gs in itertools.product([.2,.35,.5,.65],[.35,.5,.65,.8],[.05,.15,.3,.5,.65],[0,.25,.5,1],["current","voltage","combined"]):
        pred=phase_rule(aux,single,triple,ground,cw,gs)
        score=f1_score(truth[faultva],pred[faultva],average="macro")
        if score>best[0]:
            best=(score,dict(single=single,triple=triple,ground=ground,current_weight=cw,ground_source=gs))
    best_det=(-1,None)
    for weight,threshold in itertools.product([0,.01,.1,1],[1e-6,1e-4,.001,.003,.01,.03,.05,.1]):
        s=np.maximum(aux["voltage_disturbance"],weight*aux["current_disturbance"])
        p=s>=threshold
        score=balanced_accuracy_score(truth[va]>0,p[va])
        records.append(dict(current_weight=weight,threshold=threshold,validation_balanced_accuracy=score))
        if score>best_det[0]: best_det=(score,dict(current_weight=weight,threshold=threshold))
    settings={"B0":{"voltage_sag":.1,"absolute_current_A":b0_current,"rationale":"10% voltage sag OR twice training-normal 99th-percentile maximum current. Not a thermal/CT equipment setting: ratings absent."},"B1_detection":best_det[1],"B1_classification":best[1],"validation_macro_f1":best[0],"selection":"Rules tuned on validation only; B0 nominal current scale uses training normals. No test predictions."}
    prefix="" if tag=="full" else f"{tag}_"
    (ROOT / f"artifacts/{prefix}baseline_settings.json").write_text(json.dumps(settings,indent=2))
    pd.DataFrame(records).to_csv(ROOT / f"results/{prefix}baseline_detection_tuning.csv",index=False)
    return settings


def rule_predictions(aux,settings):
    b0=(aux["max_sag"]>=settings["B0"]["voltage_sag"])|(aux["max_current"]>=settings["B0"]["absolute_current_A"])
    s=settings["B1_detection"]
    score=np.maximum(aux["voltage_disturbance"],s["current_weight"]*aux["current_disturbance"])
    b1=score>=s["threshold"]
    typ=phase_rule(aux,**settings["B1_classification"])
    return b0,b1,typ,score


def network_dictionary(f,topology,train_aux,weight):
    zfile=np.load(ROOT / "artifacts/network_transfer.npz")
    z=zfile["z"]
    nodes=list(zfile["nodes"])
    ix={n:i for i,n in enumerate(nodes)}
    vs=np.maximum(np.median(np.abs(train_aux["vp"]),axis=0),1.)
    isc=np.maximum(np.median(np.abs(train_aux["ip"]),axis=0),5.)
    rows=[z[[ix[v.lower()] for v in f.v],:]/vs[:,None]]
    current_rows=[]
    for base in f.i:
        el,terminal,conductor=re_current(base)
        info=topology["line_map"][el]
        y=np.asarray(info["y_real"])+1j*np.asarray(info["y_imag"])
        k=(terminal-1)*info["conductors"]+conductor-1
        local=np.zeros((len(info["indices"]),z.shape[1]),dtype=complex)
        for j,node in enumerate(info["indices"]):
            if node>=0: local[j]=z[node]
        current_rows.append(y[k]@local)
    rows.append(weight*np.asarray(current_rows)/isc[:,None])
    transfer=np.vstack(rows)
    # Every fault label in this release belongs to a source-model 3-phase bus.
    # Derive candidate universe physically, without inspecting held-out bus labels.
    buses=sorted(b for b in topology["buses"] if all(f"{b}.{p}" in ix for p in [1,2,3]))
    banks={}
    for typ,phases in PHASES.items():
        qblocks=[]; offsets=[0]
        for bus in buses:
            h=transfer[:,[ix[f"{bus}.{p}"] for p in phases]]
            if typ in [4,5,6]: h=(h[:,0]-h[:,1])[:,None]
            u,s,_=np.linalg.svd(h,full_matrices=False)
            keep=s>max(s[0]*1e-8,1e-12)
            q=u[:,keep]
            qblocks.append(q); offsets.append(offsets[-1]+q.shape[1])
        banks[typ]={"q":np.concatenate(qblocks,axis=1),"offsets":np.array(offsets)}
    return {"banks":banks,"buses":buses,"voltage_scale":vs,"current_scale":isc,"current_weight":weight}


def re_current(base):
    import re
    m=re.fullmatch(r"(.+)_T(\d+)_Ph(\d+)",base)
    return m[1].lower(),int(m[2]),int(m[3])


def locate(aux,types,dictionary):
    y=np.concatenate([(aux["vq"]-aux["vp"])/dictionary["voltage_scale"],dictionary["current_weight"]*(aux["iq"]-aux["ip"])/dictionary["current_scale"]],axis=1)
    ranking=np.empty((len(y),5),dtype=object)
    residual=np.empty(len(y))
    buses=np.array(dictionary["buses"])
    with threadpool_limits(limits=4):
        for typ in PHASES:
            ids=np.flatnonzero(types==typ)
            q=dictionary["banks"][typ]["q"]
            offsets=dictionary["banks"][typ]["offsets"]
            for start in range(0,len(ids),500):
                k=ids[start:start+500]
                projected=np.abs(y[k]@q.conj())**2
                power=np.column_stack([projected[:,offsets[j]:offsets[j+1]].sum(axis=1) for j in range(len(buses))])
                rank=np.argsort(-power,axis=1,kind="stable")[:,:5]
                ranking[k]=buses[rank]
                norm=(np.abs(y[k])**2).sum(axis=1)
                residual[k]=np.maximum(0,1-power[np.arange(len(k)),rank[:,0]]/np.maximum(norm,1e-20))
    return ranking,residual


def main():
    if (ROOT / "results/final_freeze.json").exists():
        raise RuntimeError("Final test already opened; conventional retuning is prohibited in this experiment.")
    import joblib
    a,m,cols,topology=load_cache()
    split=pd.read_csv(ROOT / "config/split_manifest.csv").split
    aux=load_aux()
    settings=tune_rules(aux,m,split)
    b0,b1,typ,score=rule_predictions(aux,settings)
    tr=np.flatnonzero(split.eq("train")); va=np.flatnonzero(split.eq("validation") & m.Fault_Label.gt(0))
    f=PhasorFeatures(cols,topology)
    best=(-1,None,None)
    tuning=[]
    for weight in [0,.1,1.]:
        bank=network_dictionary(f,topology,{k:v[tr] for k,v in aux.items()},weight)
        rank,res=locate({k:v[va] for k,v in aux.items()},typ[va],bank)
        acc=accuracy_score(m.iloc[va].Fault_Bus,rank[:,0])
        tuning.append(dict(current_weight=weight,validation_exact_bus_accuracy=acc,median_residual=float(np.median(res))))
        print("B1 location validation",tuning[-1],flush=True)
        if acc>best[0]: best=(acc,bank,rank)
    joblib.dump(best[1],ROOT / "artifacts/b1_network.joblib",compress=3)
    settings["B1_localization"]={"current_weight":best[1]["current_weight"],"validation_exact_bus_accuracy":best[0],"candidates":len(best[1]["buses"]),"method":"least-squares sparse current-injection transfer residual, predicted fault phase constraints; source circuit approximation"}
    (ROOT / "artifacts/baseline_settings.json").write_text(json.dumps(settings,indent=2))
    pd.DataFrame(tuning).to_csv(ROOT / "results/baseline_localization_tuning.csv",index=False)
    pd.DataFrame({"SampleID":m.loc[va,"SampleID"],"actual_bus":m.loc[va,"Fault_Bus"],"predicted_bus":best[2][:,0],"actual_type":m.loc[va,"Fault_Type"],"predicted_type":np.array(FAULT_TYPES)[typ[va]]}).to_csv(ROOT / "results/baseline_validation_predictions.csv",index=False)
    print(json.dumps(settings,indent=2),flush=True)


if __name__ == "__main__": main()
