"""Phase 2 EDA: aggregate inventory plus electrical analysis on development rows only."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from features import ROOT, load_cache, FAULT_TYPES


def main():
    a,m,c,t = load_cache()
    split = pd.read_csv(ROOT / "config/split_manifest.csv").split
    dev = split.isin(["train","validation"]).to_numpy()
    d = m[dev].copy()
    out=ROOT / "reports/figures"
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.titleweight":"bold","savefig.bbox":"tight"})
    color="#087f8c"
    def save(fig,name): fig.savefig(out / f"{name}.png"); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    counts=m.Fault_Type.value_counts().reindex(FAULT_TYPES)
    axs[0].bar(counts.index,counts.values,color=color); axs[0].tick_params(axis="x",rotation=65); axs[0].set_title("Release class distribution"); axs[0].set_ylabel("Simulation rows")
    axs[1].bar(["Normal","Fault"],[m.Fault_Label.eq(0).sum(),m.Fault_Label.gt(0).sum()],color=["#8798a5",color]); axs[1].set_title("Binary detection balance")
    sc=m.Scenario_ID.value_counts().sort_index(); axs[2].bar(sc.index,sc.values,color=color); axs[2].set_title("DER scenario distribution")
    fig.tight_layout(); save(fig,"01_case_distribution")
    dv=np.load(ROOT / "results/cache/aux_voltage_disturbance.npy")[dev]
    di=np.load(ROOT / "results/cache/aux_current_disturbance.npy")[dev]
    phase=np.load(ROOT / "results/cache/aux_phase_dv.npy")[dev]
    d["max_voltage_phasor_change"]=dv; d["max_current_phasor_change"]=di
    d["resistance_bin"]=pd.cut(d.Resistance,[0,1,5,10,15,25],include_lowest=True)
    fault=d.Fault_Label.gt(0).to_numpy()
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    vix=[i for i,x in enumerate(c) if x.endswith("_Vmag_post")]
    iix=[i for i,x in enumerate(c) if x.endswith("_Imag_post")]
    sample=np.flatnonzero(dev)[::10]
    axs[0].hist(np.array(a[np.ix_(sample,vix)]).ravel(),bins=60,color=color); axs[0].set(xlabel="Post-event phase voltage (V)",ylabel="Channel observations",title="Measured voltage distribution")
    vals=np.array(a[np.ix_(sample,iix)]).ravel(); axs[1].hist(np.log10(np.maximum(vals,1e-6)),bins=65,color="#d47437"); axs[1].set(xlabel="log10 current magnitude (A; floor 1 µA)",ylabel="Channel observations",title="Current spans unloaded and faulted lines")
    fig.suptitle("Training + validation measurements; deterministic 1-in-10 row sample",fontsize=10); fig.tight_layout(); save(fig,"02_voltage_current")
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    axs[0].hist(d.loc[fault,"Resistance"],bins=25,color=color); axs[0].set(xlabel="Fault resistance (ohm)",ylabel="Fault cases",title="Fault resistance sampling")
    groups=[g.max_voltage_phasor_change.to_numpy()*100 for _,g in d[fault].groupby("resistance_bin",observed=True)]
    labels=[str(k) for k,_ in d[fault].groupby("resistance_bin",observed=True)]
    axs[1].boxplot(groups,tick_labels=labels,showfliers=False); axs[1].set(xlabel="Fault resistance interval (ohm)",ylabel="Maximum |ΔV| / |Vpre| (%)",title="Higher resistance and disturbance severity"); axs[1].tick_params(axis="x",rotation=20)
    fig.tight_layout(); save(fig,"03_resistance")
    med=np.vstack([np.median(phase[d.Fault_Label.eq(k)],axis=0) for k in range(1,11)])*100
    fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(med,cmap="YlOrRd",aspect="auto"); ax.set_xticks(range(3),["Phase A","Phase B","Phase C"]); ax.set_yticks(range(10),FAULT_TYPES[1:]); ax.set_title("Median maximum phase voltage-phasor disturbance (%)")
    for i in range(10):
        for j in range(3): ax.text(j,i,f"{med[i,j]:.1f}",ha="center",va="center",color="black")
    fig.colorbar(im,ax=ax); fig.tight_layout(); save(fig,"04_phase_signatures")
    fig,ax=plt.subplots(figsize=(9,4)); g=[v.max_voltage_phasor_change.to_numpy()*100 for _,v in d[fault].groupby("Scenario_ID")]; ax.boxplot(g,tick_labels=sorted(d.Scenario_ID.unique()),showfliers=False); ax.set(xlabel="DER configuration (PV / EV / V2G vary jointly)",ylabel="Maximum |ΔV| / |Vpre| (%)",title="Voltage disturbance across DER scenarios"); fig.tight_layout(); save(fig,"05_der_signatures")
    table=d[fault].groupby("resistance_bin",observed=True).agg(n=("SampleID","size"),median_voltage_change=("max_voltage_phasor_change","median"),median_current_change=("max_current_phasor_change","median"))
    table.to_csv(ROOT / "results/eda_resistance.csv")
    d[fault].groupby(["Scenario_ID","Fault_Type"]).max_voltage_phasor_change.median().to_csv(ROOT / "results/eda_scenario_type.csv")
    rho,p=spearmanr(d.loc[fault,"Resistance"],dv[fault])
    text=f"""# Phase 2 — electrical exploratory analysis

Figures use training/validation data for electrical relationships; the release distribution chart is a Phase 1 inventory. Final test predictions have not been generated. Every figure is in `reports/figures`.

1. **01_case_distribution:** all 11 classes have 1,800 rows; the binary task has ten times as many faults as normal rows. This is a designed dataset, not an estimate of operational fault prevalence. Deduplication further changes prevalence, so precision/accuracy cannot be transferred to a utility's operating environment.
2. **02_voltage_current:** phase voltage and line-current histograms reveal strongly different current scales, including unloaded branches. A raw current threshold requires equipment ratings that are absent. Relative pre/post current changes need a denominator floor. Voltage channels here are primary PMU voltages; they do not imply that secondary faults are directly measured.
3. **03_resistance:** resistance-bin medians are {table.median_voltage_change.round(4).to_dict()}. The development-set Spearman correlation between resistance and maximum voltage-phasor change is {rho:.4f}. This is an association across varying locations/types, not a paired causal resistance sweep. Weak signatures deserve stratified evaluation.
4. **04_phase_signatures:** phase-selective voltage changes support electrical phase rules. Two-phase grounded and ungrounded faults may share affected phases, so phase magnitudes alone cannot distinguish them; zero-sequence/incremental current information is needed. Values aggregate each phase's maximum across instruments, so phases can attain maxima at different buses.
5. **05_der_signatures:** compare entire voltage-disturbance distributions. Scenarios mix PV, EV and V2G, and their fault draws differ. They are not a controlled PV-only experiment. Source model/generation uncertainty and repeated electrical states prevent broad causal conclusions.

No test-set EDA was used to select features, thresholds or hyperparameters. The original class/scenario inventory is not model-selection evidence.
"""
    (ROOT / "reports/EDA_DER.md").write_text(text,encoding="utf-8")
    print(text,flush=True)


if __name__ == "__main__": main()
