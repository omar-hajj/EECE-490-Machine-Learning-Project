"""Post-evaluation efficiency measurements; no fitting or model selection.

Reports computing time, not fault-inception-to-decision delay. Uses a balanced
timing sample of ten retained test cases per class, three repeated passes.
"""
import json
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from features import ROOT, PhasorFeatures, load_cache
from conventional import rule_predictions, locate
from train_der import probabilities


def main():
    assert (ROOT / "results/final_freeze.json").exists(), "Benchmark frozen models only"
    a,m,columns,topology=load_cache()
    split=pd.read_csv(ROOT / "config/split_manifest.csv").split
    settings=json.loads((ROOT / "artifacts/baseline_settings.json").read_text())
    network=joblib.load(ROOT / "artifacts/b1_network.joblib")
    models={task:joblib.load(ROOT / f"artifacts/full_{task}.joblib") for task in ["detection","classification","localization"]}
    f=PhasorFeatures(columns,topology)
    rng=np.random.default_rng(490)
    ids=[]
    for label in sorted(m.Fault_Label.unique()):
        candidates=np.flatnonzero(split.eq("test") & m.Fault_Label.eq(label))
        ids.extend(rng.choice(candidates,size=min(10,len(candidates)),replace=False).tolist())
    rng.shuffle(ids)
    rows=[]

    def b0(aux):
        return (aux["max_sag"]>=settings["B0"]["voltage_sag"]) | (aux["max_current"]>=settings["B0"]["absolute_current_A"])

    def b1_detect(aux):
        s=settings["B1_detection"]
        return np.maximum(aux["voltage_disturbance"],s["current_weight"]*aux["current_disturbance"])>=s["threshold"]

    from conventional import phase_rule
    with threadpool_limits(limits=4):
        # Warm model/library paths before recording timings.
        for j in ids[:10]:
            x,aux=f.transform(a[j:j+1],return_aux=True)
            _,_,typ,_=rule_predictions(aux,settings)
            locate(aux,typ,network)
            for bundle in models.values(): probabilities(bundle,x)
        for repeat in range(3):
            order=rng.permutation(ids)
            for j in order:
                start=time.perf_counter_ns()
                x,aux=f.transform(a[j:j+1],return_aux=True)
                feature_ms=(time.perf_counter_ns()-start)/1e6
                rows.append({"repeat":repeat,"SampleID":int(m.at[j,"SampleID"]),"component":"shared_phasor_and_feature_transform","ms":feature_ms})
                typ=phase_rule(aux,**settings["B1_classification"])
                funcs={"B0_detection":lambda:b0(aux),"B1_detection":lambda:b1_detect(aux),"B1_classification":lambda:phase_rule(aux,**settings["B1_classification"]),"B1_localization":lambda:locate(aux,typ,network)}
                for task,bundle in models.items(): funcs[f"LightGBM_{task}"]=lambda b=bundle:probabilities(b,x)
                names=list(funcs); rng.shuffle(names)
                for name in names:
                    start=time.perf_counter_ns(); funcs[name](); elapsed=(time.perf_counter_ns()-start)/1e6
                    rows.append({"repeat":repeat,"SampleID":int(m.at[j,"SampleID"]),"component":name,"ms":elapsed})
    timings=pd.DataFrame(rows)
    timings.to_csv(ROOT / "results/efficiency_timing_samples.csv",index=False)
    summary=timings.groupby("component").ms.agg(n="size",mean_ms="mean",median_ms="median",p95_ms=lambda x:np.quantile(x,.95)).reset_index()
    summary.to_csv(ROOT / "results/efficiency_timing_summary.csv",index=False)
    medians=summary.set_index("component").median_ms
    time_comparison=[]
    for task in ["detection","classification","localization"]:
        base=float(medians[f"B1_{task}"]); ml=float(medians[f"LightGBM_{task}"])
        time_comparison.append({"task":task,"B1_median_ms":base,"LightGBM_median_ms":ml,"ML_time_reduction_percent":100*(1-ml/base)})
    pd.DataFrame(time_comparison).to_csv(ROOT / "results/relative_inference_timing.csv",index=False)
    metrics=pd.read_csv(ROOT / "results/metrics.csv")
    alltest=metrics[(metrics.partition=="test") & (metrics.stratum=="all")]
    def metric(tag,model,task,name):
        return float(alltest[(alltest.experiment==tag)&(alltest.model==model)&(alltest.task==task)&(alltest.metric==name)].result.iloc[0])
    errors=[]
    for task,name in [("classification","accuracy"),("localization","exact_bus_accuracy")]:
        base=metric("full","B1",task,name); ml=metric("full","LightGBM",task,name)
        errors.append({"task":task,"B1_accuracy":base,"LightGBM_accuracy":ml,"accuracy_gain_percentage_points":100*(ml-base),"relative_error_reduction_percent":100*(ml-base)/(1-base)})
    pd.DataFrame(errors).to_csv(ROOT / "results/relative_error_reduction.csv",index=False)
    sensors=[]
    for count in [93,36,21,9]:
        tag=f"available_{count}"
        for model in ["B1","LightGBM"]:
            sensors.append({"PMUs":count,"model":model,"PMU_count_reduction_percent":100*(93-count)/93,"exact_bus_accuracy_percent":100*metric(tag,model,"localization","exact_bus_accuracy"),"mean_branch_hops":metric(tag,model,"localization","mean_branch_hops")})
    sensors=pd.DataFrame(sensors)
    sensors.to_csv(ROOT / "results/sensor_budget_tradeoff.csv",index=False)
    audit={"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"threads":4,"seed":490,"distinct_test_cases":len(ids),"repeats":3,"sample_ids":m.iloc[ids].SampleID.tolist(),"sampling":"up to ten test cases per class, including all ten normals; balanced timing sample differs from test prevalence","scope":"warm single-case computing times; model loading, input validation, acquisition, phasor estimation/windowing, communications, HTTP transport and breaker action excluded","baseline_localization_scope":"predicted type computed separately; B1 localization timing excludes classification, which is separately reported","optimization":"all methods share an unoptimized full phasor/feature transform; decision-time measurements exclude that shared transform; this is a Python implementation comparison, not certified relay hardware timing"}
    (ROOT / "results/efficiency_benchmark_protocol.json").write_text(json.dumps(audit,indent=2))

    def md(df):
        return "\n".join(["| "+" | ".join(map(str,df.columns))+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+["| "+" | ".join(map(str,row))+" |" for row in df.itertuples(index=False,name=None)])
    report=f"""# Additional comparison — errors, computation time and sensor budget

This analysis uses the already frozen models. No retraining or final-test-driven selection was performed.

## What was compared

B0 is a basic fixed voltage-sag/current-threshold detector. It does not classify or locate faults. B1 combines electrical phase and zero-sequence rules with an approximate source-network current-injection residual location method. LightGBM uses separate learned detection/type/bus models. All methods receive the same available measurements within each experiment and are scored against actual held-out labels.

## A meaningful additional percentage: relative error reduction

{md(pd.DataFrame(errors).round(4))}

Accuracy gain in percentage points is 100 × (accuracy_ML − accuracy_B1). Relative error reduction is 100 × (errors_B1 − errors_ML) / errors_B1. These answer different questions. Do not call a percentage-point gain the same thing as a relative percentage improvement. Detection ties, so no error-reduction ratio is defined when both make zero observed errors. All conclusions retain the original dataset and B1 limitations.

## Actual computation benchmark

{md(summary.round(6))}

Units are milliseconds. The benchmark uses {len(ids)} distinct test cases, up to ten per class, three repeated passes, ten warm-up cases, shuffled method order and four threads. Median and 95th percentile describe computing time across the sampled runs, not a confidence interval or a worst-case guarantee. CPU/background conditions can affect results. This is a new warm single-case benchmark; do not mix it with the earlier batch-throughput timings.

{md(pd.DataFrame(time_comparison).round(4))}

Time reduction = 100 × (time_B1 − time_ML) / time_B1. Positive means less computing time for ML; negative means more. These compare the current prototype implementations on this computer, not optimized relay hardware. Localization improves computationally while ML detection and classification are slower in this implementation. The shared feature transformation is excluded from this percentage and can dominate the complete prediction path; the model-stage reduction must not be presented as an end-to-end speedup.

The shared transformation is measured separately and constructs all electrical features for the current implementation. B0/B1 rules do not intrinsically need all ML features, so the common transform is not an optimized minimum baseline preprocessing cost. B1 localization uses its predicted type; its classification cost must be added when accounting for that dependency. Model-only timings exclude loading, validation, communication and acquisition. The prototype's earlier 41.12 ms average includes validation, feature construction, all three ML models and both baselines on five replay cases; it is not a detection-only latency.

**We cannot measure physical fault detection delay from this release.** Steady-state pre/post phasors lack a fault-inception-aligned waveform stream and decision windows. End-to-end delay would include measurement/phasor estimation, communications, algorithm computation and (for clearing time) breaker action. Computing faster does not by itself establish faster fault clearing or reduced outage duration.

## Sensor-count and cost tradeoff

{md(sensors.round(4))}

PMU-count reduction is a hardware-count proxy, not measured financial savings. These are the physically local current-terminal experiments. Placements differ and are not nested, so count and instrument locations change together. Reduced sensors significantly reduce localization accuracy. Do not claim an equivalent-accuracy cheaper system from these results.

A financial model would require explicit prices/assumptions: total cost = number of PMUs × per-PMU purchase/installation cost + communications + computing + maintenance. Annualized cost, power consumption and avoided outage cost have not been measured. Unit-cost scenarios can be added later and labeled estimates, but currency savings and reduced outage costs are currently unsupported.

## Suggested presentation statement

Report the measured type/localization error reductions alongside absolute accuracy, computing-time medians/p95, and sensor-budget tradeoffs. The contribution is an accuracy–time–measurement tradeoff study; it does not assume ML is faster or cheaper. Preserve cases and metrics where B1 is better.
"""
    (ROOT / "reports/EFFICIENCY_AND_COST.md").write_text(report,encoding="utf-8")
    print(summary.round(6).to_string(index=False),flush=True)
    print(pd.DataFrame(errors).round(6).to_string(index=False),flush=True)
    print(sensors.round(4).to_string(index=False),flush=True)


if __name__=="__main__": main()
