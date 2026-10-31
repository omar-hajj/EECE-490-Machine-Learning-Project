"""Build the final report exclusively from executed saved experiment outputs."""
import json
import pandas as pd
from features import ROOT


def table(df):
    df=df.astype(object).where(pd.notna(df),"N/A")
    rows=["| "+" | ".join(map(str,df.columns))+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]
    rows += ["| "+" | ".join(str(v).replace("|","/") for v in row)+" |" for row in df.itertuples(index=False,name=None)]
    return "\n".join(rows)


def main():
    allmetrics=pd.read_csv(ROOT / "results/metrics.csv")
    def comparison(part):
        d=allmetrics[(allmetrics.experiment=="full")&(allmetrics.partition==part)&(allmetrics.stratum=="all")]
        wanted=[("detection","accuracy"),("detection","precision"),("detection","recall"),("detection","f1"),("detection","false_positive_rate"),("classification","accuracy"),("classification","macro_f1"),("localization","exact_bus_accuracy"),("localization","top3_accuracy"),("localization","top5_accuracy"),("localization","mean_branch_hops")]
        rows=[]
        for task,metric in wanted:
            row={"Task":task,"Metric":metric}
            for model in ["B0","B1","LightGBM"]:
                s=d[(d.task==task)&(d.metric==metric)&(d.model==model)].result
                row[model]="N/A" if s.empty or pd.isna(s.iloc[0]) else f"{s.iloc[0]:.6f}"
            rows.append(row)
        return table(pd.DataFrame(rows))
    def strata(stratum):
        d=allmetrics[(allmetrics.experiment=="full")&(allmetrics.partition=="test")&(allmetrics.stratum==stratum)&(allmetrics.model.isin(["B1","LightGBM"]))&(((allmetrics.task=="detection")&(allmetrics.metric=="recall"))|((allmetrics.task=="classification")&(allmetrics.metric=="macro_f1"))|((allmetrics.task=="localization")&(allmetrics.metric=="exact_bus_accuracy")))]
        z=d.pivot_table(index=["value","task","metric","n"],columns="model",values="result").reset_index().round(6)
        return table(z)
    paired=json.loads((ROOT / "results/paired_statistics.json").read_text())
    settings=json.loads((ROOT / "artifacts/ml_settings.json").read_text())
    baseline=json.loads((ROOT / "artifacts/baseline_settings.json").read_text())
    errors=json.loads((ROOT / "results/error_summary.json").read_text())
    timing=pd.read_csv(ROOT / "results/inference_timing.csv")
    cal=pd.read_csv(ROOT / "results/calibration.csv")
    deployment_path=ROOT / "results/deployment_verification.json"
    service_note=(f"The executed API contract/replay check passed. Five warm replay cases took a mean {json.loads(deployment_path.read_text())['mean_inference_ms']:.2f} ms including input validation, feature construction, all three LightGBM models, B0 and B1; model/cache loading and HTTP transport are excluded. See `results/deployment_verification.json`. Streamlit's actual app/replay harness also passed (`results/dashboard_verification.json`)." if deployment_path.exists() else "Per-case service verification has not yet produced results; do not infer it from batch prediction timings.")
    configs=[]
    for task,s in settings.items(): configs.append({"task":task,**s["config"],"temperature":s["temperature"],"threshold":s["threshold"]})
    features=[]
    for task in settings:
        z=pd.read_csv(ROOT / f"results/{task}_treeshap.csv").head(5)
        for _,r in z.iterrows(): features.append({"task":task,"feature":r.feature,"mean_abs_raw_margin":round(r.mean_abs_TreeSHAP_raw_margin,6)})
    variants=allmetrics[(allmetrics.partition=="test")&(allmetrics.stratum=="all")&(allmetrics.model.isin(["B1","LightGBM"]))&(((allmetrics.task=="detection")&(allmetrics.metric=="recall"))|((allmetrics.task=="classification")&(allmetrics.metric=="macro_f1"))|((allmetrics.task=="localization")&(allmetrics.metric.isin(["exact_bus_accuracy","top3_accuracy","mean_branch_hops"]))))]
    variants=variants.pivot_table(index=["experiment","task","metric","n"],columns="model",values="result").reset_index().round(6)
    conclusions=[]
    for r in paired:
        delta=r["accuracy_difference_ML_minus_B1"]; lo=r["cluster_bootstrap_95_lower"]; hi=r["cluster_bootstrap_95_upper"]
        conclusion="ML higher" if lo>0 else "B1 higher" if hi<0 else "difference not resolved by this interval"
        conclusions.append(f"- {r['task']}: accuracy difference ML−B1 = {100*delta:.3f} percentage points; metadata-load cluster bootstrap 95% interval [{100*lo:.3f}, {100*hi:.3f}] points ({conclusion}). ML-only correct: {r['ML_only_correct']}; B1-only correct: {r['B1_only_correct']}.")
    report=f"""# EECE 490 — final executed experiment report

Run date: 2026-10-31. Every metric is calculated from saved predictions on the verified DER-Fault release. Scores are fractions (0–1), except branch hops, seconds and explicitly labeled percentage points. This is a research decision-support prototype, not certified electrical protection.

## Research answer

{chr(10).join(conclusions)}

These are comparisons against the implemented B1, not all conventional protection techniques. Practical significance must be judged with the absolute error counts, sensor coverage, inference timing and model identifiability limits below. Detection cannot establish an ML advantage under realistic noise because normals have exactly zero disturbance. Type classification has a near-ceiling electrical baseline. Localization conclusions are limited by the approximate original circuit and B1's free-current transformer ambiguity. Shared pre-fault states prevent a claim of broad independent loading generalization.

**ML is not uniformly better.** B1 has higher full-channel top-5 location accuracy (99.5833% versus 99.4722%). With 36/21/9 available PMUs, ML has higher exact-bus accuracy but **larger mean branch-hop errors** than B1: its wrong guesses are farther away on average. Below 1 ohm, B1 type accuracy is 96.12% versus ML's 95.35% (one extra correct case among 129), so this small bin does not establish a reliable ML advantage. The most defensible conclusion is improved exact-bus ranking on this benchmark, with task- and metric-dependent tradeoffs.

## Held-out test comparison

{comparison('test')}

## Validation comparison

{comparison('validation')}

Validation selected the models/rules, so these scores are selection-biased. Temperature scaling is also fitted on validation; its validation calibration metrics are fitted values. Final test scores were generated only after all full/sensor/unseen variants were selected and SHA-256-frozen in `results/final_freeze.json`. No final-test-based retuning was performed.

## Test composition and uncertainty

The final split contains **3,610 distinct cases: 3,600 faults (360 per type) and 10 normals**, across 12 held-out representative-load indices and all six DER configurations. All 164 faulted-bus classes occur. The train and validation splits have 10,839/3,608 cases, with 39/8 normals. See `config/split_manifest.csv` for exact IDs. 1,743 redundant normal cases are excluded globally with training/validation/test priority, and no identical full measurement vector crosses partitions. The resulting prevalence is artificial; accuracy and precision do not estimate utility field performance.

FPR uses only 10 distinct test normals. With zero observed false alarms, the exact two-sided 95% upper bound is about 30.85%, not proof of negligible FPR. Recall at a 1% FPR target is **N/A** for every method because independent normal-event support is insufficient. By-scenario FPR is N/A when no retained normal case exists. More varied independent normal events are the exact next requirement for a meaningful low-FPR study.

Type and bus metrics above are conditional on true fault cases; B1 uses its predicted fault type to localize. `metrics.csv` additionally reports detection-gated end-to-end type/location accuracy. Normal bus metadata is never a fault target. B0 supports detection only, so unsupported tasks are N/A. Full confusion matrices and per-class precision/recall are in `results/confusion/`; readable detection/type plots are in `reports/figures/`.

## DER scenario analysis

{strata('Scenario_ID')}

S1/S2/S3/S4/S5/S6 configure PV/EV/V2G percentages as 0/0/0, 30/0/0, 50/20/0, 70/30/0, 70/30/30 and 30/50/50. Profile multipliers also vary by hour. Different scenarios have different sampled fault buses and resistances. These results are scenario associations, not isolated causal effects of PV. S4/S5 share PV and EV levels while V2G differs.

ML exact-bus accuracy ranges from 77.67% (S2) to 81.33% (S6), without a monotonic decline as configured PV increases. Its exact-bus advantage over B1 is 22.17 percentage points in S1, 20.83 in S2 and 28.67 in S6; this is not a controlled PV trend. The observed variation is modest relative to the main overall gap and mixes configuration, location and resistance effects.

![Held-out DER performance](figures/der_performance.png)

## Fault resistance analysis

{strata('resistance_bin')}

Bins were specified before final evaluation: 0–1, 1–5, 5–10, 10–15 and 15–25 ohm. Normal events are excluded from resistance bins because zero resistance in normal metadata is not a fault resistance. The EDA shows decreasing disturbance with resistance; model scores must be assessed separately, rather than assuming a weaker signal automatically means an ML advantage. Faults were not paired across resistance, so location/type distributions also influence each bin.

In the 15–25 ohm test bin, type accuracy is 97.73% for B1 and 99.79% for ML. The paired error analysis finds that ML-only type successes have median resistance 19.1245 ohm, while B1-only type successes have median 0.811 ohm. This supports an association between ML's type-classification benefit and weaker, high-resistance signatures in the available cases; it is not a controlled causal resistance experiment.

![Held-out resistance performance](figures/resistance_performance.png)

## Limited-sensor and unseen-configuration experiments

{table(variants)}

`full` uses all released measurements: 93 voltage buses and 726 current channels, including 339 remote terminal channels outside the published PMU_93 buses. It must not be described as a physically local 93-PMU measurement system. `available_93` retains the same 93 voltage buses but only 387 local current channels. This provides the reference for sensor-count reductions. `available_36`, `available_21` and `available_9` use intersections of the published PMU_56/36/20 lists with released voltage buses; they are **not** reproductions of the published observability placements. In every `available_N` case only line-current terminals at retained PMU buses are available. Both B1 and ML are independently selected using the same remaining measurements, load split and two-candidate ML search budget. Feature selection is fitted separately on training data for every variant. Model budget is fixed; this is not an exhaustive optimization.

`unseen_s6` excludes S6 from both training and selection validation, then evaluates S6 at held-out test loads. Its baseline is also tuned without S6. This avoids mixing seen loading groups into the shift test. Compare it to the full model's S6 rows in the scenario table. S6 changes EV/V2G as well as PV; it is a DER configuration shift, not a pure PV extrapolation.

On the same 600 held-out S6 faults, ML exact-bus accuracy is 81.17% without S6 training versus 81.33% with S6 included: one fewer correct case. This benchmark does not show material S6 localization degradation. The repeated electrical-state findings and lack of the exact generation code prevent extending that observation to real unseen DER control behavior.

Sensor placements are not nested. Reduced configurations change instrument locations as well as counts; do not claim a controlled causal sensor-count effect.

![Held-out sensor localization](figures/sensor_localization_performance.png)

## Best LightGBM models and settings

{table(pd.DataFrame(configs).round(6))}

All candidates use learning rate 0.08, maximum bin count 63, L2 penalty 1, seed 490, deterministic column-wise trees, four threads and early stopping patience 20. Training-only ANOVA screening selects k features separately for each task. Detection has balanced training class weights; other tasks use natural class frequencies. The two candidates trade 256/512 features, 7/15 leaves, 100/160 maximum rounds and 10/5 minimum child samples. Ties favor the first candidate. Saved bundles are `artifacts/full_detection.joblib`, `full_classification.joblib`, and `full_localization.joblib`; each includes the selector, trained model and temperature/threshold. `config/features.json` defines the shared deterministic preprocessing.

B1 validation-selected rule settings:

```json
{json.dumps(baseline,indent=2)}
```

## Explainability and electrical interpretation

{table(pd.DataFrame(features))}

Native LightGBM TreeSHAP contributions were actually calculated on 100 reproducibly sampled held-out cases per task. They explain raw tree margins before temperature scaling; multiclass summaries average the predicted class's contributions. Additivity checks and sampled IDs are in `results/explainability_audit.json`. The contribution plots are in `reports/figures/*_treeshap.png`; complete rankings are in `results/*_treeshap.csv`.

`V_delta_re/im` express complex voltage displacement; sag and phase rotation both contribute. `I_delta_re/im` express incremental current on a particular line terminal and physical phase. Magnitude ratios describe relative disturbances, while absolute post-fault voltage/current helps identify where a fault lies. Incremental zero sequence indicates ground return; negative sequence indicates imbalance. Features with `PhN` retain source column names, so interpret their physical phase through `config/topology.json` NodeOrder, not the conductor ordinal. Attribution is statistical evidence conditional on correlated sensors, not a causal proof or a relay setting.

## Major failures and paired cases

{table(pd.DataFrame(errors).drop(columns='example_sample_ids').round(6))}

Examples for every paired category are in `results/classification_*_examples.csv` and `results/localization_*_examples.csv`; complete errors are in `results/error_cases.csv`. Study B1-only successes as well as ML-only successes. B1's source-model current-injection subspaces can confuse transformer primary/secondary buses because fitted current is unconstrained by a fault-resistance law. This is a material localization limitation, even when coverage includes all labeled buses. The next stronger conventional comparator should use the exact study circuit and resistance-constrained candidate solves. The ML bus classifier is closed-set: a bus absent from training cannot be predicted reliably.

## Calibration and inference time

{table(cal[(cal.experiment=='full')&(cal.partition=='test')].round(6))}

Type/location temperature scaling is fitted on validation only. Confidence is calibrated at most for this artificial simulation distribution; it is not a real-world fault probability. Detection has too few normal events for a reliable calibration claim and is displayed as an uncalibrated score.

Location calibration remains imperfect: its held-out mean maximum class score is 0.8745 while exact-bus accuracy is 0.7992, and ten-bin ECE is 0.0814. Thus the location model is overconfident on average despite validation temperature scaling. The dashboard exposes a score, not a guaranteed probability of correctness; no further calibration was fitted using final-test outcomes.

{table(timing[(timing.experiment=='full')&(timing.partition=='test')].round(6))}

These are batch prediction timings from precomputed electrical features/phasors; they exclude file loading and feature extraction. They do not measure protection clearing time or guarantee real-time deadlines. The deployment smoke test additionally records per-case end-to-end service latency when available.

{service_note}

Docker configuration is supplied, but **Docker is not installed on the execution host** (`where.exe docker` found no executable). The image has not been built or run. Exact next action: install/start Docker Desktop with Linux containers, then run `docker compose up --build` from the project root and check `/health` and the dashboard. This deployment substep remains incomplete.

## Statistical and engineering limits

Primary paired uncertainty uses 2,000 bootstrap resamples of the 12 held-out metadata load groups. `results/paired_statistics.json` also provides a descriptive discordant-pair binomial p-value, which ignores dependence and must not be treated as independent-case evidence. Multiple task/scenario/resistance comparisons are exploratory; no multiplicity-adjusted discovery claim is made. A small near-ceiling type-accuracy improvement may be statistically visible without changing practical protection decisions.

The 1,033 distinct pre-fault fingerprints, repeated across different load indices, weaken any interpretation as unseen electrical operating conditions. The dataset has no measurement noise, benign switching events, CT saturation, communication loss or dynamic inverter controls, and spans only one feeder with fault resistance up to 25 ohm. Source topology supports branch hops; it does not validate impedance-distance errors or an exact study-model reconstruction. A 436-named-bus OpenDSS graph includes secondary/source buses beyond the nominal 240-bus feeder description.

## Reproduction and next research actions

Use the commands in the project README and the pinned executed environment in `requirements-lock.txt`. Do not overwrite the frozen final experiment to improve scores; use a new experiment directory and a new independent holdout for follow-up tuning. Collect independent noisy normal events, obtain the exact generation circuit/scripts, add resistance-constrained conventional localization, and validate on a second network before making deployment or broad ML-superiority claims.
"""
    (ROOT / "reports/FINAL_PERFORMANCE_REPORT.md").write_text(report,encoding="utf-8")
    print(comparison("test"),flush=True)


if __name__=="__main__": main()
