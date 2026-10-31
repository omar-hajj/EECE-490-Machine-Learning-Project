"""Inspect saved estimators and executed results without training or changing them."""
import hashlib
import json
from pathlib import Path
import joblib
import lightgbm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"detection": "Fault detection", "classification": "Fault type", "localization": "Faulted bus"}


def table(rows):
    df = pd.DataFrame(rows)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)])


def main():
    freeze = json.loads((ROOT / "results/final_freeze.json").read_text())
    for name, expected in freeze["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    schema = json.loads((ROOT / "config/features.json").read_text())
    split = json.loads((ROOT / "config/split.json").read_text())
    records = []
    for path in sorted((ROOT / "artifacts").glob("*.joblib")):
        task = next((task for task in TASKS if path.stem.endswith("_" + task)), None)
        if task is None:
            continue
        b = joblib.load(path)
        m = b["model"]
        records.append({
            "task": task, "experiment": b["tag"], "artifact": path.relative_to(ROOT).as_posix(),
            "estimator": type(m).__module__ + "." + type(m).__name__,
            "lightgbm_version": lightgbm.__version__, "parameters": m.get_params(),
            "booster_parameters": m.booster_.params,
            "class_count": len(m.classes_), "classes": m.classes_.tolist(),
            "selected_feature_count": len(b["indices"]), "selected_feature_names": b["feature_names"],
            "iterations": m.booster_.current_iteration(), "actual_tree_count": m.booster_.num_trees(),
            "training_record": b["config"], "temperature": b.get("temperature", 1.0),
            "decision_threshold": b.get("threshold"),
        })
    summary = {"model_family": "LightGBM gradient-boosted decision trees (gbdt)",
               "training_performed_by_this_script": False, "frozen_artifact_hashes_verified": True,
               "raw_electrical_columns": len(schema["raw_columns"]), "engineered_features": len(schema["feature_names"]),
               "split": split, "models": records}
    (ROOT / "results/model_inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    main_models = {r["task"]: r for r in records if r["experiment"] == "full"}
    model_rows = []
    for task, r in main_models.items():
        p = r["parameters"]
        model_rows.append({"Task": TASKS[task], "Classes": r["class_count"], "Selected features": r["selected_feature_count"],
                           "Max leaves/tree": p["num_leaves"], "Round limit": p["n_estimators"],
                           "Retained rounds": r["iterations"], "Saved trees": r["actual_tree_count"],
                           "Min samples/leaf": p["min_child_samples"], "Recorded candidate time (s)": f"{r['training_record']['training_seconds']:.3f}"})
    params = []
    for key in main_models["detection"]["parameters"]:
        params.append({"Parameter": key, **{TASKS[t]: str(main_models[t]["parameters"][key]) for t in TASKS}})
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    metric_rows = []
    for partition in ["validation", "test"]:
        part = metrics[(metrics.experiment == "full") & (metrics.partition == partition) & (metrics.stratum == "all")]
        for task, metric in [("detection", "accuracy"), ("detection", "recall"), ("detection", "false_positive_rate"), ("classification", "accuracy"), ("classification", "macro_f1"), ("localization", "exact_bus_accuracy"), ("localization", "top3_accuracy"), ("localization", "top5_accuracy"), ("localization", "mean_branch_hops")]:
            row = {"Partition": partition, "Task": TASKS[task], "Metric": metric}
            for model in ["B0", "B1", "LightGBM"]:
                match = part[(part.model == model) & (part.task == task) & (part.metric == metric)]
                value = match.result.iloc[0] if len(match) else float("nan")
                row[model] = "N/A" if pd.isna(value) else (f"{value:.3f} hops" if metric == "mean_branch_hops" else f"{value:.2%}")
            metric_rows.append(row)
    split_rows = []
    for part in ["train", "validation", "test"]:
        total = split["rows"][part]; normal = split["composition"][part]["Normal"]
        split_rows.append({"Partition": part, "Load groups": len(split["groups"][part]), "Total cases": total, "Faults": total-normal, "Normals": normal})
    timing = pd.read_csv(ROOT / "results/efficiency_timing_summary.csv").set_index("component")
    time_rows = [{"Task": TASKS[t], **{model + " median (ms)": f"{timing.at[model+'_'+t,'median_ms']:.4f}" for model in ["B1", "LightGBM"]}} for t in TASKS]
    error_rows = pd.read_csv(ROOT / "results/relative_error_reduction.csv").to_dict("records")
    for r in error_rows:
        for key in ["B1_accuracy", "LightGBM_accuracy"]: r[key] = f"{r[key]:.2%}"
        r["relative_error_reduction_percent"] = f"{r['relative_error_reduction_percent']:.2f}%"
        r["accuracy_gain_percentage_points"] = f"{r['accuracy_gain_percentage_points']:.3f}"
    tuned = pd.concat([pd.read_csv(ROOT / f"results/full_{task}_tuning.csv") for task in TASKS])
    shap_rows = []
    for task in TASKS:
        ranking = pd.read_csv(ROOT / f"results/{task}_treeshap.csv")
        for row in ranking[ranking.mean_abs_TreeSHAP_raw_margin > 0].head(5).itertuples(index=False):
            shap_rows.append({"Task": TASKS[task], "Feature": row.feature, "Mean absolute raw-margin attribution": f"{row.mean_abs_TreeSHAP_raw_margin:.5f}"})
    sections = ["# Project and trained-model handbook",
        "Generated by inspecting saved estimators, the frozen split and executed result files. No models were trained, retuned or recalibrated by this export. The frozen artifact checksums passed.",
        "## What we trained",
        "We trained supervised LightGBM gradient-boosted decision-tree classifiers. Boosting builds successive trees to improve the ensemble's predictions. There are three separate primary models: normal/fault detection, ten-class fault classification, and 164-class bus localization. The type and location models are fitted only on fault rows; normal predictions suppress type and bus in the application. Localization predicts one of the known labeled buses, rather than a continuous distance or an unseen bus.",
        f"The saved experiment contains {len(records)} selected model artifacts across six measurement/scenario protocols, including the three primary full-measurement models. The running dashboard uses those three primary models. Their LightGBM version is {lightgbm.__version__}.",
        table(model_rows),
        "Rounds and trees differ: binary detection has one tree per round; multiclass models have one tree per class per round. Thus 160 rounds × 10 classes = 1,600 type trees, and 39 rounds × 164 classes = 6,396 location trees. Candidate training times include the model fit and validation scoring recorded in the training loop; they are not total project runtime, preprocessing time or detection latency.",
        "## Data and physically meaningful inputs",
        f"The audited release has 19,800 cases, including 18,000 faults and 1,800 nominal normal rows, six DER scenarios and ten fault types. There are {len(schema['raw_columns']):,} electrical input columns: pre/post voltage/current magnitudes and angles. These represent 175 voltage and 726 current channels. All released currents include remote terminals, so the full protocol is not a physically local 93-PMU system.",
        "Fault labels: LGA, LGB, LGC, LLAB, LLBC, LLAC, LLABG, LLBCG, LLACG and LLLG. LGA/LGB/LGC mean phase-to-ground; LLAB/LLBC/LLAC mean phase-to-phase; the corresponding G labels include ground. LLLG is the released three-phase-ground label. Normal is the detection negative class.",
        f"The transform creates {len(schema['feature_names']):,} candidate features: pre/post magnitudes, magnitude ratios, normalized complex changes (real and imaginary), and incremental zero-, positive- and negative-sequence magnitudes for valid colocated ABC phasors. Complex phasors use magnitude × exp(j × angle). Voltage normalization uses a 1 V floor and current normalization a 1 A floor. Current phases use verified OpenDSS NodeOrder, not the source PhN ordinal. Each task selects its own features using training-only ANOVA scores. No global z-score standardizer is fitted. No missing-value imputation was necessary in the audited release; uploaded invalid measurements are rejected.",
        "Fault type, bus, resistance, scenario, loading label, IDs and timestamps are excluded from predictive inputs. The system requires both pre-event and post-event measurements; it cannot operate from a single post-event snapshot alone.",
        "## Split, duplicate control and actual unseen-data scope", table(split_rows),
        "Whole representative-load groups stay together across all DER scenarios, stratified by load category (seeds 490 and 491). Duplicate complete electrical vectors are retained once with train, validation, then test priority. This removes 1,743 repeated normals, leaving 57 distinct normal vectors. The final test has 360 examples of every fault type and all 164 labeled buses. No complete vector crosses splits. However, some prefault electrical states repeat across load labels: a held-out metadata group is not proof of wholly unseen electrical operating conditions.",
        "All candidate selection, thresholds and temperature scaling used development data before the final test was opened. Saved selected models were not refitted on the final test. The S6 shift experiment additionally excluded S6 from both training and validation. Our upload examples replay cases from the existing holdout; they are not new independent validation data.",
        "## Exact primary estimator parameters", table(params),
        "The sklearn constructor records objective=None because LightGBM inferred binary or multiclass from the labels. The actual booster objectives and every booster parameter are exported in results/model_inventory.json. Learning_rate=0.08 controls the contribution of successive trees. Num_leaves bounds tree complexity. Min_child_samples limits small leaves. Reg_lambda=1 applies L2 regularization. Max_depth=-1 means no explicit depth limit, although the leaf limit still applies. Subsample=1 and colsample_bytree=1 retain all rows/features available to the estimator; feature selection has already restricted the inputs. Class_weight='balanced' is used only for detection to compensate for very few retained normals. Seed 490, four threads, deterministic mode and column-wise construction define the recorded training setup.",
        "## Tuning, stopping and calibration",
        "Two candidate configurations per task/protocol were tested: (256 features, 7 leaves, 100 maximum rounds, 10 minimum samples/leaf) and (512 features, 15 leaves, 160 maximum rounds, 5 minimum samples/leaf). This is a modest validation search, not proof of a global optimum. Validation selection used balanced accuracy for detection, macro-F1 for type and exact-bus accuracy for location. Early stopping used validation loss with patience 20 rounds. The exported booster is truncated to its retained best iteration; the location candidate allowed 160 rounds but retained 39. Detection tied at perfect validation performance, so the first, smaller candidate was retained.",
        f"The six primary candidate-loop time records sum to {tuned.training_seconds.sum():.2f} seconds, excluding feature preparation, feature screening, calibration, serialization, variant experiments and other project work. The selected candidates alone recorded {sum(r['training_record']['training_seconds'] for r in main_models.values()):.2f} seconds. These sums are accounting of logged candidate times, not a new wall-clock benchmark.",
        f"Detection uses threshold {main_models['detection']['decision_threshold']:.2f}, chosen from validation thresholds. Its score is uncalibrated. Type and location temperature scaling was fitted to validation log loss only, with temperatures {main_models['classification']['temperature']:.8f} and {main_models['localization']['temperature']:.8f}. A displayed 90% score is not a guarantee of 90% real-world correctness. Location remains overconfident on the simulated test data (mean maximum score 87.45%, accuracy 79.92%).",
        "## B0 and B1 comparison",
        "B0 detects voltage sag ≥10% OR maximum post-event current ≥72.4992394 A (twice the training-normal 99th percentile). It has no type or location output. B1 detection uses normalized complex voltage disturbance ≥0.000001. This tiny threshold is selected under exact-no-change normal simulations and is not a field relay setting. B1 classification combines voltage/current phase signatures and voltage zero-sequence rules; selected single/triple/ground ratios are 0.35/0.50/0.15, with current weight 1. B1 localization ranks all 164 candidate buses using a source-network current-injection transfer residual, predicted phase constraints and current weight 0.1. The original source topology is an approximation to the study circuit; free candidate fault currents can confuse transformer sides. Results establish a comparison against this implemented baseline, not all conventional methods.",
        "## Actual validation and final test results", table(metric_rows),
        "Only ten distinct test normals support detection FPR. Zero observed false alarms has a two-sided exact 95% upper FPR bound of about 30.85%; recall at a 1% FPR target is unavailable. Type/location tables are conditional on actual faults. B1 has slightly better full-channel top-5 location accuracy. Validation results are selection-biased; the final test was evaluated after freezing.",
        "## Reductions and measured computation", table(error_rows), table(time_rows),
        "Fault-type errors decrease from 50 to 13; exact-bus errors from 1,627 to 723. Relative error reduction is (B1 errors − ML errors) / B1 errors × 100%. Accuracy gain in percentage points is a different quantity. Paired metadata-load bootstrap intervals are +0.722 to +1.361 points for type accuracy and +22.667 to +27.417 points for exact-bus accuracy, subject to shared electrical-state limitations.",
        "The timing benchmark uses 110 distinct test cases, ten warm-up cases, three passes, randomized method order and four threads. LightGBM localization uses 82.2% less median model-stage computation; B1 detection and type computation are faster. Feature preparation is separate (median 24.53 ms), and B1 localization's predicted-type computation is separate. Loading, validation, measurement acquisition and communications are excluded. Static pre/post phasors do not measure physical fault-to-detection or clearing delay. See EFFICIENCY_AND_COST.md for p95 and protocol details.",
        "## DER, resistance and sensor experiments",
        "The full model's exact-bus accuracy ranges from 77.67% to 81.33% across S1–S6, without a monotonic PV trend. Scenarios also change EV/V2G and sampled faults, so differences cannot be attributed to PV alone. The S6-excluded model achieves 81.17% exact-bus accuracy on 600 held-out S6 faults versus 81.33% for the model trained with S6: one correct case difference.",
        "In the 15–25 ohm bin, B1/ML type accuracy is 97.73%/99.79%. In the ≤1 ohm bin B1 has one extra correct type prediction (96.12%/95.35%, 129 cases). Resistance bins change fault draws as well as signature strength. Local sensor subsets of 93/36/21/9 PMUs show decreasing location accuracy. Relative to 93, the count reductions are 61.29%/77.42%/90.32%; these are not monetary savings. Smaller placements are intersections and nonnested. ML gives more exact matches but worse mean network-distance errors than B1 under reduced sensors.",
        "## Important electrical features", table(shap_rows),
        "Native TreeSHAP was evaluated on 100 sampled held-out cases per task. Delta_re/im features describe normalized complex changes, including phase rotation and magnitude change. Delta_seq1 describes positive-sequence change; sequence-zero features indicate ground-return imbalance. Strong line-current changes help locate network branches. TreeSHAP values explain raw model margins and are not percentages or causal evidence. The dominant detector feature is bus2004 positive-sequence voltage change, consistent with the release's exact-no-change normals; it also illustrates why realistic normal variability matters.",
        "## Application, completed work and remaining evidence",
        "The local Streamlit app loads saved model bundles directly; it does not train while predicting. It supports retained test-case replay, strict 3,604-column CSV/JSON uploads, confidence scores, actual replay labels, B0/B1 comparisons, candidate-bus ranking and a source-network schematic. The accuracy/time tab reports the actual saved benchmark. The model-details tab exposes this inventory. The user requested one simple local app; Docker and a separate API were removed.",
        "The implemented benchmark, saved models, audits, EDA, held-out evaluation, DER/resistance/sensor studies, explanations and local prototype are completed and executed. Additional scientific evidence is still needed for deployment: independently varied noisy normal events; switching and load changes; exact study circuit and generation scripts; resistance-constrained conventional localization; another feeder; dynamic inverter behavior; and waveform-aligned timing. No financial, field fault-detection-delay or certified protection claim is supported.",
        "## Files for review and defense",
        "- results/model_inventory.json: all selected estimators, exact constructor/booster parameters, class labels, selected feature names and retained tree counts.\n- artifacts/full_detection.joblib, full_classification.joblib, full_localization.joblib: the three primary saved models.\n- reports/FINAL_PERFORMANCE_REPORT.md: scenario/resistance/validation/test tables, paired uncertainty and failure analysis.\n- reports/EFFICIENCY_AND_COST.md: measured timing, relative error reductions and sensor tradeoffs.\n- reports/DATASET_FEASIBILITY.md and PREPROCESSING_AND_BASELINES.md: data audit and electrical reasoning.\n- results/confusion/ and reports/figures/: confusion matrices and figures.\n- config/split_manifest.csv: exact retained groups/case assignments.\n- results/final_freeze.json: frozen artifact checksums.\n- scripts/train_der.py and evaluate_der.py: reproducible training/evaluation logic.\n- reports/STREAMLIT_GUIDE.md: how to run and interpret the dashboard.",
        "A defensible presentation statement: We trained three LightGBM classifiers and compared them with electrical protection rules and a network-based locator on a grouped, deduplicated DER-Fault holdout. ML improved fault-type and exact-bus accuracy and reduced localization computation in this prototype. Detection tied, B1 was faster at detection/type decisions, and sensor/distance tradeoffs and simulation limitations prevent a claim of universal superiority."]
    (ROOT / "reports/MODEL_AND_PROJECT_HANDBOOK.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Verified frozen artifacts; exported {len(records)} saved models and project handbook. No training performed.")


if __name__ == "__main__":
    main()
