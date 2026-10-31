"""Run from repository root: python -m streamlit run app/dashboard.py"""
from pathlib import Path
import sys
import json
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.engine import get_engine
from app.network_plot import draw_network

st.set_page_config(page_title="DER-Fault diagnosis",page_icon="☀",layout="wide")
st.title("Solar-integrated network fault diagnosis")
st.caption("EECE 490 · LightGBM and conventional protection · DER-Fault simulation replay")
st.info("Research decision support. This prototype does not replace certified electrical protection equipment.")


@st.cache_resource
def engine(): return get_engine()


def format_result(metric, value):
    if pd.isna(value): return "N/A"
    if metric in {"mean_branch_hops", "median_branch_hops"}: return f"{value:.3f} hops"
    return f"{100*value:.2f}%"


def show_efficiency():
    paths=[ROOT / "results" / name for name in [
        "relative_error_reduction.csv", "efficiency_timing_summary.csv",
        "sensor_budget_tradeoff.csv", "efficiency_benchmark_protocol.json"]]
    if not all(path.exists() for path in paths):
        st.info("Run scripts/benchmark_efficiency.py to generate the measured comparison.")
        return
    errors=pd.read_csv(paths[0]).set_index("task")
    timing=pd.read_csv(paths[1]).set_index("component")
    sensors=pd.read_csv(paths[2])
    protocol=json.loads(paths[3].read_text())
    location_reduction=100*(1-timing.at["LightGBM_localization","median_ms"]/timing.at["B1_localization","median_ms"])
    st.subheader("Who diagnoses faults better, and who computes faster?")
    st.write("B0 uses fixed detection thresholds. B1 uses electrical rules and a network-based location method. LightGBM uses the saved learned models. These are measured results from the existing frozen experiment.")
    cols=st.columns(3)
    cols[0].metric("Fewer type errors vs B1",f"{errors.at['classification','relative_error_reduction_percent']:.1f}%")
    cols[1].metric("Fewer exact-bus errors vs B1",f"{errors.at['localization','relative_error_reduction_percent']:.1f}%")
    cols[2].metric("Less localization computation time",f"{location_reduction:.1f}%")
    st.caption("Error reductions use the full held-out fault set. The time reduction compares median localization-stage computation only; shared feature preparation is excluded.")

    st.subheader("Accuracy and error reduction")
    metrics=pd.read_csv(ROOT / "results/metrics.csv")
    test=metrics[(metrics.experiment=="full") & (metrics.partition=="test") & (metrics.stratum=="all")]
    rows=[]
    for task,label in [("classification","Fault type"),("localization","Exact fault bus")]:
        r=errors.loc[task]
        name="accuracy" if task=="classification" else "exact_bus_accuracy"
        n=int(test[(test.model=="B1") & (test.task==task) & (test.metric==name)].n.iloc[0])
        rows.append({"Task":label,"Fault cases":n,"B1 accuracy":f"{100*r.B1_accuracy:.2f}%",
                     "LightGBM accuracy":f"{100*r.LightGBM_accuracy:.2f}%",
                     "B1 errors":round(n*(1-r.B1_accuracy)),"LightGBM errors":round(n*(1-r.LightGBM_accuracy)),
                     "Accuracy gain (percentage points)":f"{r.accuracy_gain_percentage_points:.2f}",
                     "Relative error reduction":f"{r.relative_error_reduction_percent:.2f}%"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.caption("Relative error reduction = (B1 errors − LightGBM errors) / B1 errors × 100%. An accuracy gain in percentage points is a different quantity.")
    st.write("Detection accuracy is 100.00% for both B1 and LightGBM on this test set. Only ten distinct normal cases are available, so this does not establish reliable field false-alarm performance. B0 detects faults only; type and location are N/A.")

    st.subheader("Measured computation time")
    rows=[]
    for task,label in [("detection","Fault detection"),("classification","Fault type"),("localization","Fault localization")]:
        b1=timing.loc[f"B1_{task}"]; ml=timing.loc[f"LightGBM_{task}"]
        faster="LightGBM" if ml.median_ms<b1.median_ms else "B1"
        reduction=100*(1-min(b1.median_ms,ml.median_ms)/max(b1.median_ms,ml.median_ms))
        rows.append({"Task":label,"B1 median (ms)":b1.median_ms,"LightGBM median (ms)":ml.median_ms,
                     "B1 p95 (ms)":b1.p95_ms,"LightGBM p95 (ms)":ml.p95_ms,
                     "Faster implementation":faster,"Time reduction vs slower method":f"{reduction:.2f}%"})
    times=pd.DataFrame(rows)
    st.dataframe(times,hide_index=True,width="stretch",column_config={
        name:st.column_config.NumberColumn(format="%.4f") for name in times if name.endswith("(ms)")})
    st.bar_chart(times.set_index("Task")[["B1 median (ms)","LightGBM median (ms)"]],horizontal=True,x_label="Median computation time (ms)")
    shared=timing.loc["shared_phasor_and_feature_transform"]
    st.caption(f"Saved benchmark: {protocol['distinct_test_cases']} distinct held-out cases × {protocol['repeats']} passes; {protocol['threads']} threads; ten warm-up cases; shuffled method order. p95 is the 95th percentile, not a worst-case guarantee. B0 detection median: {timing.at['B0_detection','median_ms']:.4f} ms. Shared feature preparation median: {shared.median_ms:.2f} ms (p95 {shared.p95_ms:.2f} ms), reported separately.")
    st.write("B1 computes detection and classification faster; LightGBM computes localization faster and makes fewer type/bus errors overall. B1 localization uses a predicted type whose classification cost is reported separately. The timings exclude feature preparation, model loading, input validation, measurement acquisition and communication. CPU load affects timings; these are prototype implementations on this computer.")
    st.info(f"Physical fault-to-detection delay was not measured. This dataset contains pre/post steady-state phasors, without an onset-aligned waveform stream. The {location_reduction:.1f}% localization-stage reduction does not represent faster physical fault clearing or an end-to-end speedup.")

    st.subheader("Sensor reduction and its accuracy cost")
    rows=[]
    for count,group in sensors.groupby("PMUs",sort=False):
        g=group.set_index("model")
        rows.append({"PMUs":int(count),"Sensor-count reduction vs 93 PMUs":f"{group.PMU_count_reduction_percent.iloc[0]:.2f}%",
                     "B1 exact-bus accuracy":f"{g.at['B1','exact_bus_accuracy_percent']:.2f}%",
                     "LightGBM exact-bus accuracy":f"{g.at['LightGBM','exact_bus_accuracy_percent']:.2f}%",
                     "B1 mean distance (hops)":g.at['B1','mean_branch_hops'],
                     "LightGBM mean distance (hops)":g.at['LightGBM','mean_branch_hops']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.caption("Sensor counts are not financial savings. These experiments use physically local terminals and different, nonnested placements. Fewer sensors reduce accuracy; under reduced sensors LightGBM makes more exact matches but has larger mean distance errors. The all-released-channel accuracy table above includes additional current terminals and is a different measurement protocol.")


def show(result):
    ml=result["lightgbm"]; b1=result["b1"]
    cols=st.columns(3)
    cols[0].metric("LightGBM detection","Fault" if ml["fault_detected"] else "Normal")
    cols[1].metric("Fault type",ml["fault_type"] or "—")
    cols[2].metric("Faulted bus",ml["fault_bus"] or "—")
    records=[]
    for label,key in [("B0","b0"),("B1","b1"),("LightGBM","lightgbm")]:
        r=result[key]; records.append({"Method":label,"Detection":"Fault" if r["fault_detected"] else "Normal","Type":r["fault_type"] or "N/A","Bus":r["fault_bus"] or "N/A"})
    st.dataframe(pd.DataFrame(records),hide_index=True,width="stretch")
    if "actual" in result: st.write("Actual held-out labels",result["actual"])
    if ml["fault_detected"]:
        st.write(f"Type score: {ml['type_confidence']:.2%} · Location score: {ml['location_confidence']:.2%}")
        ranked=pd.DataFrame(ml["top5_buses"]).rename(columns={"bus":"Bus","score":"Location score"})
        ranked["Location score"]=ranked["Location score"].map(lambda score:f"{score:.2%}")
        st.dataframe(ranked,hide_index=True)
    st.caption(result["confidence_note"])
    st.caption(f"Per-case computation: {result['inference_ms']:.1f} ms; includes feature computation and both comparators.")
    actual=result.get("actual",{}).get("fault_bus")
    st.pyplot(draw_network(engine().topology,ml["fault_bus"],actual))
    st.caption("See Accuracy, time & reductions for the repeated B1/LightGBM benchmark and held-out error reductions.")
    with st.expander("Detailed inference response"):
        display=json.loads(json.dumps(result))
        for key in ["fault_score","type_confidence","location_confidence"]:
            value=display["lightgbm"][key]
            display["lightgbm"][key]=f"{value:.2%}" if value is not None else None
        for item in display["lightgbm"]["top5_buses"]: item["score"]=f"{item['score']:.2%}"
        st.json(display)


replay,upload,results,efficiency=st.tabs(["Replay a held-out case","Upload measurements","Executed experiment results","Accuracy, time & reductions"])
with replay:
    manifest=pd.read_csv(ROOT / "config/split_manifest.csv")
    cases=manifest[manifest.split.eq("test")]
    scenario=st.selectbox("DER configuration",["All",*sorted(cases.Scenario_ID.unique())])
    fault_type=st.selectbox("Actual replay condition",["All",*sorted(cases.Fault_Type.unique())])
    if scenario!="All": cases=cases[cases.Scenario_ID.eq(scenario)]
    if fault_type!="All": cases=cases[cases.Fault_Type.eq(fault_type)]
    if len(cases):
        sample=st.selectbox("Held-out SampleID",cases.SampleID.tolist())
        if st.button("Diagnose selected case",type="primary"):
            try: show(engine().replay(int(sample)))
            except (ValueError,FileNotFoundError) as exc: st.error(str(exc))
    else: st.write("No retained case for this combination.")
with upload:
    st.write("Upload one case using exactly the released electrical column names. Voltage is in volts, current in amperes, and angles in degrees from −180 to 180. Both pre-event and post-event phasors are required.")
    e=engine()
    st.download_button("Download CSV header template",(",".join(e.columns)+"\n").encode(),file_name="measurement_template.csv",mime="text/csv")
    file=st.file_uploader("CSV (one row) or JSON measurement object",type=["csv","json"])
    if file and st.button("Diagnose uploaded case"):
        try:
            if file.name.lower().endswith("csv"):
                df=pd.read_csv(file)
                if len(df)!=1 or df.columns.duplicated().any(): raise ValueError("Upload exactly one row with unique headers")
                if "SampleID" in df: df=df.drop(columns="SampleID")
                measurements=df.iloc[0].to_dict()
            else:
                measurements=json.load(file); measurements=measurements.get("measurements",measurements)
            show(e.predict(measurements))
        except (ValueError,TypeError,KeyError) as exc: st.error(str(exc))
with results:
    path=ROOT / "results/metrics.csv"
    if path.exists():
        df=pd.read_csv(path)
        view=df[(df.experiment=="full")&(df.partition=="test")&(df.stratum=="all")]
        view=view[["model","task","metric","result","n"]].copy()
        view["result"]=[format_result(metric,value) for metric,value in zip(view.metric,view.result)]
        st.dataframe(view.rename(columns={"model":"Method","task":"Task","metric":"Metric","result":"Result","n":"Cases"}),hide_index=True,width="stretch")
        st.warning("Only ten distinct normal test events: low-FPR protection claims are unsupported. Scenario comparisons mix PV, EV and V2G changes.")
    else: st.write("Run the frozen held-out evaluation to populate this panel.")
with efficiency:
    show_efficiency()
