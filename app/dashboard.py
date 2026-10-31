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
        st.write(f"Type score: {ml['type_confidence']:.3f} · Location score: {ml['location_confidence']:.3f}")
        st.dataframe(pd.DataFrame(ml["top5_buses"]),hide_index=True)
    st.caption(result["confidence_note"])
    st.caption(f"Warm per-case inference: {result['inference_ms']:.1f} ms; includes feature computation and both comparators.")
    actual=result.get("actual",{}).get("fault_bus")
    st.pyplot(draw_network(engine().topology,ml["fault_bus"],actual))
    with st.expander("Full inference response"): st.json(result)


replay,upload,results=st.tabs(["Replay a held-out case","Upload measurements","Executed experiment results"])
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
        st.dataframe(view[["model","task","metric","result","n"]],hide_index=True)
        st.warning("Only ten distinct normal test events: low-FPR protection claims are unsupported. Scenario comparisons mix PV, EV and V2G changes.")
    else: st.write("Run the frozen held-out evaluation to populate this panel.")
