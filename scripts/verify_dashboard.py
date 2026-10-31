"""Run Streamlit's actual application test harness and diagnose a replay case."""
import json
from pathlib import Path
import os
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for folder in [ROOT / ".runtime/tmp", ROOT / ".runtime/mpl"]: folder.mkdir(parents=True,exist_ok=True)
os.environ["TEMP"]=os.environ["TMP"]=str(ROOT / ".runtime/tmp")
os.environ["MPLCONFIGDIR"]=str(ROOT / ".runtime/mpl")
from streamlit.testing.v1 import AppTest


def main():
    assert (ROOT / "results/final_freeze.json").exists()
    at=AppTest.from_file(str(ROOT / "app/dashboard.py"),default_timeout=60)
    at.run()
    assert len(at.exception)==0,[e.message for e in at.exception]
    assert len(at.selectbox)>=3
    assert len(at.tabs)==5
    initial_metrics={m.label:m.value for m in at.metric}
    assert initial_metrics["Fewer type errors vs B1"]=="74.0%"
    assert initial_metrics["Fewer exact-bus errors vs B1"]=="55.6%"
    summary=pd.read_csv(ROOT / "results/efficiency_timing_summary.csv").set_index("component")
    reduction=100*(1-summary.at["LightGBM_localization","median_ms"]/summary.at["B1_localization","median_ms"])
    assert initial_metrics["Less localization computation time"]==f"{reduction:.1f}%"
    frames=[element.value for element in at.dataframe]
    accuracy=next(df for df in frames if "B1 errors" in df)
    assert accuracy["B1 errors"].tolist()==[50,1627]
    assert accuracy["LightGBM errors"].tolist()==[13,723]
    times=next(df for df in frames if "Faster implementation" in df)
    assert times["Faster implementation"].tolist()==["B1","B1","LightGBM"]
    executed=next(df for df in frames if "Metric" in df)
    assert all(value=="N/A" or value.endswith("%") or value.endswith(" hops") for value in executed.Result)
    assert "99.64%" in executed.Result.tolist()
    model_table=next(df for df in frames if "Saved trees" in df)
    assert model_table["Saved trees"].tolist()==[100,1600,6396]
    assert model_table["Retained rounds"].tolist()==[100,160,39]
    split_table=next(df for df in frames if "Normals" in df)
    assert split_table.Cases.tolist()==[10839,3608,3610]
    # Verify the model selector exposes fitted multiclass parameters as well.
    at.selectbox[3].select("localization").run()
    assert len(at.exception)==0,[e.message for e in at.exception]
    assert any("multiclass" in element.value for element in at.json)
    # Exercise a fault so that its type/location scores and ranking are rendered.
    at.selectbox[1].select("LGA").run()
    at.button[0].click().run(timeout=60)
    assert len(at.exception)==0,[e.message for e in at.exception]
    assert len(at.metric)==6
    ranking=next(element.value for element in at.dataframe if "Location score" in element.value)
    assert ranking["Location score"].str.endswith("%").all()
    assert any("Type score:" in element.value and "%" in element.value for element in at.markdown)
    freeze=json.loads((ROOT / "results/final_freeze.json").read_text())
    for name,expected in freeze["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest()==expected,name
    result={"streamlit_app_test_passed":True,"metric_labels":[m.label for m in at.metric],"metric_values":[m.value for m in at.metric],"percentage_display_verified":True,"timing_and_error_reductions_verified":True,"model_inventory_and_parameters_verified":True,"frozen_artifact_hashes_unchanged":True,"scope":"Actual initial Streamlit execution, saved timing/error tables, percentage formatting, model details/selector, fault replay and unchanged frozen artifacts; upload validation covered by direct local inference tests. Browser pixel layout not verified by this harness."}
    (ROOT / "results/dashboard_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
