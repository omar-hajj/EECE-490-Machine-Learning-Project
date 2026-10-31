"""Run Streamlit's actual application test harness and diagnose a replay case."""
import json
from pathlib import Path
import os

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
    at.button[0].click().run(timeout=60)
    assert len(at.exception)==0,[e.message for e in at.exception]
    assert len(at.metric)==3
    result={"streamlit_app_test_passed":True,"metric_labels":[m.label for m in at.metric],"metric_values":[m.value for m in at.metric],"scope":"Actual initial app execution and held-out replay button; upload validation covered by shared engine/API contract tests. Browser pixel layout and Docker image not verified by this harness."}
    (ROOT / "results/dashboard_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
