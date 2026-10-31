"""Electrical, split and local Streamlit inference checks; no web service needed."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from features import PhasorFeatures
from app.engine import get_engine


def main():
    assert (ROOT / "results/final_freeze.json").exists(),"Tests replay final cases only after selection freeze"
    e=get_engine()
    # Physically balanced positive sequence must not create zero/negative sequence.
    alpha=np.exp(2j*np.pi/3)
    seq=PhasorFeatures.sequence(np.array([[1,alpha**2,alpha]]))
    assert np.allclose(seq,[[0,1,0]],atol=1e-12)
    manifest=pd.read_csv(ROOT / "config/split_manifest.csv")
    retained=manifest[manifest.split.ne("excluded_duplicate")]
    assert retained.measurement_hash.is_unique
    assert retained.groupby("Representative_Load_Index").split.nunique().max()==1
    assert len(e.columns)==3604
    def rejects(operation):
        try: operation()
        except ValueError: return
        raise AssertionError("Invalid input was accepted")
    rejects(lambda:e.predict({}))
    rejects(lambda:e.replay(-1))
    training_id=int(manifest.loc[manifest.split.eq("train"),"SampleID"].iloc[0])
    rejects(lambda:e.replay(training_id))
    saved=pd.read_csv(ROOT / "results/full_test_predictions.csv")
    records=[]
    for condition in ["Normal","LGA","LLAB","LLABG","LLLG"]:
        row=saved[saved.Fault_Type.eq(condition)].iloc[0]
        result=e.replay(int(row.SampleID)); ml=result["lightgbm"]
        assert ml["fault_detected"]==bool(row.LightGBM_detection)
        if ml["fault_detected"]:
            assert ml["fault_bus"]==row.LightGBM_location
        else:
            assert ml["fault_type"] is None and ml["fault_bus"] is None
        assert result["b1"]["fault_detected"]==bool(row.B1_detection)
        if result["b1"]["fault_detected"]: assert result["b1"]["fault_bus"]==row.B1_location
        records.append({"sample_id":int(row.SampleID),"condition":condition,"inference_ms":result["inference_ms"]})
    meta=pd.read_csv(ROOT / "results/cache/metadata.csv"); raw=np.load(ROOT / "results/cache/raw.npy",mmap_mode="r")
    sample=int(saved[saved.Fault_Label.gt(0)].SampleID.iloc[0]); ix=int(np.flatnonzero(meta.SampleID.eq(sample))[0])
    measurements=dict(zip(e.columns,raw[ix].tolist()))
    assert isinstance(e.predict(measurements)["lightgbm"]["fault_detected"],bool)
    invalid=measurements.copy(); invalid["Fault_Type"]="LGA"
    rejects(lambda:e.predict(invalid))
    invalid=measurements.copy(); invalid[e.columns[0]]=-1
    rejects(lambda:e.predict(invalid))
    invalid=measurements.copy(); invalid[e.columns[1]]=181
    rejects(lambda:e.predict(invalid))
    invalid=measurements.copy(); invalid[e.columns[0]]="NaN"
    rejects(lambda:e.predict(invalid))
    # Direct feature transform and cached training/evaluation path must agree.
    expected=np.load(ROOT / "results/cache/features.npy",mmap_mode="r")[ix]
    assert np.allclose(e.transform.transform(raw[ix:ix+1])[0],expected,atol=1e-5)
    result={"checks_passed":True,"checks":["balanced sequence physics","disjoint load groups","no duplicate vectors across partitions","local input schema and invalid values","replay restricted to held-out test cases","five replay types match saved frozen predictions","normal output masks bus/type","cache vs live features"],"warm_replay_latencies":records,"mean_inference_ms":float(np.mean([r["inference_ms"] for r in records])),"deployment":"one local Streamlit app calling the saved models directly"}
    (ROOT / "results/streamlit_inference_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
