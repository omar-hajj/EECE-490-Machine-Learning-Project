"""Meaningful electrical, split, inference-contract and saved-prediction checks."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from features import PhasorFeatures
from app.engine import get_engine
from app.api import app
from fastapi.testclient import TestClient


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
    c=TestClient(app)
    assert c.get("/health").status_code==200
    assert c.get("/schema").json()["required_columns"]==3604
    assert c.post("/predict",json={"measurements":{}}).status_code==422
    assert c.post("/predict",json={"measurements":{},"Fault_Type":"LGA"}).status_code==422
    assert c.get("/replay/-1").status_code==404
    assert c.get("/cases?limit=201").status_code==422
    saved=pd.read_csv(ROOT / "results/full_test_predictions.csv")
    records=[]
    for condition in ["Normal","LGA","LLAB","LLABG","LLLG"]:
        row=saved[saved.Fault_Type.eq(condition)].iloc[0]
        response=c.get(f"/replay/{int(row.SampleID)}")
        assert response.status_code==200,response.text
        result=response.json(); ml=result["lightgbm"]
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
    assert c.post("/predict",json={"measurements":measurements}).status_code==200
    invalid=measurements.copy(); invalid[e.columns[0]]=-1
    assert c.post("/predict",json={"measurements":invalid}).status_code==422
    invalid=measurements.copy(); invalid[e.columns[1]]=181
    assert c.post("/predict",json={"measurements":invalid}).status_code==422
    invalid=measurements.copy(); invalid[e.columns[0]]="NaN"
    assert c.post("/predict",json={"measurements":invalid}).status_code==422
    # Direct feature transform and cached training/evaluation path must agree.
    expected=np.load(ROOT / "results/cache/features.npy",mmap_mode="r")[ix]
    assert np.allclose(e.transform.transform(raw[ix:ix+1])[0],expected,atol=1e-5)
    # API docs and JSON schema must be serializable and exposed.
    assert c.get("/openapi.json").status_code==200
    result={"checks_passed":True,"checks":["balanced sequence physics","disjoint load groups","no duplicate vectors across partitions","API schema and invalid inputs","five replay types match saved frozen predictions","normal output masks bus/type","cache vs live features","OpenAPI"],"warm_replay_latencies":records,"mean_inference_ms":float(np.mean([r["inference_ms"] for r in records]))}
    (ROOT / "results/deployment_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
