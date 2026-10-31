"""Run: python -m uvicorn app.api:app --host 127.0.0.1 --port 8000"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from app.engine import get_engine

app=FastAPI(title="DER-Fault diagnosis prototype",version="1.0",description="EECE 490 research decision support. Steady-state phasors; not certified protection.")


class MeasurementCase(BaseModel):
    model_config=ConfigDict(extra="forbid")
    measurements:dict[str,float]


@app.get("/health")
def health():
    try:
        e=get_engine()
        return {"status":"ready","measurement_columns":len(e.columns),"model_family":"LightGBM","certified_protection":False}
    except FileNotFoundError as exc: raise HTTPException(503,"Trained model/configuration files are missing") from exc


@app.get("/schema")
def schema():
    e=get_engine()
    return {"columns":e.columns,"voltage_magnitude_units":"V","current_magnitude_units":"A","angle_units":"degrees [-180,180]","states":["pre","post"],"required_columns":len(e.columns),"metadata_predictors":[]}


@app.post("/predict")
def predict(case:MeasurementCase):
    try: return get_engine().predict(case.measurements)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    except FileNotFoundError as exc: raise HTTPException(503,"Model artifacts unavailable") from exc


@app.get("/cases")
def cases(limit:int=50,offset:int=0):
    import pandas as pd
    from app.engine import ROOT
    if not 1<=limit<=200 or offset<0: raise HTTPException(422,"limit must be 1–200 and offset nonnegative")
    m=pd.read_csv(ROOT / "config/split_manifest.csv")
    z=m[m.split.eq("test")].iloc[offset:offset+limit]
    return {"cases":z[["SampleID","Scenario_ID","Fault_Type"]].to_dict(orient="records"),"total":int(m.split.eq("test").sum())}


@app.get("/replay/{sample_id}")
def replay(sample_id:int):
    try: return get_engine().replay(sample_id)
    except ValueError as exc: raise HTTPException(404,str(exc)) from exc
    except FileNotFoundError as exc: raise HTTPException(503,"Mount results/cache for replay") from exc
