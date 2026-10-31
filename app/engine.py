"""Shared inference path for FastAPI and Streamlit; no training at inference time."""
from functools import lru_cache
from pathlib import Path
import json
import sys
import time
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
from features import PhasorFeatures, FAULT_TYPES
from conventional import rule_predictions, locate
from train_der import probabilities


class Engine:
    def __init__(self):
        schema=json.loads((ROOT / "config/features.json").read_text())
        self.columns=schema["raw_columns"]
        self.topology=json.loads((ROOT / "config/topology.json").read_text())
        self.transform=PhasorFeatures(self.columns,self.topology)
        self.models={task:joblib.load(ROOT / f"artifacts/full_{task}.joblib") for task in ["detection","classification","localization"]}
        self.settings=json.loads((ROOT / "artifacts/baseline_settings.json").read_text())
        self.network=joblib.load(ROOT / "artifacts/b1_network.joblib")

    def validate(self,measurements):
        missing=set(self.columns)-set(measurements)
        extra=set(measurements)-set(self.columns)
        if missing or extra:
            raise ValueError(f"Expected exactly {len(self.columns)} electrical columns; missing {len(missing)}, unexpected {len(extra)}. Missing examples: {sorted(missing)[:4]}; unexpected examples: {sorted(extra)[:4]}.")
        try: a=np.array([[float(measurements[c]) for c in self.columns]],dtype=np.float64)
        except (TypeError,ValueError): raise ValueError("Every measurement must be numeric")
        if not np.isfinite(a).all(): raise ValueError("NaN and infinity are not valid electrical measurements")
        for j,c in enumerate(self.columns):
            if "mag_" in c and a[0,j]<0: raise ValueError(f"Magnitude cannot be negative: {c}")
            if "_Vmag_" in c and a[0,j]==0: raise ValueError(f"Zero voltage is outside the verified training domain: {c}")
            if "ang_" in c and not -180.000001<=a[0,j]<=180.000001: raise ValueError(f"Angle must use degrees in [-180,180]: {c}")
        return a

    def predict(self,measurements):
        started=time.perf_counter(); a=self.validate(measurements)
        x,aux=self.transform.transform(a,return_aux=True)
        probs={task:probabilities(b,x)[0] for task,b in self.models.items()}
        detected=bool(probs["detection"][1]>=self.models["detection"].get("threshold",.5))
        dtype=int(self.models["classification"]["model"].classes_[np.argmax(probs["classification"])])
        order=np.argsort(-probs["localization"])
        buses=self.models["localization"]["model"].classes_
        b0,b1,typ,_=rule_predictions(aux,self.settings)
        rank,res=locate(aux,typ,self.network)
        result={
            "lightgbm":{"fault_detected":detected,"fault_score":float(probs["detection"][1]),"fault_type":FAULT_TYPES[dtype] if detected else None,"fault_bus":str(buses[order[0]]) if detected else None,
                        "type_confidence":float(probs["classification"].max()) if detected else None,"location_confidence":float(probs["localization"].max()) if detected else None,
                        "top5_buses":[{"bus":str(buses[i]),"score":float(probs["localization"][i])} for i in order[:5]] if detected else []},
            "b0":{"fault_detected":bool(b0[0]),"fault_type":None,"fault_bus":None},
            "b1":{"fault_detected":bool(b1[0]),"fault_type":FAULT_TYPES[int(typ[0])] if b1[0] else None,"fault_bus":str(rank[0,0]) if b1[0] else None,"top5_buses":rank[0].tolist() if b1[0] else [],"relative_network_residual":float(res[0]) if b1[0] else None},
            "confidence_note":"Detection is an uncalibrated model score. Type/location scores use validation temperature scaling and apply only to this simulation distribution. B1 residual is not a probability.",
            "scope_note":"Steady-state DER-Fault research prototype; closed-set bus prediction. Not certified electrical protection. Unknown networks, channels, units and dynamic/noisy conditions are unsupported."
        }
        result["inference_ms"]=1000*(time.perf_counter()-started)
        return result

    def replay(self,sample_id):
        manifest=pd.read_csv(ROOT / "config/split_manifest.csv")
        selected=manifest[(manifest.SampleID==sample_id)&(manifest.split=="test")]
        if selected.empty: raise ValueError("Replay accepts only retained held-out test SampleIDs")
        meta=pd.read_csv(ROOT / "results/cache/metadata.csv")
        matches=np.flatnonzero(meta.SampleID.to_numpy()==sample_id)
        if len(matches)!=1: raise ValueError("Replay cache is unavailable or inconsistent")
        a=np.load(ROOT / "results/cache/raw.npy",mmap_mode="r")
        result=self.predict(dict(zip(self.columns,a[matches[0]])))
        m=meta.iloc[matches[0]]
        result["actual"]={"SampleID":int(m.SampleID),"fault_type":m.Fault_Type,"fault_bus":m.Fault_Bus if m.Fault_Label>0 else None,"scenario":m.Scenario_ID,"resistance_ohm":float(m.Resistance) if m.Fault_Label>0 else None}
        return result


@lru_cache(maxsize=1)
def get_engine(): return Engine()
