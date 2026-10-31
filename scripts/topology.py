"""Inspect the independently sourced Iowa OpenDSS circuit; do not modify originals."""
from pathlib import Path
import json
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from opendssdirect import dss

ROOT = Path(__file__).resolve().parents[1]


def main():
    master = next((ROOT / "data/raw/iowa/model").rglob("Master.dss"))
    cwd = os.getcwd()
    dss.Basic.AllowChangeDir(False)
    dss.Text.Command(f'compile "{master}"')
    os.chdir(cwd)
    assert dss.Solution.Converged(), "Original network load flow did not converge"
    nodes = [n.lower() for n in dss.Circuit.YNodeOrder()]
    buses = sorted({n.split(".")[0] for n in nodes})
    node_index = {n: i for i, n in enumerate(nodes)}
    edges, lines, phase_map = [], {}, {}
    for element in dss.Circuit.AllElementNames():
        if not element.lower().startswith(("line.", "transformer.")):
            continue
        dss.Circuit.SetActiveElement(element)
        names = dss.CktElement.BusNames()
        if len(names) < 2:
            continue
        endpoints = [n.split(".")[0].lower() for n in names]
        edges.append({"element": element.lower(), "u": endpoints[0], "v": endpoints[1], "kind": element.split(".")[0].lower()})
        if element.lower().startswith("line."):
            nc = dss.CktElement.NumConductors()
            nt = dss.CktElement.NumTerminals()
            order = dss.CktElement.NodeOrder()
            y = np.array(dss.CktElement.YPrim()).reshape(-1, 2)
            y = (y[:, 0] + 1j*y[:, 1]).reshape(nc*nt, nc*nt, order="F")
            indices = []
            for t in range(nt):
                for c in range(nc):
                    phase = order[t*nc+c]
                    indices.append(node_index.get(f"{endpoints[t]}.{phase}", -1))
                    phase_map[f"{element.lower()}_t{t+1}_ph{c+1}"] = {"bus": endpoints[t], "phase": int(phase)}
            lines[element.lower()] = {"buses": endpoints, "indices": indices, "y_real": y.real.tolist(), "y_imag": y.imag.tolist(), "conductors": nc}
    coords = {}
    for b in buses:
        dss.Circuit.SetActiveBus(b)
        coords[b] = [dss.Bus.X(), dss.Bus.Y()]
    raw = ROOT / "data/raw/der_fault"
    m = pd.read_csv(raw / "labels_metadata_DERfaults.csv")
    columns = next(__import__("csv").reader((raw / "DERfaults_measurements.csv").open()))[1:]
    vnodes = [c.split("_Vmag_pre")[0].lower() for c in columns if c.endswith("_Vmag_pre")]
    cnames = [c.split("_Imag_pre")[0] for c in columns if c.endswith("_Imag_pre")]
    missing_v = sorted(set(vnodes)-set(nodes))
    missing_b = sorted(set(m.loc[m.Fault_Label.gt(0), "Fault_Bus"].str.lower())-set(buses))
    missing_i = sorted(c for c in cnames if c.lower().split("_t")[0] not in lines)
    info = {"source": "https://wzy.ece.iastate.edu/publication/OpenDSS%20Model.zip", "converged": True, "buses": buses, "nodes": nodes, "edges": edges, "coordinates": coords, "line_map": lines, "current_phase_map": phase_map, "missing_voltage_nodes": missing_v, "missing_fault_buses": missing_b, "missing_current_lines": missing_i}
    (ROOT / "config/topology.json").write_text(json.dumps(info, indent=2))
    audit = {"buses": len(buses), "nodes": len(nodes), "branches": len(edges), "missing_voltage_nodes": missing_v, "missing_fault_buses": missing_b, "missing_current_lines": missing_i, "current_conductor_phase_relabels": sum(v["phase"] != int(k.rsplit("ph",1)[1]) for k,v in phase_map.items()), "caution": "Original source topology matches names only; study-specific settings, taps and DER changes are not supplied. Treat circuit impedance matching as approximate."}
    (ROOT / "reports/topology_audit.json").write_text(json.dumps(audit, indent=2))
    assert not missing_v and not missing_b and not missing_i, audit
    y = csc_matrix(dss.YMatrix.getYsparse())
    z = splu(y).solve(np.eye(len(nodes), dtype=complex))
    err = np.max(np.abs(y @ z - np.eye(len(nodes))))
    np.savez_compressed(ROOT / "artifacts/network_transfer.npz", z=z, nodes=nodes)
    print(json.dumps(audit, indent=2))
    print("max |Y Z - I|", err)


if __name__ == "__main__":
    main()
