"""
diagnose_run.py — figure out WHERE the errors come from in a saved run (no QPU time used).

Usage (from the repo root):
    python diagnose_run.py results/garnet_d3_20261024_161043.json

Prints (and saves to results/<name>_diagnosis.txt):
  1. Detector firing rate per round, split into X-checks and Z-checks.
     Healthy surface code: a few % and roughly flat across rounds.
     ~50% means that check is giving no information (pure noise).
  2. Which error model best explains the correlations between detectors (for one round count):
       data_idle : data qubits decohere while the ancillas are being read out
                   -> neighbouring checks fire TOGETHER in the same round ("space" pairs)
       anc_flip  : an ancilla changes state after being measured (non-ideal mid-circuit readout)
                   -> the same check fires in two CONSECUTIVE rounds ("time dt=1" pairs)
       readout   : the readout electronics report the wrong bit (qubit unaffected)
                   -> the same check fires two rounds apart ("time dt=2" pairs, no-reset signature)
     Each model's strength is tuned to reproduce the hardware's overall firing rate, then its
     correlation pattern is compared with the hardware's.
"""

import json
import os
import sys

import numpy as np
import stim

from surface_code import make_stim_circuit, DEFAULT_NOISE
from internal_helpers import counts_to_measurement_array, get_qubit_lists


def hardware_detectors(run, distance):
    R = run["rounds"]
    shots = sum(run["counts"].values())
    sc = make_stim_circuit(distance, R, noise=None)
    raw = counts_to_measurement_array(run["counts"], sc.num_measurements, shots)
    det, obs = sc.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    return sc, det, obs


def firing_table(sc, det):
    coords = sc.get_detector_coordinates()
    qcoords = {tuple(map(float, v)): k for k, v in sc.get_final_qubit_coordinates().items()}
    first_h = next(i for i in sc.flattened() if i.name == "H")
    x_anc = {t.value for t in first_h.targets_copy()}
    rows = {}
    for k in range(sc.num_detectors):
        x, y, t = coords[k][:3]
        typ = "X" if qcoords[(x, y)] in x_anc else "Z"
        rows.setdefault(int(t), {}).setdefault(typ, []).append(float(det[:, k].mean()))
    return rows


def model(distance, R, kind, p):
    c = make_stim_circuit(distance, R, noise=DEFAULT_NOISE)
    data, anc = get_qubit_lists(c)
    ms = [i for i, ins in enumerate(c) if ins.name == "M"]
    out = stim.Circuit()
    for i, ins in enumerate(c):
        mid = ins.name == "M" and i != ms[-1]
        if kind == "readout" and mid and p > 0:
            out.append("M", ins.targets_copy(), p)          # noisy readout
        else:
            out.append(ins)
        if mid and p > 0:
            if kind == "data_idle":
                out.append("DEPOLARIZE1", data, p)
            if kind == "anc_flip":
                out.append("X_ERROR", anc, p)
    return out


def correlations(det, c):
    coords = c.get_detector_coordinates()
    x = det.astype(float)
    m = x.mean(0)
    cov = (x.T @ x) / len(x) - np.outer(m, m)
    t1, t2, sp = [], [], []
    n = det.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            (xi, yi, ti), (xj, yj, tj) = coords[i][:3], coords[j][:3]
            same = (xi, yi) == (xj, yj)
            if same and abs(ti - tj) == 1:
                t1.append(cov[i, j])
            elif same and abs(ti - tj) == 2:
                t2.append(cov[i, j])
            elif ti == tj and not same and abs(xi - xj) <= 2 and abs(yi - yj) <= 2:
                sp.append(cov[i, j])
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return float(det.mean()), mean(t1), mean(t2), mean(sp)


def main(path, model_rounds=3):
    r = json.load(open(path))
    d = r["distance"]
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say(f"Diagnosis of {os.path.basename(path)}  ({r['target']}, d={d}, {r['shots']} shots)")
    say("\n1. Detector firing rate per round (fraction of shots each check reported a change)")
    say("   round:  X-checks   Z-checks")
    for run in r["runs"]:
        sc, det, obs = hardware_detectors(run, d)
        rows = firing_table(sc, det)
        say(f"  -- experiment with rounds={run['rounds']} (raw logical flip rate {obs[:, 0].mean():.3f})")
        for t in sorted(rows):
            xs = rows[t].get("X")
            zs = rows[t].get("Z")
            fx = f"{np.mean(xs):.3f}" if xs else "  -  "
            fz = f"{np.mean(zs):.3f}" if zs else "  -  "
            say(f"     {t:>3}:    {fx:>6}     {fz:>6}")

    run = next((x for x in r["runs"] if x["rounds"] == model_rounds), None)
    if run is None or model_rounds < 3:
        say("\n(need a run with rounds>=3 for the correlation analysis)")
    else:
        sc, det, _ = hardware_detectors(run, d)
        f, a, b, s = correlations(det, sc)
        say(f"\n2. Error-model comparison (rounds={model_rounds}); covariance between detector pairs")
        say(f"   {'':26}{'firing':>8}{'time dt=1':>11}{'time dt=2':>11}{'space':>9}")
        say(f"   {'hardware':26}{f:>8.3f}{a:>11.4f}{b:>11.4f}{s:>9.4f}")
        for kind in ("data_idle", "anc_flip", "readout"):
            best = None
            for p in np.linspace(0.0, 0.5, 26):
                c = model(d, model_rounds, kind, p)
                dd, _ = c.compile_detector_sampler().sample(20000, separate_observables=True)
                if best is None or abs(dd.mean() - f) < abs(best[1].mean() - f):
                    best = (p, dd, c)
            p, dd, c = best
            f2, a2, b2, s2 = correlations(dd, c)
            say(f"   {kind + f' (p={p:.2f})':26}{f2:>8.3f}{a2:>11.4f}{b2:>11.4f}{s2:>9.4f}")
        say("   A model 'matches' if its three covariance columns look like the hardware row.")

    out = path.replace(".json", "_diagnosis.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
