"""
diagnose_run.py — figure out WHERE the errors come from in a saved run (no QPU time used).

Usage (from the repo root):
    python diagnose_run.py results/<run>.json

Works for results files from run_experiment.py, either track (older files without a
"track" field are treated as qiskit).

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

from tracks import TRACKS
from noise_models import Rates, annotate

BASE_P = 0.003      # background noise every comparison model has, on top of its one error type


def firing_table(det, info):
    rows = {}
    for k, d in enumerate(info):
        rows.setdefault(d["round"], {}).setdefault(d["type"], []).append(float(det[:, k].mean()))
    return rows


def model(twin, kind, p):
    """The twin with background noise plus one error type of strength p."""
    b = BASE_P
    return annotate(twin, Rates(
        gate1=lambda q: b,
        gate2=lambda a, c: b,
        readout=lambda q: b + (p if kind == "readout" else 0.0),
        after_readout=lambda q: b + (p if kind == "anc_flip" else 0.0),
        idle=lambda q: ((b + p) / 3,) * 3 if kind == "data_idle" else (b / 3,) * 3,
    ))


def correlations(det, info):
    x = det.astype(float)
    m = x.mean(0)
    cov = (x.T @ x) / len(x) - np.outer(m, m)
    t1, t2, sp = [], [], []
    n = det.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = info[i], info[j]
            same = a["ancilla"] == b["ancilla"]
            dt = abs(a["round"] - b["round"])
            close = abs(a["pos"][0] - b["pos"][0]) <= 2 and abs(a["pos"][1] - b["pos"][1]) <= 2
            if same and dt == 1:
                t1.append(cov[i, j])
            elif same and dt == 2:
                t2.append(cov[i, j])
            elif dt == 0 and not same and close:
                sp.append(cov[i, j])
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return float(det.mean()), mean(t1), mean(t2), mean(sp)


def main(path, model_rounds=3):
    r = json.load(open(path))
    d = r.get("distance", 3)
    basis = r.get("basis", "Z")
    track = TRACKS[r.get("track", "qiskit")]
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say(f"Diagnosis of {os.path.basename(path)}  ({r['target']}, track {track.name}, d={d}, "
        f"{basis}-memory, {r['shots']} shots{', DD' if r.get('dd') else ''})")
    say("\n1. Detector firing rate per round (fraction of shots each check reported a change)")
    say("   round:  X-checks   Z-checks")
    events = {}
    for run in r["runs"]:
        det, obs = track.events_from_counts(d, run["rounds"], basis, run["counts"])
        events[run["rounds"]] = det
        rows = firing_table(det, track.detector_info(d, run["rounds"], basis))
        say(f"  -- experiment with rounds={run['rounds']} (raw logical flip rate {obs[:, 0].mean():.3f})")
        for t in sorted(rows):
            xs, zs = rows[t].get("X"), rows[t].get("Z")
            fx = f"{np.mean(xs):.3f}" if xs else "  -  "
            fz = f"{np.mean(zs):.3f}" if zs else "  -  "
            say(f"     {t:>3}:    {fx:>6}     {fz:>6}")

    if model_rounds not in events:
        say(f"\n(need a run with rounds={model_rounds} for the correlation analysis)")
    else:
        info = track.detector_info(d, model_rounds, basis)
        twin = track.twin(d, model_rounds, basis)
        f, a, b, s = correlations(events[model_rounds], info)
        say(f"\n2. Error-model comparison (rounds={model_rounds}); covariance between detector pairs")
        say(f"   {'':26}{'firing':>8}{'time dt=1':>11}{'time dt=2':>11}{'space':>9}")
        say(f"   {'hardware':26}{f:>8.3f}{a:>11.4f}{b:>11.4f}{s:>9.4f}")
        for kind in ("data_idle", "anc_flip", "readout"):
            best = None
            for p in np.linspace(0.0, 0.5, 26):
                c = model(twin, kind, p)
                dd, _ = c.compile_detector_sampler().sample(20000, separate_observables=True)
                if best is None or abs(dd.mean() - f) < abs(best[1].mean() - f):
                    best = (p, dd)
            p, dd = best
            f2, a2, b2, s2 = correlations(dd, info)
            say(f"   {kind + f' (p={p:.2f})':26}{f2:>8.3f}{a2:>11.4f}{b2:>11.4f}{s2:>9.4f}")
        say("   A model 'matches' if its three covariance columns look like the hardware row.")

    out = path.replace(".json", "_diagnosis.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
