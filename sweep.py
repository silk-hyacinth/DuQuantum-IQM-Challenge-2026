# -*- coding: utf-8 -*-
"""
sweep.py — measure the logical error rate per round, epsilon_L.

Why a sweep
-----------
One hardware run gives P_error: the chance the logical qubit is wrong after n
rounds. The challenge scores epsilon_L: how much worse it gets PER ROUND. To get
it we run the same experiment at several round counts and fit the curve
(README, Google 2021/2023):

    2 P_error(n) = 1 - (1 - 2 eps_L)^n          ("README fit")

P_error starts low and rises towards 0.5 (= information completely lost); the
faster it rises, the larger eps_L.

We also report a second fit with a free amplitude A,

    2 P_error(n) = 1 - A (1 - 2 eps_L)^n        ("fit with SPAM amplitude")

because the first and last rounds of a memory experiment (state preparation
and final data readout, "SPAM") add a one-off error that is not per-round. The
README fit folds that one-off error into eps_L; the amplitude fit separates
it. Both are saved; the README fit is the headline number.

Usage
-----
    python sweep.py --dry-run                         compile all 12 circuits, no credits
    python sweep.py --label baseline                  rounds 1,2,3,4,6,8 in Z and X, 5000 shots
    python sweep.py --label dd --dd                   same with dynamical decoupling
    python sweep.py --load results/sweeps/<file>.json re-fit / re-plot a saved sweep (no credits)
    python sweep.py --load <sweep>.json --decoder decoders:garnet_uniform
                                                      re-decode the same shots with another decoder
    python sweep.py --compare A.json B.json           eps_L side by side, difference in sigma

Outputs (in results/sweeps/)
----------------------------
    <time>_<label>.json   per (basis, rounds): P_error, sigma, undecoded error,
                          detector rate, run file; per basis: fitted eps_L +- sigma
    <time>_<label>.png    P_error vs rounds with the fitted curves
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import curve_fit

from run_on_hardware import (load_hardware_results, run_qrisp_batch,
                             summarize_hardware_results)


def _p_model(n, eps):
    """README model: P_error after n rounds for per-round rate eps."""
    return 0.5 * (1.0 - (1.0 - 2.0 * eps) ** n)


def _p_model_amp(n, eps, amp):
    """Same with a one-off fidelity factor `amp` (state prep + final readout)."""
    return 0.5 * (1.0 - amp * (1.0 - 2.0 * eps) ** n)


def fit_epsilon(rounds, p_error, sigma) -> dict:
    """Fit eps_L (and optionally an amplitude) to P_error(rounds).

    Inputs : round counts, measured P_error, their 1-sigma errors.
    Output : {"eps_L", "eps_L_sigma"} for the README fit and
             {"eps_L_amp", "eps_L_amp_sigma", "amp", "amp_sigma"} for the
             amplitude fit (None if there are too few points).
    Weighted least squares: points with small error bars count more.
    """
    n = np.asarray(rounds, float)
    p = np.asarray(p_error, float)
    s = np.maximum(np.asarray(sigma, float), 1e-4)   # a 0-error point would get infinite weight
    out = {}
    popt, pcov = curve_fit(_p_model, n, p, p0=[0.1], sigma=s, absolute_sigma=True,
                           bounds=([0.0], [0.5]))
    out["eps_L"], out["eps_L_sigma"] = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    if len(n) >= 3:
        popt, pcov = curve_fit(_p_model_amp, n, p, p0=[0.1, 0.9], sigma=s, absolute_sigma=True,
                               bounds=([0.0, 0.0], [0.5, 1.5]))
        out["eps_L_amp"], out["eps_L_amp_sigma"] = float(popt[0]), float(np.sqrt(pcov[0, 0]))
        out["amp"], out["amp_sigma"] = float(popt[1]), float(np.sqrt(pcov[1, 1]))
    else:
        out.update(eps_L_amp=None, eps_L_amp_sigma=None, amp=None, amp_sigma=None)
    return out


def analyse(runs: list, decoder=None) -> dict:
    """Decode every run, group by basis, fit eps_L per basis.

    decoder : None = decode_hardware_results (PyMatching, uniform weights), or
              any function f(syndromes) -> (ler, err) to compare a new decoder.
    """
    points = []
    for run in runs:
        s = summarize_hardware_results(run, decoder=decoder)
        points.append({
            "basis": run["basis"], "rounds": run["rounds"], "shots": s["shots"],
            "p_error": s["logical_error"], "sigma": s["logical_error_sigma"],
            "undecoded": s["undecoded_logical_error"],
            "detector_rate": s["mean_detector_rate"],
            "run_file": run.get("saved_as"),
        })
    fits = {}
    for basis in sorted({p["basis"] for p in points}):
        pts = sorted((p for p in points if p["basis"] == basis), key=lambda p: p["rounds"])
        fits[basis] = fit_epsilon([p["rounds"] for p in pts], [p["p_error"] for p in pts],
                                  [p["sigma"] for p in pts])
    return {"points": points, "fits": fits}


def plot(summary: dict, path: str, title: str) -> None:
    """P_error vs rounds per basis, with error bars and both fitted curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"Z": "tab:blue", "X": "tab:red"}
    for basis, fit in summary["fits"].items():
        pts = sorted((p for p in summary["points"] if p["basis"] == basis), key=lambda p: p["rounds"])
        n = np.array([p["rounds"] for p in pts])
        c = colours.get(basis, "k")
        ax.errorbar(n, [p["p_error"] for p in pts], yerr=[p["sigma"] for p in pts], fmt="o", color=c,
                    capsize=3, label=f"{basis} memory (decoded)")
        ax.plot(n, [p["undecoded"] for p in pts], "x", color=c, alpha=0.4,
                label=f"{basis} memory (no correction)")
        grid = np.linspace(0, n.max(), 200)
        ax.plot(grid, _p_model(grid, fit["eps_L"]), "-", color=c,
                label=f"{basis}: eps_L = {fit['eps_L']:.3f} ± {fit['eps_L_sigma']:.3f}")
        if fit["eps_L_amp"] is not None:
            ax.plot(grid, _p_model_amp(grid, fit["eps_L_amp"], fit["amp"]), ":", color=c,
                    label=f"{basis}: with amplitude, eps_L = {fit['eps_L_amp']:.3f}")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("rounds")
    ax.set_ylabel("logical error probability")
    ax.set_ylim(0, 0.55)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def print_summary(summary: dict) -> None:
    for p in sorted(summary["points"], key=lambda p: (p["basis"], p["rounds"])):
        print(f"  {p['basis']} r={p['rounds']:<2} P_error={p['p_error']:.4f} ± {p['sigma']:.4f}"
              f"   no correction={p['undecoded']:.4f}   detector rate={p['detector_rate']:.3f}")
    for basis, f in summary["fits"].items():
        line = f"  eps_L[{basis}] = {f['eps_L']:.4f} ± {f['eps_L_sigma']:.4f}  (README fit)"
        if f["eps_L_amp"] is not None:
            line += (f";  {f['eps_L_amp']:.4f} ± {f['eps_L_amp_sigma']:.4f} with amplitude "
                     f"A = {f['amp']:.3f} ± {f['amp_sigma']:.3f}")
        print(line)


def compare(path_a: str, path_b: str) -> None:
    """Print eps_L of two saved sweeps side by side, with the difference and its significance.

    "Significance" = difference / combined 1-sigma. Above ~2 means the change is
    probably real; below ~1 means it is within statistical noise.
    """
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    print(f"  A = {a['label']}  (job {a.get('job_id')}, calibration {a.get('calibration_set_id')})")
    print(f"  B = {b['label']}  (job {b.get('job_id')}, calibration {b.get('calibration_set_id')})")
    if a.get("calibration_set_id") != b.get("calibration_set_id"):
        print("  NOTE: different calibration sets; part of any difference may be chip drift.")
    for basis in sorted(set(a["fits"]) & set(b["fits"])):
        fa, fb = a["fits"][basis], b["fits"][basis]
        diff = fb["eps_L"] - fa["eps_L"]
        sig = np.hypot(fa["eps_L_sigma"], fb["eps_L_sigma"])
        print(f"  eps_L[{basis}]: A {fa['eps_L']:.4f} ± {fa['eps_L_sigma']:.4f}   "
              f"B {fb['eps_L']:.4f} ± {fb['eps_L_sigma']:.4f}   "
              f"B - A = {diff:+.4f} ({diff / sig:+.1f} sigma, {100 * diff / fa['eps_L']:+.1f}%)")


def _load_decoder(spec: str):
    """'module:function' -> the function, e.g. 'my_decoders:weighted'."""
    import importlib
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round sweep and epsilon_L fit on IQM hardware.")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--bases", nargs="+", default=["Z", "X"], choices=["Z", "X"])
    parser.add_argument("--shots", type=int, default=5_000)
    parser.add_argument("--device", default="garnet")
    parser.add_argument("--label", default=None,
                        help="name for this sweep (default: baseline, or the loaded sweep's label)")
    parser.add_argument("--dd", action="store_true", help="enable IQM dynamical decoupling")
    parser.add_argument("--herald", action="store_true", help="discard shots that did not start in |0>")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--load", help="re-analyse a saved sweep summary .json (no credits)")
    parser.add_argument("--decoder", help="module:function to decode with instead of the default, "
                                          "e.g. my_decoders:weighted (use with --load)")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="compare eps_L of two saved sweeps and exit")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        raise SystemExit(0)

    os.makedirs("results/sweeps", exist_ok=True)
    decoder = _load_decoder(args.decoder) if args.decoder else None

    if args.load:
        # Re-decode the saved hardware data. Written to a NEW file so the
        # original sweep summary is never overwritten.
        with open(args.load) as f:
            saved = json.load(f)
        runs = [load_hardware_results(p["run_file"]) for p in saved["points"]]
        for run, p in zip(runs, saved["points"]):
            run["saved_as"] = p["run_file"]
        label = args.label or (saved["label"] + ("_" + args.decoder.split(":")[-1] if args.decoder else "_reanalysed"))
        stem = f"results/sweeps/{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    else:
        args.label = args.label or "baseline"
        from iqm.iqm_client import CircuitCompilationOptions, DDMode, HeraldingMode
        # max_circuit_duration_over_t2=0.0 switches off the server's "circuit
        # longer than T2" rejection, which would otherwise refuse the long
        # sweeps. It does not change how the circuit runs.
        options = CircuitCompilationOptions(
            max_circuit_duration_over_t2=0.0,
            dd_mode=DDMode.ENABLED if args.dd else DDMode.DISABLED,
            heralding_mode=HeraldingMode.ZEROS if args.herald else HeraldingMode.NONE,
        )
        configs = [{"rounds": r, "basis": b} for b in args.bases for r in args.rounds]
        runs = run_qrisp_batch(configs, shots=args.shots, quantum_computer=args.device,
                               compilation_options=options, dry_run=args.dry_run, label=args.label)
        if args.dry_run:
            raise SystemExit(0)
        label = args.label
        stem = f"results/sweeps/{time.strftime('%Y%m%d-%H%M%S')}_{label}"

    summary = analyse(runs, decoder=decoder)
    summary.update(label=label, job_id=runs[0].get("job_id"),
                   calibration_set_id=runs[0].get("calibration_set_id"),
                   compilation_options=runs[0].get("compilation_options"))
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    plot(summary, stem + ".png", f"Garnet d=3 surface code, {label}")
    print_summary(summary)
    print(f"saved {stem}.json and {stem}.png")
