"""
run_experiment.py — the full memory experiment: run several round counts, decode, fit the
logical error rate, save everything, and plot.

Usage (from the repo root):
    python run_experiment.py --fake                     # local test on IQM's fake Garnet (free)
    python run_experiment.py --qc garnet                # real hardware (needs IQM_TOKEN env var)
    python run_experiment.py --qc garnet --dd           # with dynamical decoupling
    python run_experiment.py --qc emerald --rounds 1 2 3 4 5 6 --shots 5000

Outputs (in results/):
    <name>_calibration.json  device error rates at the time of the run (saved first)
    <name>.json              settings, and per round count: raw counts, P_error, physical qubits used.
                             Re-written after EVERY job, so a crash or quota stop loses nothing.
                             "status" is "in_progress" until all jobs finish, then "complete" with eps_L.
    <name>.png               P_error vs rounds with the fit
Saving the raw counts means you can re-decode later (e.g. with a better noise model)
without spending QPU time again.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_on_hardware import run_qiskit, decode_hardware_results, get_backend, get_calibration


def p_error_model(n_rounds, eps):
    """README fit: 2 P_error = 1 - (1 - 2 eps_L)^n_rounds."""
    return 0.5 * (1 - (1 - 2 * eps) ** n_rounds)


def fit_eps(rounds, p, err):
    """Fit eps_L (logical error per round) to P_error(rounds). Returns (eps, eps_err)."""
    sigma = np.maximum(np.asarray(err), 1e-6)
    popt, pcov = curve_fit(p_error_model, rounds, p, p0=[0.01], sigma=sigma,
                           absolute_sigma=True, bounds=(0, 0.5))
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", default="garnet", help="quantum computer name on Resonance")
    ap.add_argument("--fake", action="store_true", help="use IQMFakeGarnet locally instead of hardware")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--dd", action="store_true", help="enable dynamical decoupling (hardware only)")
    ap.add_argument("--tag", default="", help="extra label for the output file name")
    args = ap.parse_args()

    if args.fake:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
        backend = IQMFakeGarnet()
    else:
        backend = get_backend(args.qc)               # connect once, reuse for every job
    target = "fakegarnet" if args.fake else args.qc

    os.makedirs("results", exist_ok=True)
    name = f"{target}_d{args.distance}{'_dd' if args.dd else ''}{'_' + args.tag if args.tag else ''}" \
           f"_{datetime.now():%Y%m%d_%H%M%S}"
    json_path = f"results/{name}.json"

    # Calibration snapshot, taken right before the jobs and saved immediately
    calibration = get_calibration(backend)
    with open(f"results/{name}_calibration.json", "w") as f:
        json.dump(calibration, f, indent=1)
    print(f"  Saved calibration snapshot to results/{name}_calibration.json")

    record = dict(target=target, distance=args.distance, shots=args.shots, dd=args.dd,
                  started_at=datetime.now().isoformat(timespec="seconds"),
                  calibration_file=f"{name}_calibration.json",
                  status="in_progress", runs=[])

    def save():
        with open(json_path, "w") as f:
            json.dump(record, f, indent=1)

    save()
    for r in args.rounds:
        s = run_qiskit(distance=args.distance, rounds=r, shots=args.shots,
                       quantum_computer=args.qc, backend=backend, dd=args.dd)
        p, err = decode_hardware_results(s, args.distance, r)
        undecoded = float(s["obs_flips"][:, 0].mean())
        print(f"  rounds={r}: P_error={p:.4f} ± {err:.4f}   (no correction: {undecoded:.3f})")
        record["runs"].append(dict(rounds=r, p_error=p, p_error_err=err, undecoded=undecoded,
                                   swap_free=bool(s["swap_free"]),
                                   transpiled_depth=int(s["transpiled"].depth()),
                                   finished_at=datetime.now().isoformat(timespec="seconds"),
                                   layout=s["layout"],
                                   counts=s["counts"]))
        save()                                      # saved after EVERY job, so nothing is lost if it crashes
        print(f"  (saved to {json_path})")

    results = record["runs"]
    R   = np.array([x["rounds"] for x in results])
    P   = np.array([x["p_error"] for x in results])
    E   = np.array([x["p_error_err"] for x in results])
    eps, eps_err = (fit_eps(R, P, E) if len(R) >= 2 else (float("nan"), float("nan")))
    print(f"\n  Logical error rate per round: eps_L = {eps:.4f} ± {eps_err:.4f}")

    record.update(eps_L=eps, eps_L_err=eps_err, status="complete")
    save()

    xs = np.linspace(0, max(R) + 0.5, 200)
    plt.figure(figsize=(6, 4))
    plt.errorbar(R, P, yerr=E, fmt="o", capsize=3, label=f"{target}, d={args.distance}")
    plt.plot(xs, p_error_model(xs, eps), "-",
             label=rf"fit: $\varepsilon_L$ = {eps:.4f} ± {eps_err:.4f}")
    plt.xlabel("rounds")
    plt.ylabel(r"logical error probability $P_\mathrm{error}$")
    plt.title("Surface code memory experiment")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{name}.png", dpi=150)
    print(f"  Saved {json_path} and results/{name}.png")


if __name__ == "__main__":
    main()
