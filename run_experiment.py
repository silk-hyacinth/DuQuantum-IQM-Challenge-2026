"""
run_experiment.py — the full memory experiment: run several round counts, decode, fit the
logical error rate, save everything, and plot.

Usage (from the repo root):
    python run_experiment.py --fake                         # local test on IQM's fake Garnet (free)
    python run_experiment.py --qc garnet                    # real hardware (needs IQM_TOKEN env var)
    python run_experiment.py --qc garnet --track qrisp      # the Qrisp-branch circuit instead
    python run_experiment.py --qc garnet --dd               # with dynamical decoupling
    python run_experiment.py --qc garnet --noise calibration --decoder pymatching

Options
    --track     qiskit (default) | qrisp       how the circuit is built and compiled (tracks.py)
    --noise     uniform (default) | uniform:P | calibration
                                               the decoder's noise model (noise_models.py)
    --decoder   pymatching (default) | module:function
                                               the decoder (decoding.py)
    --qc, --fake, --distance, --rounds, --shots, --basis, --dd, --tag   as before

All round counts are submitted as ONE job, so they run back to back on one calibration.

Outputs (in results/):
    <name>_calibration.json  device error rates at the time of the run (saved first)
    <name>.json              settings, calibration, and per round count: raw counts, P_error,
                             detector rate, physical qubits used. Saved as soon as the job
                             returns, and again after decoding. "status": "complete" at the end.
    <name>.png               P_error vs rounds with both fits
Saving the raw counts means you can re-decode later (another noise model or decoder)
without spending QPU time again.

Two fits are reported:
    README fit      2 P_error = 1 - (1 - 2 eps_L)^n
    amplitude fit   2 P_error = 1 - A (1 - 2 eps_L)^n
The amplitude A absorbs the one-off cost of state preparation and final readout, so
eps_L is the cost of each extra round. On Garnet round 1 is much cheaper than later
rounds, which drags the README fit down; the amplitude fit is the more honest
per-round number. It needs at least 3 round counts.
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

from tracks import TRACKS
from noise_models import NoiseContext, add_noise, parse_spec
from decoding import get_decoder, logical_error
from run_on_hardware import get_calibration


def p_error_model(n_rounds, eps):
    """README fit: 2 P_error = 1 - (1 - 2 eps_L)^n_rounds."""
    return 0.5 * (1 - (1 - 2 * eps) ** n_rounds)


def p_error_model_amp(n_rounds, eps, amp):
    """Amplitude fit: 2 P_error = 1 - A (1 - 2 eps_L)^n_rounds."""
    return 0.5 * (1 - amp * (1 - 2 * eps) ** n_rounds)


def fit_eps(rounds, p, err):
    """README fit. Returns (eps, eps_err)."""
    sigma = np.maximum(np.asarray(err), 1e-6)
    popt, pcov = curve_fit(p_error_model, rounds, p, p0=[0.01], sigma=sigma,
                           absolute_sigma=True, bounds=(0, 0.5))
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))


def fit_eps_amp(rounds, p, err):
    """Amplitude fit. Returns (eps, eps_err, amp, amp_err); needs >= 3 round counts."""
    sigma = np.maximum(np.asarray(err), 1e-6)
    popt, pcov = curve_fit(p_error_model_amp, rounds, p, p0=[0.05, 1.0], sigma=sigma,
                           absolute_sigma=True, bounds=([0, 0], [0.5, 2]))
    e = np.sqrt(np.diag(pcov))
    return float(popt[0]), float(e[0]), float(popt[1]), float(e[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="qiskit", choices=sorted(TRACKS),
                    help="how the circuit is built: qiskit (Stim -> Qiskit) or qrisp (native CZ, Qrisp branch)")
    ap.add_argument("--noise", default="uniform",
                    help="decoder noise model: uniform | uniform:<p> | calibration (see noise_models.py)")
    ap.add_argument("--decoder", default="pymatching",
                    help="pymatching | qubit_calibrated | module:function (see decoding.py)")
    ap.add_argument("--qc", default="garnet", help="quantum computer name on Resonance")
    ap.add_argument("--fake", action="store_true", help="use IQMFakeGarnet locally instead of hardware")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--dd", action="store_true", help="enable dynamical decoupling (hardware only)")
    ap.add_argument("--basis", default="Z", choices=["Z", "X"],
                    help="Z-memory (bit-flip protection) or X-memory (phase-flip protection)")
    ap.add_argument("--tag", default="", help="extra label for the output file name")
    args = ap.parse_args()

    # Check the cheap things before spending QPU time.
    parse_spec(args.noise)
    if args.decoder != "qubit_calibrated":
        get_decoder(args.decoder)
    track = TRACKS[args.track]
    target = "fakegarnet" if args.fake else args.qc

    os.makedirs("results", exist_ok=True)
    name = (f"{target}_{args.track}_d{args.distance}{'_X' if args.basis == 'X' else ''}"
            f"{'_dd' if args.dd else ''}{'_' + args.tag if args.tag else ''}_{datetime.now():%Y%m%d_%H%M%S}")
    json_path = f"results/{name}.json"

    # Calibration snapshot, taken right before the job and saved immediately.
    if args.fake:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
        calibration = get_calibration(IQMFakeGarnet())
    else:
        calibration = get_calibration(quantum_computer=args.qc)
    with open(f"results/{name}_calibration.json", "w") as f:
        json.dump(calibration, f, indent=1)
    print(f"  Saved calibration snapshot to results/{name}_calibration.json ({calibration['source']})")
    if parse_spec(args.noise)[0] == "calibration" and calibration["source"] == "unavailable":
        raise SystemExit("--noise calibration needs calibration data, but it could not be fetched")

    record = dict(target=target, track=args.track, distance=args.distance, basis=args.basis,
                  shots=args.shots, dd=args.dd, noise=args.noise, decoder=args.decoder,
                  started_at=datetime.now().isoformat(timespec="seconds"),
                  calibration_file=f"{name}_calibration.json", calibration=calibration,
                  status="running", runs=[])

    def save():
        with open(json_path, "w") as f:
            json.dump(record, f, indent=1)

    save()

    # 1. Run every round count as one job.
    runs = track.run(args.distance, args.rounds, args.basis, args.shots,
                     quantum_computer=args.qc, fake=args.fake, dd=args.dd)
    for run in runs:
        record["runs"].append({k: v for k, v in run.items() if k not in ("det_events", "obs_flips")})
    record["status"] = "decoding"
    save()                                        # raw counts are safe on disk from here on
    print(f"  (raw counts saved to {json_path})")

    # 2. Decode each round count with the chosen noise model and decoder.
    for run, saved in zip(runs, record["runs"]):
        ctx = NoiseContext(calibration=calibration, qubit_map=run["qubit_map"], dd=args.dd)
        twin = track.twin(args.distance, run["rounds"], args.basis)
        if args.decoder == "qubit_calibrated":
            # Noiseless twin: logical_error adds one weight per physical qubit.
            p, err, _ = logical_error(run["det_events"], run["obs_flips"], twin,
                                      args.decoder, context=ctx)
        else:
            noisy = add_noise(twin, args.noise, ctx)
            p, err, _ = logical_error(run["det_events"], run["obs_flips"], noisy, args.decoder)
        undecoded = float(np.asarray(run["obs_flips"])[:, 0].mean())
        det_rate = float(np.asarray(run["det_events"]).mean())
        saved.update(p_error=p, p_error_err=err, undecoded=undecoded, detector_rate=det_rate)
        print(f"  rounds={run['rounds']}: P_error={p:.4f} ± {err:.4f}   "
              f"(no correction: {undecoded:.3f}, detectors fire {det_rate:.1%})")
    save()

    # 3. Fit the logical error rate.
    R = np.array([x["rounds"] for x in record["runs"]])
    P = np.array([x["p_error"] for x in record["runs"]])
    E = np.array([x["p_error_err"] for x in record["runs"]])
    fits = {}
    if len(R) >= 2:
        eps, eps_err = fit_eps(R, P, E)
        fits["readme"] = dict(eps_L=eps, eps_L_err=eps_err)
        print(f"\n  README fit:     eps_L = {eps:.4f} ± {eps_err:.4f} per round")
    if len(R) >= 3:
        eps_a, eps_a_err, amp, amp_err = fit_eps_amp(R, P, E)
        fits["amplitude"] = dict(eps_L=eps_a, eps_L_err=eps_a_err, amp=amp, amp_err=amp_err)
        print(f"  amplitude fit:  eps_L = {eps_a:.4f} ± {eps_a_err:.4f} per round  (A = {amp:.3f} ± {amp_err:.3f})")
    record.update(fits=fits, status="complete")
    # kept for older scripts that read these two keys
    if "readme" in fits:
        record.update(eps_L=fits["readme"]["eps_L"], eps_L_err=fits["readme"]["eps_L_err"])
    save()

    # 4. Plot.
    xs = np.linspace(0, max(R) + 0.5, 200)
    plt.figure(figsize=(6.5, 4))
    plt.errorbar(R, P, yerr=E, fmt="o", capsize=3,
                 label=f"{target}, {args.track}, d={args.distance}, {args.basis}-memory{', DD' if args.dd else ''}")
    if "readme" in fits:
        f = fits["readme"]
        plt.plot(xs, p_error_model(xs, f["eps_L"]), "--", color="gray",
                 label=rf"README fit: $\varepsilon_L$ = {f['eps_L']:.4f} ± {f['eps_L_err']:.4f}")
    if "amplitude" in fits:
        f = fits["amplitude"]
        plt.plot(xs, p_error_model_amp(xs, f["eps_L"], f["amp"]), "-",
                 label=rf"amplitude fit: $\varepsilon_L$ = {f['eps_L']:.4f} ± {f['eps_L_err']:.4f}")
    plt.xlabel("rounds")
    plt.ylabel(r"logical error probability $P_\mathrm{error}$")
    plt.title(f"Surface code memory experiment (noise: {args.noise}, decoder: {args.decoder})", fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/{name}.png", dpi=150)
    print(f"  Saved {json_path} and results/{name}.png")


if __name__ == "__main__":
    main()
