"""
run_experiment.py — the full memory experiment: run several round counts, decode, fit the
logical error rate, save everything, and plot.

Usage (from the repo root):
    python run_experiment.py --fake                         # local test on IQM's fake Garnet (free)
    python run_experiment.py --qc garnet                    # real hardware (needs IQM_TOKEN env var)
    python run_experiment.py --qc garnet --track qrisp      # the Qrisp-branch circuit instead
    python run_experiment.py --qc garnet --dd --herald      # dynamical decoupling + heralding
    python run_experiment.py --qc garnet --noise calibration --decoder pymatching

  No QPU time:
    python run_experiment.py --load results/<run>.json --noise calibration
                                       re-decode a saved run with another noise model / decoder
    python run_experiment.py --load results/sweeps/<sweep>.json --basis X
                                       the same for a sweep.py (Qrisp branch) file, one basis
    python run_experiment.py --compare results/<A>.json results/<B>.json
                                       eps_L of two runs side by side

Options
    --track     qiskit (default) | qrisp       how the circuit is built and compiled (tracks.py)
    --noise     uniform (default) | uniform:P | calibration
                                               the decoder's noise model (noise_models.py)
    --decoder   pymatching (default) | module:function
                                               the decoder (decoding.py)
    --herald    measure every qubit before the circuit, drop shots that did not start in 0
    --load F    re-decode the saved run F instead of running anything
    --compare A B                              compare two saved runs and exit
    --qc, --fake, --distance, --rounds, --shots, --basis, --dd, --tag   as before

All round counts are submitted as ONE job, so they run back to back on one calibration.
IQM's "circuit longer than T2" rejection is switched off so long sweeps (6-8 rounds) run.

Outputs (in results/):
    <name>_calibration.json  device error rates at the time of the run (saved first)
    <name>.json              settings, calibration, and per round count: raw counts, P_error,
                             detector rate, physical qubits used. Saved as soon as the job
                             returns, and again after decoding. "status": "complete" at the end.
    <name>.png               P_error vs rounds with both fits, and the no-correction points
A --load writes a NEW file (<original>_redecode_<noise>_<decoder>_<time>); the original is
never overwritten.

Two fits are reported:
    simple fit      2 P_error = 1 - (1 - 2 eps_L)^n          (the challenge README's formula)
    amplitude fit   2 P_error = 1 - A (1 - 2 eps_L)^n
The amplitude A absorbs the one-off cost of state preparation and final readout, so
eps_L is the cost of each extra round. On Garnet round 1 is much cheaper than later
rounds, which drags the simple fit down; the amplitude fit is the more honest
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


# ─────────────────────────────────────────────────────────────────────────────
#  Fits
# ─────────────────────────────────────────────────────────────────────────────

def p_error_model(n_rounds, eps):
    """Simple fit (the challenge README's formula): 2 P_error = 1 - (1 - 2 eps_L)^n_rounds."""
    return 0.5 * (1 - (1 - 2 * eps) ** n_rounds)


def p_error_model_amp(n_rounds, eps, amp):
    """Amplitude fit: 2 P_error = 1 - A (1 - 2 eps_L)^n_rounds."""
    return 0.5 * (1 - amp * (1 - 2 * eps) ** n_rounds)


def fit_eps(rounds, p, err):
    """Simple fit. Returns (eps, eps_err)."""
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


def fit_record(record) -> dict:
    """Both fits for the round counts in a record; prints them."""
    R = np.array([x["rounds"] for x in record["runs"]])
    P = np.array([x["p_error"] for x in record["runs"]])
    E = np.array([x["p_error_err"] for x in record["runs"]])
    fits = {}
    if len(R) >= 2:
        eps, eps_err = fit_eps(R, P, E)
        fits["simple"] = dict(eps_L=eps, eps_L_err=eps_err)
        print(f"\n  simple fit:     eps_L = {eps:.4f} ± {eps_err:.4f} per round")
    if len(R) >= 3:
        eps_a, eps_a_err, amp, amp_err = fit_eps_amp(R, P, E)
        fits["amplitude"] = dict(eps_L=eps_a, eps_L_err=eps_a_err, amp=amp, amp_err=amp_err)
        print(f"  amplitude fit:  eps_L = {eps_a:.4f} ± {eps_a_err:.4f} per round  (A = {amp:.3f} ± {amp_err:.3f})")
    return fits


# ─────────────────────────────────────────────────────────────────────────────
#  Decoding, plotting, comparing
# ─────────────────────────────────────────────────────────────────────────────

def decode_record(record, events, noise, decoder):
    """Decode every round count of `record` (events = [(det_events, obs_flips), ...])."""
    track = TRACKS[record["track"]]
    for saved, (det, obs) in zip(record["runs"], events):
        ctx = NoiseContext(calibration=record.get("calibration"),
                           qubit_map={int(k): v for k, v in (saved.get("qubit_map") or {}).items()},
                           dd=bool(record.get("dd")))
        noisy = add_noise(track.twin(record["distance"], saved["rounds"], record["basis"]), noise, ctx)
        p, err, _ = logical_error(det, obs, noisy, decoder)
        undecoded = float(np.asarray(obs)[:, 0].mean())
        det_rate = float(np.asarray(det).mean())
        saved.update(p_error=p, p_error_err=err, undecoded=undecoded, detector_rate=det_rate,
                     shots_used=int(len(det)))
        print(f"  rounds={saved['rounds']}: P_error={p:.4f} ± {err:.4f}   "
              f"(no correction: {undecoded:.3f}, detectors fire {det_rate:.1%}, {len(det)} shots)")


def plot_record(record, path):
    R = np.array([x["rounds"] for x in record["runs"]])
    P = np.array([x["p_error"] for x in record["runs"]])
    E = np.array([x["p_error_err"] for x in record["runs"]])
    U = np.array([x["undecoded"] for x in record["runs"]])
    fits = record.get("fits", {})
    xs = np.linspace(0.5, max(R) + 0.5, 200)   # from half a round: the amplitude curve can dip below 0 near n=0
    plt.figure(figsize=(6.5, 4))
    extras = ", DD" if record.get("dd") else ""
    extras += ", herald" if record.get("herald") else ""
    plt.errorbar(R, P, yerr=E, fmt="o", capsize=3, color="tab:blue",
                 label=f"{record['target']}, {record['track']}, d={record['distance']}, "
                       f"{record['basis']}-memory{extras} (decoded)")
    plt.plot(R, U, "x", color="tab:blue", alpha=0.45, label="no correction")
    if "simple" in fits:
        f = fits["simple"]
        plt.plot(xs, p_error_model(xs, f["eps_L"]), "--", color="gray",
                 label=rf"simple fit: $\varepsilon_L$ = {f['eps_L']:.4f} ± {f['eps_L_err']:.4f}")
    if "amplitude" in fits:
        f = fits["amplitude"]
        plt.plot(xs, p_error_model_amp(xs, f["eps_L"], f["amp"]), "-", color="tab:blue",
                 label=rf"amplitude fit: $\varepsilon_L$ = {f['eps_L']:.4f} ± {f['eps_L_err']:.4f}")
    plt.axhline(0.5, color="gray", lw=0.8, ls=":")
    plt.ylim(0, 0.55)
    plt.xlabel("rounds")
    plt.ylabel(r"logical error probability $P_\mathrm{error}$")
    plt.title(f"Surface code memory experiment (noise: {record['noise']}, decoder: {record['decoder']})",
              fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def compare(path_a, path_b):
    """eps_L of two saved runs side by side, the difference, and how many error bars apart it is."""
    a, b = (json.load(open(p)) for p in (path_a, path_b))

    def describe(r):
        flags = [r.get("track", "qiskit"), f"{r.get('basis', 'Z')}-memory", f"noise {r.get('noise', 'uniform')}",
                 f"decoder {r.get('decoder', 'pymatching')}"]
        flags += ["DD"] if r.get("dd") else []
        flags += ["herald"] if r.get("herald") else []
        return f"{r.get('target')} ({', '.join(flags)}), started {r.get('started_at')}"

    print(f"  A = {os.path.basename(path_a)}\n      {describe(a)}")
    print(f"  B = {os.path.basename(path_b)}\n      {describe(b)}")
    cal_a = (a.get("calibration") or {}).get("calibration_set_id")
    cal_b = (b.get("calibration") or {}).get("calibration_set_id")
    if cal_a != cal_b:
        print(f"  NOTE: different calibration sets ({cal_a} vs {cal_b}); part of any difference may be chip drift.")
    for kind in ("simple", "amplitude"):
        # files written before the rename call the simple fit "readme"
        fa, fb = (r.get("fits", {}).get(kind) or r.get("fits", {}).get("readme" if kind == "simple" else kind)
                  for r in (a, b))
        if kind == "simple" and not fa and "eps_L" in a:        # older files
            fa = {"eps_L": a["eps_L"], "eps_L_err": a["eps_L_err"]}
        if kind == "simple" and not fb and "eps_L" in b:
            fb = {"eps_L": b["eps_L"], "eps_L_err": b["eps_L_err"]}
        if not (fa and fb):
            print(f"  {kind} fit: missing in one of the files")
            continue
        diff = fb["eps_L"] - fa["eps_L"]
        sig = np.hypot(fa["eps_L_err"], fb["eps_L_err"])
        verdict = "probably real" if abs(diff / sig) > 2 else "within noise"
        print(f"  {kind:9} eps_L:  A {fa['eps_L']:.4f} ± {fa['eps_L_err']:.4f}   B {fb['eps_L']:.4f} ± {fb['eps_L_err']:.4f}"
              f"   B - A = {diff:+.4f} ({diff / sig:+.1f} sigma, {100 * diff / fa['eps_L']:+.1f}%) -> {verdict}")
    rounds = sorted({x["rounds"] for x in a["runs"]} & {x["rounds"] for x in b["runs"]})
    if rounds:
        print("  P_error per round count (A -> B):")
        pa = {x["rounds"]: x for x in a["runs"]}
        pb = {x["rounds"]: x for x in b["runs"]}
        for r in rounds:
            print(f"    rounds={r}: {pa[r]['p_error']:.4f} ± {pa[r]['p_error_err']:.4f}  ->  "
                  f"{pb[r]['p_error']:.4f} ± {pb[r]['p_error_err']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Loading saved runs (no QPU time)
# ─────────────────────────────────────────────────────────────────────────────

def load_record(path, basis="Z"):
    """A saved run -> (record, [(det_events, obs_flips) per round count]).

    Reads run_experiment.py results files, and sweep.py summaries from the Qrisp branch
    (those hold both bases; `basis` picks one).
    """
    saved = json.load(open(path))
    if "points" in saved:                                    # sweep.py summary
        from run_on_hardware import load_hardware_results
        base = os.path.dirname(os.path.dirname(os.path.abspath(path)))   # results/sweeps/x -> repo root
        points = sorted((p for p in saved["points"] if p["basis"] == basis), key=lambda p: p["rounds"])
        if not points:
            raise SystemExit(f"{path} has no {basis}-memory points")
        runs, events, meta = [], [], None
        for p in points:
            run_file = p["run_file"] if os.path.exists(p["run_file"]) else os.path.join(os.path.dirname(base), p["run_file"])
            run = load_hardware_results(run_file)
            meta = meta or run
            events.append((run["det_events"], run["obs_flips"]))
            runs.append({"rounds": run["rounds"], "qubit_map": run["placement"], "job_id": run.get("job_id"),
                         "run_file": p["run_file"]})
        compiled = str(meta.get("compilation_options", ""))
        record = dict(target=meta.get("quantum_computer", "garnet"), track="qrisp", distance=meta.get("distance", 3),
                      basis=basis, shots=meta.get("shots"), dd="DDMode.ENABLED" in compiled,
                      herald="HeraldingMode.ZEROS" in compiled,
                      calibration={**meta.get("calibration", {}), "calibration_set_id": meta.get("calibration_set_id")},
                      loaded_from=path, runs=runs)
        return record, events

    record = dict(saved)
    record.setdefault("track", "qiskit")
    record.setdefault("target", record.get("target", "garnet"))
    if not record.get("calibration") and record.get("calibration_file"):
        cal_path = os.path.join(os.path.dirname(path), record["calibration_file"])
        if os.path.exists(cal_path):
            record["calibration"] = json.load(open(cal_path))
    if (record.get("calibration") or {}).get("fidelities"):       # files from before the merge
        from noise_models import calibration_from_legacy
        record["calibration"] = calibration_from_legacy(record["calibration"])
        print("  note: old-format calibration converted; --noise calibration results are approximate")
    track = TRACKS[record["track"]]
    events = []
    for run in record["runs"]:
        if not run.get("qubit_map") and run.get("layout"):    # files from before the merge
            run["qubit_map"] = {l["stim_qubit"]: l["physical"] for l in run["layout"]}
        events.append(track.events_from_counts(record["distance"], run["rounds"], record["basis"], run["counts"]))
    record["loaded_from"] = path
    return record, events


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="qiskit", choices=sorted(TRACKS),
                    help="how the circuit is built: qiskit (Stim -> Qiskit) or qrisp (native CZ, Qrisp branch)")
    ap.add_argument("--noise", default="uniform",
                    help="decoder noise model: uniform | uniform:<p> | calibration (see noise_models.py)")
    ap.add_argument("--decoder", default="pymatching",
                    help="decoder: pymatching, or module:function (see decoding.py)")
    ap.add_argument("--qc", default="garnet", help="quantum computer name on Resonance")
    ap.add_argument("--fake", action="store_true", help="use IQMFakeGarnet locally instead of hardware")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--dd", action="store_true", help="enable dynamical decoupling (hardware only)")
    ap.add_argument("--herald", action="store_true",
                    help="drop shots where a qubit did not start in 0 (hardware only)")
    ap.add_argument("--basis", default="Z", choices=["Z", "X"],
                    help="Z-memory (bit-flip protection) or X-memory (phase-flip protection)")
    ap.add_argument("--tag", default="", help="extra label for the output file name")
    ap.add_argument("--load", metavar="FILE", help="re-decode a saved run (no QPU time)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two saved runs and exit")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    # Check the cheap things before spending QPU time.
    parse_spec(args.noise)
    get_decoder(args.decoder)
    os.makedirs("results", exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    if args.load:
        # ---- re-decode a saved run --------------------------------------------------
        record, events = load_record(args.load, args.basis)
        if parse_spec(args.noise)[0] == "calibration" and not (record.get("calibration") or {}).get("prx"):
            raise SystemExit("--noise calibration needs calibration data, and this file has none")
        record.update(noise=args.noise, decoder=args.decoder, status="decoding",
                      redecoded_at=datetime.now().isoformat(timespec="seconds"))
        stem = os.path.splitext(os.path.basename(args.load))[0]
        safe = lambda s: s.replace(":", "-").replace("/", "-")
        name = f"{stem}_redecode_{safe(args.noise)}_{safe(args.decoder)}{'_' + args.tag if args.tag else ''}_{stamp}"
        json_path = f"results/{name}.json"
        print(f"  Re-decoding {args.load} ({record['track']} track, {record['basis']}-memory) "
              f"with noise={args.noise}, decoder={args.decoder}")
        decode_record(record, events, args.noise, args.decoder)
        record.update(fits=fit_record(record), status="complete")
        with open(json_path, "w") as f:
            json.dump(record, f, indent=1)
        plot_record(record, f"results/{name}.png")
        print(f"  Saved {json_path} and results/{name}.png")
        return

    # ---- a new run ------------------------------------------------------------------
    track = TRACKS[args.track]
    target = "fakegarnet" if args.fake else args.qc
    if args.fake and (args.dd or args.herald):
        print("  note: --dd and --herald only act on real hardware; the fake backend ignores them")
    name = (f"{target}_{args.track}_d{args.distance}{'_X' if args.basis == 'X' else ''}"
            f"{'_dd' if args.dd else ''}{'_herald' if args.herald else ''}"
            f"{'_' + args.tag if args.tag else ''}_{stamp}")
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
                  shots=args.shots, dd=args.dd, herald=args.herald, noise=args.noise, decoder=args.decoder,
                  started_at=datetime.now().isoformat(timespec="seconds"),
                  calibration_file=f"{name}_calibration.json", calibration=calibration,
                  status="running", runs=[])

    def save():
        with open(json_path, "w") as f:
            json.dump(record, f, indent=1)

    save()

    # 1. Run every round count as one job.
    runs = track.run(args.distance, args.rounds, args.basis, args.shots,
                     quantum_computer=args.qc, fake=args.fake, dd=args.dd, herald=args.herald)
    for run in runs:
        record["runs"].append({k: v for k, v in run.items() if k not in ("det_events", "obs_flips")})
    record["status"] = "decoding"
    save()                                        # raw counts are safe on disk from here on
    print(f"  (raw counts saved to {json_path})")

    # 2. Decode, 3. fit, 4. plot.
    decode_record(record, [(r["det_events"], r["obs_flips"]) for r in runs], args.noise, args.decoder)
    save()
    record.update(fits=fit_record(record), status="complete")
    if "simple" in record["fits"]:                # kept for older scripts that read these keys
        record.update(eps_L=record["fits"]["simple"]["eps_L"], eps_L_err=record["fits"]["simple"]["eps_L_err"])
    save()
    plot_record(record, f"results/{name}.png")
    print(f"  Saved {json_path} and results/{name}.png")


if __name__ == "__main__":
    main()
