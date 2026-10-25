from surface_code import *
from stim_to_qiskit import stim_to_qiskit
from build_emerald_qubit_rotated import *
from extract_syndromes import *
from qiskit import transpile


import os
import numpy as np
import pymatching

TOKEN = ""   # leave empty and set the IQM_TOKEN environment variable instead (never commit a token)
SERVER_URL = "https://resonance.iqm.tech"

def get_backend(quantum_computer: str = "garnet", token: str = None):
    """Connect to IQM Resonance and return the backend for `quantum_computer`."""
    from iqm.qiskit_iqm import IQMProvider
    # IQM's client reads IQM_TOKEN from the environment by itself and refuses a token passed
    # both ways, so only pass one explicitly when the environment variable is NOT set.
    explicit = token or TOKEN
    if explicit and not os.environ.get("IQM_TOKEN"):
        provider = IQMProvider(SERVER_URL, quantum_computer=quantum_computer, token=explicit)
    else:
        provider = IQMProvider(SERVER_URL, quantum_computer=quantum_computer)
    return provider.get_backend()


def get_calibration(backend=None, quantum_computer: str = "garnet") -> dict:
    """
    Snapshot of the device's error rates, in one JSON-safe format for both tracks.
    Save it next to your results: calibration drifts during the day, and the
    "calibration" noise model needs the numbers from when the data was taken.

    Fields (per physical qubit, errors are probabilities, times in seconds):
        prx, readout, e01, e10, leakage, qndness, t1, t2 (Ramsey), t2_echo,
        cz["QB1__QB2"] (per coupler), plus qubits, calibration_set_id, source.
    Real hardware -> garnet_layout.fetch_calibration (exact IQM field names).
    Fake backend  -> the noise the simulator uses.
    If fetching fails, the run continues and the snapshot says "source": "unavailable".
    """
    from datetime import datetime
    from noise_models import calibration_from_fake, calibration_to_json

    out = {"retrieved_at": datetime.now().isoformat(timespec="seconds"),
           "backend": quantum_computer}
    if backend is not None and hasattr(backend, "error_profile"):          # IQM fake backend
        return {**out, "source": "fake_error_profile", **calibration_to_json(calibration_from_fake(backend))}
    try:
        from garnet_layout import fetch_calibration
        cal = fetch_calibration(quantum_computer=quantum_computer,
                                server_url=os.environ.get("IQM_SERVER_URL", SERVER_URL))
        return {**out, "source": "resonance", **calibration_to_json(cal)}
    except Exception as e:
        print(f"[get_calibration] could not fetch calibration data: {e!r}")
        return {**out, "source": "unavailable", "error": repr(e)}


def get_physical_layout(qc_t, stim_circ, backend, phys=None) -> list:
    """
    Which physical qubit each code qubit ran on, e.g.
    [{"stim_qubit": 1, "role": "data", "coords": [1, 1], "physical": "QB7"}, ...]
    Needed to match calibration numbers to the right part of the code.
    """
    data_q, anc_q = get_qubit_lists(stim_circ)
    stim_qubits = sorted(set(data_q) | set(anc_q))     # same ordering stim_to_qiskit uses
    coords = stim_circ.get_final_qubit_coordinates()
    if phys is None:                                   # circuit qubit i -> chip qubit index
        phys = qc_t.layout.initial_index_layout(filter_ancillas=True)
    out = []
    for i, sq in enumerate(stim_qubits):
        p = phys[i]
        name = backend.index_to_qubit_name(p) if hasattr(backend, "index_to_qubit_name") else f"Q{p}"
        out.append({"stim_qubit": sq, "role": "data" if sq in data_q else "ancilla",
                    "coords": [float(c) for c in coords[sq]], "physical": name})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  5.  HARDWARE EXECUTION ON IQM RESONANCE
#
#  Two equal tracks. Pick ONE. Both return the same syndromes dict
#  (at least "det_events" and "obs_flips"), so decode_hardware_results()
#  works for either.
#
#  NOTE: Emerald does not support mid-circuit reset -> use the no-reset circuit.
# ─────────────────────────────────────────────────────────────────────────────

def run_qiskit(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "garnet",
    no_reset: bool   = True,
    backend          = None,
    initial_layout   = None,
    dd: bool         = False,
    basis: str       = "Z",
) -> dict:
    """
    Runs the no-reset surface code via Qiskit and returns detection events.

        make_qiskit_circuit → transpile → backend.run → counts
        → raw measurements → extract_syndromes()

    Parameters
    ----------
    distance, rounds, shots, no_reset : experiment settings
    token            : IQM Resonance API token (normally leave None and set the IQM_TOKEN env var)
    quantum_computer : "garnet", "emerald", ...  (ignored if `backend` is given)
    backend          : optional ready-made backend, e.g. IQMFakeGarnet() for local testing
    initial_layout   : optional list of physical qubits (position i = physical qubit for circuit qubit i).
                       None -> the transpiler picks a SWAP-free placement itself.
    dd               : turn on IQM's dynamical decoupling (real hardware only)
    basis            : "Z" (stores |0>_L, tests bit flips) or "X" (stores |+>_L, tests phase flips)

    Returns
    -------
    syndromes : dict from extract_syndromes(), plus
                "counts"       raw Qiskit counts (save these!)
                "transpiled"   the circuit that actually ran
                "swap_free"    True if every CX became exactly one CZ
                "layout"       which physical qubit each code qubit ran on
    """
    if backend is None:
        backend = get_backend(quantum_computer, token)

    # 1. Circuit (Qiskit) + its Stim twin (defines the detectors for this exact circuit)
    qc        = make_qiskit_circuit(distance, rounds, basis=basis, no_reset=no_reset)
    stim_circ = make_stim_circuit(distance, rounds, basis=basis, no_reset=no_reset, noise=None)

    # 2. Transpile: rewrite into the device's native gates (r, cz) and assign physical qubits
    qc_t = transpile(qc, backend, optimization_level=3, seed_transpiler=1,
                     initial_layout=initial_layout)
    n_cx, n_cz = qc.count_ops().get("cx", 0), qc_t.count_ops().get("cz", 0)
    print(f"[run_qiskit] d={distance} basis={basis} rounds={rounds}: CX={n_cx} -> CZ={n_cz} "
          f"({'0 SWAPs' if n_cx == n_cz else f'{(n_cz - n_cx) // 3} SWAPs!'}), depth {qc_t.depth()}")

    # 3. Run
    run_kwargs = {}
    if dd:
        from iqm.iqm_client import CircuitCompilationOptions, DDMode
        run_kwargs["circuit_compilation_options"] = CircuitCompilationOptions(dd_mode=DDMode.ENABLED)
    job    = backend.run(qc_t, shots=shots, **run_kwargs)
    counts = job.result().get_counts()

    # 4. Counts -> (shots, num_measurements) array -> detection events
    raw_meas  = counts_to_measurement_array(counts, stim_circ.num_measurements, shots)
    syndromes = extract_syndromes(raw_meas, stim_circ, print_summary=False)
    syndromes.update(counts=counts, transpiled=qc_t, swap_free=(n_cx == n_cz),
                     layout=get_physical_layout(qc_t, stim_circ, backend))
    return syndromes



def _load_credentials(token: str = None) -> None:
    """Put the IQM token into the environment (from `token`, TOKEN, or the local .env)."""
    import os
    try:                                   # optional: read IQM_TOKEN from a local .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if token or TOKEN:
        os.environ["IQM_TOKEN"] = token or TOKEN
    if not os.environ.get("IQM_TOKEN"):
        raise RuntimeError("no IQM token: set IQM_TOKEN in .env or pass token=")


def run_qrisp_batch(
    configs: list,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "garnet",
    compilation_options = None,
    save_dir: str    = "results",
    dry_run: bool    = False,
    label: str       = "",
) -> list:
    """
    Run several surface code experiments as ONE hardware job (Qrisp track).

    Why one job: every circuit in a job runs back to back on the same
    calibration, so a round sweep compares like with like, and we only wait in
    the queue once. Resonance allows up to 100 circuits per job.

    What happens, step by step
    --------------------------
      1. Build each circuit with make_qrisp_experiment (surface_code.py).
      2. Read today's calibration and pick the best zero-SWAP placement
         (garnet_layout.py). On Garnet there is only one possible qubit set,
         and every config uses the same one.
      3. Compile: pin each circuit onto those physical qubits (and prove every
         CZ is on a real coupler), then turn each SQRT_Y gate into one PRX pulse.
      4. Submit all circuits in one job and wait. For each circuit the hardware
         returns "counts": how many shots produced each measurement bitstring.
      5. Convert each bitstring into detection events and the observable flip,
         using the post-processing Qrisp generated from our parity() calls.
      6. Save each circuit's run to results/ so it can be re-analysed later.

    Inputs
    ------
    configs             : list of dicts, one per circuit, with keys
                            rounds (required), basis ("Z"/"X", default "Z"),
                            distance (3), no_reset (True), idle_time (0.0)
    shots               : repetitions per circuit (Resonance max 20 000)
    token               : API token; normally leave None and keep it in .env
    quantum_computer    : "garnet"
    compilation_options : iqm.iqm_client.CircuitCompilationOptions, e.g. to turn
                          on dynamical decoupling or heralding. None = IQM defaults.
    save_dir            : where to save the runs (None = don't save)
    dry_run             : do steps 1-3 and print the compiled gate counts,
                          but do not submit (costs nothing)
    label               : short tag stored with each run and put in file names,
                          e.g. "baseline" or "dd"

    Output
    ------
    list of dicts (same order as configs), each with
      det_events : bool array (shots, n_detectors). True = that alarm fired.
      obs_flips  : bool array (shots, 1). True = the logical value came out
                   flipped before any correction.
      plus metadata (job_id, calibration_set_id, placement, rounds, basis, ...)
      that decode_hardware_results() needs.
    """
    import json
    import os
    import time
    from iqm.qrisp_iqm import IQMBackend
    from iqm.qrisp_iqm.iqm_converter import qrisp_to_iqm_converter
    from jax import jit as jax_jit
    from qrisp import PassManager, convert_to_prx
    from qrisp.jasp import make_jaspr
    from garnet_layout import fetch_calibration, fixed_layout, rank_placements

    _load_credentials(token)
    if shots > 20_000:
        raise ValueError("Resonance allows at most 20 000 shots per circuit")
    if not 1 <= len(configs) <= 100:
        raise ValueError("a job holds between 1 and 100 circuits")

    # Step 1. Build every experiment WITHOUT noise annotations (hardware
    # supplies the real noise). Tracing it ("make_jaspr") gives:
    #   qc              : the plain quantum circuit to send to the chip
    #   post_processing : a compiled function bits -> (detectors..., observable)
    prepared = []
    for cfg in configs:
        cfg = {"distance": 3, "basis": "Z", "no_reset": True, "idle_time": 0.0, **cfg}
        experiment = make_qrisp_experiment(cfg["distance"], cfg["rounds"], basis=cfg["basis"],
                                           no_reset=cfg["no_reset"], idle_time=cfg["idle_time"])
        jaspr = make_jaspr(experiment._func)()
        prepared.append((cfg, experiment, jaspr.to_qc()[-1], jax_jit(jaspr.extract_post_processing())))

    # Step 2. Calibration-ranked placement (index 0 = lowest expected faults).
    # All configs share a distance, hence a layout, hence one placement.
    if len({cfg["distance"] for cfg, *_ in prepared}) != 1:
        raise ValueError("all configs in one batch must use the same distance")
    cal = fetch_calibration(quantum_computer=quantum_computer,
                            server_url=os.environ.get("IQM_SERVER_URL", SERVER_URL))
    best = rank_placements(prepared[0][1].layout, cal)[0]

    # Step 3. An EMPTY PassManager means "do not let IQM's default compiler
    # re-route or re-place anything". We add exactly two passes:
    #   fixed_layout   : wire order = our placement, and raises on any non-native CZ
    #   convert_to_prx : SQRT_Y / SQRT_Y_DAG -> one PRX pulse each
    backend = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
        pass_manager=PassManager(),
        compilation_options=compilation_options,
    )
    backend.pm += fixed_layout(best.placement, backend._dqa.qubits, backend.connectivity)
    backend.pm += convert_to_prx

    # Everything needed to reproduce / re-decode a run later, shared by the
    # whole batch. The per-qubit calibration of the qubits we used is saved
    # too, so drift between runs can be compared.
    used = set(best.qubits)
    shared = {
        "label": label,
        "quantum_computer": quantum_computer,
        "calibration_set_id": str(backend._calibration_set_id),
        "shots": shots,
        "compilation_options": repr(compilation_options),
        "placement": {str(i): q for i, q in sorted(best.placement.items())},
        "budget_per_round": best.budget,
        "calibration": {
            key: {q: v for q, v in cal[key].items() if q in used}
            for key in ("prx", "readout", "e01", "e10", "leakage", "t1", "t2", "t2_echo")
        } | {"cz": {"__".join(sorted(p)): e for p, e in cal["cz"].items() if p <= used}},
    }
    print(f"placement on {quantum_computer}: {best.qubits}")

    # Compile each circuit once locally to count what will run (and catch
    # errors before spending credits).
    metas = []
    for cfg, _, qc, _ in prepared:
        iqm_circuit = qrisp_to_iqm_converter(backend.pm.run(qc), backend._dqa)
        op_counts = {}
        for op in iqm_circuit.instructions:
            op_counts[op.name] = op_counts.get(op.name, 0) + 1
        metas.append({**shared, **cfg, "native_ops": op_counts})
        print(f"  d={cfg['distance']} r={cfg['rounds']:<2} {cfg['basis']}: {op_counts}")
    print("no swap/move ops: every CZ is a native coupler")
    if dry_run:
        return [{"det_events": None, "obs_flips": None, **m} for m in metas]

    # Step 4. Submit all circuits as one job and wait.
    job = backend.run_async([qc for _, _, qc, _ in prepared], shots)
    print(f"submitted job {job.job_id} ({len(prepared)} circuits x {shots} shots); waiting ...")
    all_counts = job.result().all_counts

    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for (cfg, _, _, post_processing), meta, counts in zip(prepared, metas, all_counts):
        meta["job_id"] = str(job.job_id)

        # Step 5. Each distinct bitstring is post-processed once, then repeated
        # `count` times, giving one row per shot. The last output of
        # post_processing is the observable; everything before it is a
        # detector, in the same order make_qrisp_experiment returned them.
        det_rows, obs_rows, weights = [], [], []
        for bitstring, count in counts.items():
            out = np.asarray(post_processing(np.array([c == "1" for c in bitstring])), dtype=bool)
            det_rows.append(out[:-1])
            obs_rows.append(out[-1:])
            weights.append(count)
        det_events = np.repeat(np.array(det_rows), weights, axis=0)
        obs_flips = np.repeat(np.array(obs_rows), weights, axis=0)
        meta["shots_returned"] = int(det_events.shape[0])

        # Step 6. Save: .npz = the arrays, .json = metadata + raw counts. The
        # raw counts are identical to what the Resonance dashboard shows.
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            stem = (f"{stamp}_{quantum_computer}_d{cfg['distance']}_r{cfg['rounds']}_{cfg['basis']}"
                    f"{'_' + label if label else ''}_{meta['job_id'][:8]}")
            np.savez_compressed(os.path.join(save_dir, stem + ".npz"),
                                det_events=det_events, obs_flips=obs_flips)
            with open(os.path.join(save_dir, stem + ".json"), "w") as f:
                json.dump({**meta, "counts": counts}, f, indent=1)
            meta["saved_as"] = os.path.join(save_dir, stem + ".json")

        results.append({"det_events": det_events, "obs_flips": obs_flips, "counts": counts, **meta})
    if save_dir:
        print(f"saved {len(results)} runs to {save_dir}/{stamp}_*")
    return results


def run_qrisp(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "garnet",
    no_reset: bool   = True,
    basis: str       = "Z",
    idle_time: float = 0.0,
    compilation_options = None,
    save_dir: str    = "results",
    dry_run: bool    = False,
) -> dict:
    """
    Run ONE surface code memory experiment on real IQM hardware (Qrisp track).

    This is run_qrisp_batch with a single circuit; see that function for the
    step-by-step description.

    Inputs
    ------
    distance            : code distance (3 is the only one that fits Garnet)
    rounds              : stabilizer rounds; sweep this to get the per-round rate
    shots               : repetitions of the circuit (Resonance max 20 000)
    token               : API token; normally leave None and keep it in .env
    quantum_computer    : "garnet"
    no_reset            : True on hardware (no mid-circuit reset available)
    basis               : "Z" protects a bit value, "X" protects a phase
    idle_time           : extra wait (seconds) on ancillas between rounds;
                          nonzero adds iqm.delay ops, which need pulse-level access
    compilation_options : iqm.iqm_client.CircuitCompilationOptions or None
    save_dir            : where to save the run (None = don't save)
    dry_run             : compile and print gate counts, but do not submit

    Output
    ------
    dict with det_events (shots, n_detectors), obs_flips (shots, 1) and the
    run metadata decode_hardware_results() needs.
    """
    config = {"distance": distance, "rounds": rounds, "basis": basis,
              "no_reset": no_reset, "idle_time": idle_time}
    return run_qrisp_batch([config], shots=shots, token=token, quantum_computer=quantum_computer,
                           compilation_options=compilation_options, save_dir=save_dir,
                           dry_run=dry_run)[0]


def load_hardware_results(path: str) -> dict:
    """Reload a run saved by run_qrisp (give either the .npz or the .json path).

    Returns the same dict shape as run_qrisp, so it can go straight into
    decode_hardware_results() or summarize_hardware_results(). No token needed.
    """
    import json
    stem = path.rsplit(".", 1)[0]
    arrays = np.load(stem + ".npz")
    with open(stem + ".json") as f:
        meta = json.load(f)
    meta.pop("counts", None)
    return {"det_events": arrays["det_events"], "obs_flips": arrays["obs_flips"], **meta}



def decode_hardware_results(
    syndromes: dict,
    distance: int = None,
    rounds: int = None,
    noise = None,
    basis: str = None,
    decoder: str = "pymatching",
    track: str = None,
) -> tuple[float, float]:
    """
    Decode one run (either track) and return its logical error probability.

    Works with run_qiskit() output, run_qrisp()/load_hardware_results() output, and
    one round count of a run_experiment.py results file.

    Parameters
    ----------
    syndromes : dict with "det_events" (shots, detectors) and "obs_flips" (shots, 1)
    distance, rounds, basis : default to the values stored in `syndromes`
    noise     : the decoder's noise model:
                  None or "uniform"   0.3% everywhere
                  "uniform:0.005"     your own p
                  "calibration"       per-qubit, from syndromes["calibration"]
                  a dict of DEFAULT_NOISE keys (older style, as in decoders.py)
    decoder   : a name from decoding.DECODERS or "module:function"
    track     : "qiskit" or "qrisp"; guessed from `syndromes` when left out

    Returns
    -------
    p   : P_error, the logical error probability after `rounds` rounds
    err : 1-sigma statistical error, sqrt(p(1-p)/shots)
    """
    from tracks import TRACKS, qubit_map_from
    from noise_models import NoiseContext, add_noise
    from decoding import logical_error

    distance = distance or syndromes.get("distance", 3)
    rounds = rounds or syndromes["rounds"]
    basis = basis or syndromes.get("basis", "Z")
    track = track or syndromes.get("track") or ("qrisp" if "placement" in syndromes else "qiskit")

    if isinstance(noise, dict):          # DEFAULT_NOISE-style dict: each track's own generator
        if track == "qrisp":
            noisy = make_qrisp_experiment(distance, rounds, basis=basis, noise=noise).to_stim()
        else:
            noisy = make_stim_circuit(distance, rounds, basis=basis, noise=noise)
    else:
        ctx = NoiseContext(calibration=syndromes.get("calibration"),
                           qubit_map=qubit_map_from(syndromes),
                           dd=bool(syndromes.get("dd", False)))
        noisy = add_noise(TRACKS[track].twin(distance, rounds, basis), noise or "uniform", ctx)
    p, err, _ = logical_error(syndromes["det_events"], syndromes["obs_flips"], noisy, decoder)
    return p, err


def summarize_hardware_results(syndromes: dict, noise=None, decoder=None) -> dict:
    """The numbers to look at after every run.

    logical_error           : P_error after decoding (lower is better)
    logical_error_sigma     : its 1-sigma uncertainty
    undecoded_logical_error : how often the logical value flipped with NO
                              correction. If decoding does not beat this, the
                              decoder or the circuit is wrong.
    mean_detector_rate      : average fraction of shots in which an alarm fired.
                              ~0 = suspiciously quiet; ~0.5 = pure noise, the
                              code is overwhelmed.
    detector_rates          : the same per detector, in experiment order.

    decoder : optional replacement for decode_hardware_results, any function
              f(syndromes) -> (ler, err) (the decoders.py convention).
    """
    ler, err = decoder(syndromes) if decoder else decode_hardware_results(syndromes, noise=noise)
    det = np.asarray(syndromes["det_events"], dtype=bool)
    obs = np.asarray(syndromes["obs_flips"], dtype=bool)
    return {
        "shots": int(det.shape[0]),
        "logical_error": ler,
        "logical_error_sigma": err,
        "undecoded_logical_error": float(obs[:, 0].mean()),
        "mean_detector_rate": float(det.mean()),
        "detector_rates": det.mean(axis=0).round(3).tolist(),
    }


# Command line:
#   python run_on_hardware.py --rounds 1 --dry-run         validate, no credits
#   python run_on_hardware.py --rounds 3 --basis X         real run (default 2000 shots)
#   python run_on_hardware.py --load results/<run>.json    re-decode a saved run
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the Qrisp surface code memory on IQM hardware.")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--shots", type=int, default=2_000)
    parser.add_argument("--basis", choices=["Z", "X"], default="Z")
    parser.add_argument("--device", default="garnet")
    parser.add_argument("--dry-run", action="store_true", help="validate on the device without submitting")
    parser.add_argument("--load", help="decode a saved run instead of submitting")
    args = parser.parse_args()

    if args.load:
        result = load_hardware_results(args.load)
    else:
        result = run_qrisp(distance=3, rounds=args.rounds, shots=args.shots, basis=args.basis,
                           quantum_computer=args.device, dry_run=args.dry_run)
    if result["det_events"] is not None:
        print(json.dumps(summarize_hardware_results(result), indent=1))
