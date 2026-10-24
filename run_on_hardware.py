from surface_code import *
from stim_to_qiskit import stim_to_qiskit
from build_emerald_qubit_rotated import *
from extract_syndromes import *
from qiskit import transpile


# Leave TOKEN empty in git. run_qrisp() reads IQM_TOKEN from the local, gitignored
# .env file instead, so the token never ends up in the repository.
TOKEN = ""
SERVER_URL = "https://resonance.iqm.tech"

# ─────────────────────────────────────────────────────────────────────────────
#  5.  HARDWARE EXECUTION ON IQM RESONANCE
#
#  Two equal tracks. Pick ONE. Both return the same syndromes dict
#  (at least "det_events" and "obs_flips"), so decode_hardware_results()
#  works for either.
#
#  NOTE: Emerald does not support mid-circuit reset -> use the no-reset circuit.
#
#  DuQuantum uses the QRISP track (run_qrisp below) on IQM Garnet.
#  run_qiskit is the untouched template from main and is not used by us.
# ─────────────────────────────────────────────────────────────────────────────

def run_qiskit(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
) -> dict:
    """
    Build in qiskit. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qiskit circuit → transpile → run → counts
        → raw measurements → extract_syndromes()

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token (falls back to TOKEN)
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  output of extract_syndromes()
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables), ...
    """
    from iqm.qiskit_iqm import IQMProvider

    # Build circuits, exchange for make_qiskit_circuit() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    #CREATE A CIRCUIT TO BE EXECUTED ON HARDWARE
    qc = []
    #qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)
    #qc = make_qiskit_circuit(args)


    # OPTIONAL: IF YOU WANT TO ASSIGN QUBIT MAP BY HAND
    # Map abstract Qiskit qubits to emerald physical qubits
    qubit_map  = build_emerald_qubit_map(stim_circ)

    # HIGHLY RECOMMEND PRINTING THIS, if you are using it.
    # print(qubit_map)
    # initial_layout: Qiskit needs a list where position i = physical qubit for logical i
    n_qubits   = qc.num_qubits
    layout     = [qubit_map.get(i, i) for i in range(n_qubits)]

    # Connect to IQM Resonance
    provider   = IQMProvider(
        SERVER_URL,
        quantum_computer=quantum_computer,
        token=token or TOKEN
    )
    backend    = provider.get_backend()

    # Transpile: Qiskit transpilerwill insert SWAPs for diagonal CX pairs automatically
    # think about whether this is necessary
    qc_t       = transpile(
    # arguments
    )
    print(f"Transpiled circuit depth: {qc_t.depth()}  "
          f"(ideal: {qc.depth()}, extra depth from SWAPs)")

    # Run
    job        = backend.run(qc_t, shots=shots)
    counts     = job.result().get_counts()

    # Convert counts dict → (shots, num_measurements) numpy array
    raw_meas   = counts_to_measurement_array(counts, stim_circ.num_measurements,
                                               shots)
    # m2d converter must come from the circuit that actually ran
    return extract_syndromes(raw_meas, stim_circ)



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
    Run the surface code memory experiment on real IQM hardware (Qrisp track).

    What happens, step by step
    --------------------------
      1. Build the circuit with make_qrisp_experiment (surface_code.py).
      2. Read today's calibration and pick the best zero-SWAP placement
         (garnet_layout.py). On Garnet there is only one possible qubit set.
      3. Compile: pin the circuit onto those physical qubits (and prove every
         CZ is on a real coupler), then turn each SQRT_Y gate into one PRX pulse.
      4. Submit to IQM Resonance and wait. The hardware returns "counts": for
         each 17-bit (or longer) string of measurement results, how many shots
         produced it.
      5. Convert each bitstring into detection events and the observable flip,
         using the post-processing Qrisp generated from our parity() calls.
      6. Save everything to results/ so the run can be re-analysed later.

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
    compilation_options : iqm.iqm_client.CircuitCompilationOptions, e.g. to turn
                          on dynamical decoupling or heralding. None = IQM defaults.
    save_dir            : where to save the run (None = don't save)
    dry_run             : do steps 1-3 and print the compiled gate counts,
                          but do not submit (costs nothing)

    Output
    ------
    dict with
      det_events : bool array (shots, n_detectors). True = that alarm fired.
      obs_flips  : bool array (shots, 1). True = the logical value came out
                   flipped before any correction.
      plus metadata (job_id, calibration_set_id, placement, rounds, basis, ...)
      that decode_hardware_results() needs.
    """
    import json
    import os
    import time
    from dotenv import load_dotenv
    from iqm.qrisp_iqm import IQMBackend
    from iqm.qrisp_iqm.iqm_converter import qrisp_to_iqm_converter
    from jax import jit as jax_jit
    from qrisp import PassManager, convert_to_prx
    from qrisp.jasp import make_jaspr
    from garnet_layout import fetch_calibration, fixed_layout, rank_placements

    # Credentials: .env holds IQM_TOKEN, IQM_SERVER_URL, IQM_QUANTUM_COMPUTER.
    load_dotenv()
    if token or TOKEN:
        os.environ["IQM_TOKEN"] = token or TOKEN
    if not os.environ.get("IQM_TOKEN"):
        raise RuntimeError("no IQM token: set IQM_TOKEN in .env or pass token=")
    if shots > 20_000:
        raise ValueError("Resonance allows at most 20 000 shots per circuit")

    # Step 1. Build the experiment WITHOUT noise annotations (hardware supplies
    # the real noise). Tracing it ("make_jaspr") gives:
    #   qc              : the plain quantum circuit to send to the chip
    #   post_processing : a compiled function bits -> (detectors..., observable)
    experiment = make_qrisp_experiment(distance, rounds, basis=basis, no_reset=no_reset,
                                       idle_time=idle_time)
    jaspr = make_jaspr(experiment._func)()
    qc = jaspr.to_qc()[-1]
    post_processing = jax_jit(jaspr.extract_post_processing())

    # Step 2. Calibration-ranked placement (index 0 = lowest expected faults).
    cal = fetch_calibration(quantum_computer=quantum_computer)
    best = rank_placements(experiment.layout, cal)[0]

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

    # Compile once locally just to count what will run (and catch errors early).
    native = backend.pm.run(qc)
    iqm_circuit = qrisp_to_iqm_converter(native, backend._dqa)
    op_counts = {}
    for op in iqm_circuit.instructions:
        op_counts[op.name] = op_counts.get(op.name, 0) + 1

    # Everything needed to reproduce / re-decode this run later. The per-qubit
    # calibration of the qubits we used is saved too, so drift between runs
    # can be compared.
    used = set(best.qubits)
    meta = {
        "quantum_computer": quantum_computer,
        "calibration_set_id": str(backend._calibration_set_id),
        "distance": distance, "rounds": rounds, "basis": basis,
        "no_reset": no_reset, "idle_time": idle_time, "shots": shots,
        "placement": {str(i): q for i, q in sorted(best.placement.items())},
        "budget_per_round": best.budget,
        "native_ops": op_counts,
        "calibration": {
            key: {q: v for q, v in cal[key].items() if q in used}
            for key in ("prx", "readout", "e01", "e10", "leakage", "t1", "t2", "t2_echo")
        } | {"cz": {"__".join(sorted(p)): e for p, e in cal["cz"].items() if p <= used}},
    }
    print(f"placement on {quantum_computer}: {best.qubits}")
    print(f"native ops: {op_counts}  (no swap/move ops: every CZ is a native coupler)")
    if dry_run:
        return {"det_events": None, "obs_flips": None, **meta}

    # Step 4. Submit and wait. counts = {"0110...": how many shots gave this}.
    job = backend.run_async(qc, shots)
    print(f"submitted job {job.job_id}; waiting for results ...")
    counts = job.result().all_counts[0]
    meta["job_id"] = str(job.job_id)

    # Step 5. Each distinct bitstring is post-processed once, then repeated
    # `count` times, giving one row per shot. The last output of
    # post_processing is the observable; everything before it is a detector,
    # in the same order make_qrisp_experiment returned them.
    det_rows, obs_rows, weights = [], [], []
    for bitstring, count in counts.items():
        out = np.asarray(post_processing(np.array([c == "1" for c in bitstring])), dtype=bool)
        det_rows.append(out[:-1])
        obs_rows.append(out[-1:])
        weights.append(count)
    det_events = np.repeat(np.array(det_rows), weights, axis=0)
    obs_flips = np.repeat(np.array(obs_rows), weights, axis=0)
    meta["shots_returned"] = int(det_events.shape[0])

    # Step 6. Save: .npz = the arrays, .json = metadata + raw counts. The raw
    # counts are identical to what the Resonance dashboard shows for the job.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = (f"{time.strftime('%Y%m%d-%H%M%S')}_{quantum_computer}_d{distance}"
                f"_r{rounds}_{basis}_{meta['job_id'][:8]}")
        np.savez_compressed(os.path.join(save_dir, stem + ".npz"),
                            det_events=det_events, obs_flips=obs_flips)
        with open(os.path.join(save_dir, stem + ".json"), "w") as f:
            json.dump({**meta, "counts": counts}, f, indent=1)
        print(f"saved {save_dir}/{stem}.npz and .json")

    return {"det_events": det_events, "obs_flips": obs_flips, **meta}


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
    noise = None,
) -> tuple[float, float]:
    """
    Decode a run and return its logical error probability.

    How it works
    ------------
    The decoder (PyMatching, "minimum-weight perfect matching") looks at which
    alarms (detectors) fired in a shot and finds the most likely set of
    physical errors that explains them. From that it predicts whether the
    logical value got flipped. If the prediction disagrees with what actually
    happened (obs_flips), the shot is a logical error.

    To know "how likely" each error is, the decoder needs a detector error model
    (DEM). We get it by rebuilding the SAME circuit with noise annotations
    (`noise`) and asking Stim for its DEM. Because it is the same generator, the
    detectors are numbered identically to the hardware data.

    Inputs
    ------
    syndromes : output of run_qrisp() or load_hardware_results()
    noise     : noise model used only for decoder weights. None -> DEFAULT_NOISE
                (0.3% everywhere). A float sets every channel; a dict sets them
                individually. Better weights (from calibration) -> fewer
                decoding mistakes.

    Outputs
    -------
    ler : logical error probability after `rounds` rounds, P_error.
          (Not yet the per-round rate. Do several round counts and fit
          2 P_error = 1 - (1 - 2 eps_L)^rounds, see README.)
    err : 1-sigma statistical uncertainty, sqrt(P (1-P) / shots).
    """
    twin = make_qrisp_experiment(
        syndromes["distance"], syndromes["rounds"], basis=syndromes["basis"],
        no_reset=syndromes["no_reset"], noise=DEFAULT_NOISE if noise is None else noise,
    ).to_stim()
    det_events = np.asarray(syndromes["det_events"], dtype=bool)
    obs_flips = np.asarray(syndromes["obs_flips"], dtype=bool)
    if det_events.shape[1] != twin.num_detectors:
        raise ValueError(f"{det_events.shape[1]} detectors in data, {twin.num_detectors} in the DEM")

    matcher = pymatching.Matching.from_detector_error_model(
        twin.detector_error_model(decompose_errors=True))
    predictions = matcher.decode_batch(det_events)        # (shots, 1) predicted flips
    failures = np.any(predictions != obs_flips, axis=1)   # True = decoder got it wrong
    ler = float(failures.mean())
    err = float(np.sqrt(ler * (1 - ler) / len(failures)))
    return ler, err


def summarize_hardware_results(syndromes: dict, noise=None) -> dict:
    """The numbers to look at after every run.

    logical_error           : P_error after decoding (lower is better)
    logical_error_sigma     : its 1-sigma uncertainty
    undecoded_logical_error : how often the logical value flipped with NO
                              correction. If decoding does not beat this, the
                              decoder or the circuit is wrong.
    mean_detector_rate      : average fraction of shots in which an alarm fired.
                              ~0 = suspiciously quiet; ~0.5 = pure noise, the
                              code is overwhelmed.
    detector_rates          : the same per detector, in experiment order. Map a
                              high one back to its qubits to find the culprit.
    """
    ler, err = decode_hardware_results(syndromes, noise)
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
