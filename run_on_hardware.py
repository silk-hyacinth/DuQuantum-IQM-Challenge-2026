from surface_code import *
from stim_to_qiskit import stim_to_qiskit
from build_emerald_qubit_rotated import *
from extract_syndromes import *
from qiskit import transpile


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
    Build in qrisp. Runs the surface code on IQM Garnet via IQM Resonance.

        make_qrisp_experiment → pin to the best zero-SWAP Garnet placement
        → PRX lowering → IQMBackend → counts → detection events + observable flips

    The DetectorExperiment stores the info the detectors and observables itself, so
    counts_to_measurement_array() and extract_syndromes() are not needed here.

    Credentials come from ``token`` or IQM_TOKEN (loaded from .env). Every
    hardware run is saved to ``save_dir`` as <stem>.npz (detection events,
    observable flips) and <stem>.json (job id, calibration set, placement,
    raw counts) so results survive the session.

    Parameters
    ----------
    distance            : code distance (3 is the only one that fits Garnet)
    rounds              : measurement rounds
    shots               : number of circuit executions (at most 20 000)
    token               : IQM Resonance API token
    quantum_computer    : e.g. "garnet"
    no_reset            : build the no-reset circuit (required on hardware)
    basis               : "Z" or "X" memory
    idle_time           : extra ancilla delay between rounds in seconds; nonzero
                          values need pulse-level access
    compilation_options : iqm.iqm_client.CircuitCompilationOptions or None
    save_dir            : directory for the saved run, or None to skip saving
    dry_run             : transpile and validate against the device, but do not submit

    Returns
    -------
    syndromes : dict  same keys as extract_syndromes() from qiskit function
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables),
                plus the run metadata needed to decode it
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

    load_dotenv()
    if token or TOKEN:
        os.environ["IQM_TOKEN"] = token or TOKEN
    if not os.environ.get("IQM_TOKEN"):
        raise RuntimeError("no IQM token: set IQM_TOKEN in .env or pass token=")
    if shots > 20_000:
        raise ValueError("Resonance allows at most 20 000 shots per circuit")

    experiment = make_qrisp_experiment(distance, rounds, basis=basis, no_reset=no_reset,
                                       idle_time=idle_time)
    jaspr = make_jaspr(experiment._func)()
    qc = jaspr.to_qc()[-1]
    post_processing = jax_jit(jaspr.extract_post_processing())

    cal = fetch_calibration(quantum_computer)
    best = rank_placements(experiment.layout, cal)[0]

    backend = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
        pass_manager=PassManager(),
        compilation_options=compilation_options,
    )
    backend.pm += fixed_layout(best.placement, backend._dqa.qubits, backend.connectivity)
    backend.pm += convert_to_prx

    native = backend.pm.run(qc)
    iqm_circuit = qrisp_to_iqm_converter(native, backend._dqa)
    op_counts = {}
    for op in iqm_circuit.instructions:
        op_counts[op.name] = op_counts.get(op.name, 0) + 1

    meta = {
        "quantum_computer": quantum_computer,
        "calibration_set_id": str(backend._calibration_set_id),
        "distance": distance, "rounds": rounds, "basis": basis,
        "no_reset": no_reset, "idle_time": idle_time, "shots": shots,
        "placement": {str(i): q for i, q in sorted(best.placement.items())},
        "budget_per_round": best.budget,
        "native_ops": op_counts,
    }
    print(f"placement on {quantum_computer}: {best.qubits}")
    print(f"native ops: {op_counts}  (no swap/move ops: every CZ is a native coupler)")
    if dry_run:
        return {"det_events": None, "obs_flips": None, **meta}

    job = backend.run_async(qc, shots)
    print(f"submitted job {job.job_id}; waiting for results ...")
    counts = job.result().all_counts[0]
    meta["job_id"] = str(job.job_id)

    det_rows, obs_rows, weights = [], [], []
    for bitstring, count in counts.items():
        out = np.asarray(post_processing(np.array([c == "1" for c in bitstring])), dtype=bool)
        det_rows.append(out[:-1])
        obs_rows.append(out[-1:])
        weights.append(count)
    det_events = np.repeat(np.array(det_rows), weights, axis=0)
    obs_flips = np.repeat(np.array(obs_rows), weights, axis=0)
    meta["shots_returned"] = int(det_events.shape[0])

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
    """Reload a run saved by run_qrisp (pass either the .npz or the .json path)."""
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
    Decodes the output of run_qiskit() or run_qrisp() via your desired decoder.

    NOTE: Other parameters can be important for this function, such as a noise model.
    If you decode with a detector error model (DEM): the DEM’s detectors must match the circuit that ran.


    Parameters
    ----------
    syndromes           : dict  output of run_qrisp() or load_hardware_results()
    noise               : noise model for the matching weights (None -> DEFAULT_NOISE).
                          The DEM comes from the same generator with this noise added,
                          so its detectors match the circuit that ran.

    Returns
    -------
    ler : float   estimated logical error probability for this round count (fit over several round counts to get the logical error rate, see README)
    err : float   OPTIONAL 1-sigma statistical error
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
    predictions = matcher.decode_batch(det_events)
    failures = np.any(predictions != obs_flips, axis=1)
    ler = float(failures.mean())
    err = float(np.sqrt(ler * (1 - ler) / len(failures)))
    return ler, err


def summarize_hardware_results(syndromes: dict, noise=None) -> dict:
    """Decoded and undecoded logical error, plus how often detectors fire."""
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