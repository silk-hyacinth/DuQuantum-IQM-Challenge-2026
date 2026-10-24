from surface_code import *
from stim_to_qiskit import stim_to_qiskit
from build_emerald_qubit_rotated import *
from extract_syndromes import *
from qiskit import transpile


import os
import numpy as np
import pymatching

TOKEN = os.environ.get("IQM_TOKEN", "")   # set the IQM_TOKEN environment variable, or paste your token here
SERVER_URL = "https://resonance.iqm.tech"

def get_backend(quantum_computer: str = "garnet", token: str = None):
    """Connect to IQM Resonance and return the backend for `quantum_computer`."""
    from iqm.qiskit_iqm import IQMProvider
    provider = IQMProvider(SERVER_URL, quantum_computer=quantum_computer, token=token or TOKEN)
    return provider.get_backend()


def get_calibration(backend) -> dict:
    """
    Snapshot of the device's current error rates, as a JSON-safe dict. Save it next to your
    results: calibration changes over the day, and a calibration-aware decoder needs the
    numbers from when the data was taken.

    Real hardware -> {"source": "resonance", "fidelities": {...}}
        fidelities["CZ"]["[QB1, QB2]"], ["1Q"]["QB1"], ["readout"]["QB1"], ...  are FIDELITIES
        (error = 1 - fidelity); ["t1"]["QB1"], ["t2"]["QB1"] are coherence times.
    Fake backend   -> {"source": "fake_error_profile", "error_profile": {...}}
        the noise model the fake backend simulates (error rates, T1/T2 in ns, readout errors).
    """
    from datetime import datetime
    out = {"retrieved_at": datetime.now().isoformat(timespec="seconds"),
           "backend": getattr(backend, "name", str(backend))}

    if hasattr(backend, "error_profile"):                      # IQM fake backend
        import dataclasses

        def clean(x):
            if isinstance(x, dict):
                return {("-".join(k) if isinstance(k, tuple) else str(k)): clean(v) for k, v in x.items()}
            return x
        out["source"] = "fake_error_profile"
        out["error_profile"] = clean(dataclasses.asdict(backend.error_profile))
        return out

    try:                                                        # real hardware
        from iqm.qubit_selector.qubit_selector import CalibrationDataManager
        out["source"] = "resonance"
        out["fidelities"] = CalibrationDataManager().get_calibration_fidelities(backend)
    except Exception as e:
        out["source"] = "unavailable"
        out["error"] = repr(e)
        print(f"[get_calibration] could not fetch calibration data: {e!r}")
    return out


def get_physical_layout(qc_t, stim_circ, backend) -> list:
    """
    Which physical qubit each code qubit ran on, e.g.
    [{"stim_qubit": 1, "role": "data", "coords": [1, 1], "physical": "QB7"}, ...]
    Needed to match calibration numbers to the right part of the code.
    """
    data_q, anc_q = get_qubit_lists(stim_circ)
    stim_qubits = sorted(set(data_q) | set(anc_q))     # same ordering stim_to_qiskit uses
    coords = stim_circ.get_final_qubit_coordinates()
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
) -> dict:
    """
    Runs the no-reset surface code via Qiskit and returns detection events.

        make_qiskit_circuit → transpile → backend.run → counts
        → raw measurements → extract_syndromes()

    Parameters
    ----------
    distance, rounds, shots, no_reset : experiment settings
    token            : IQM Resonance API token (falls back to TOKEN / IQM_TOKEN env var)
    quantum_computer : "garnet", "emerald", ...  (ignored if `backend` is given)
    backend          : optional ready-made backend, e.g. IQMFakeGarnet() for local testing
    initial_layout   : optional list of physical qubits (position i = physical qubit for circuit qubit i).
                       None -> the transpiler picks a SWAP-free placement itself.
    dd               : turn on IQM's dynamical decoupling (real hardware only)

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
    qc        = make_qiskit_circuit(distance, rounds, no_reset=no_reset)
    stim_circ = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    # 2. Transpile: rewrite into the device's native gates (r, cz) and assign physical qubits
    qc_t = transpile(qc, backend, optimization_level=3, seed_transpiler=1,
                     initial_layout=initial_layout)
    n_cx, n_cz = qc.count_ops().get("cx", 0), qc_t.count_ops().get("cz", 0)
    print(f"[run_qiskit] d={distance} rounds={rounds}: CX={n_cx} -> CZ={n_cz} "
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



def run_qrisp(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
) -> dict:
    """
    Build in qrisp. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qrisp DetectorExperiment → IQMBackend → run
        → detection events + observable flips

    The DetectorExperiment stores the info the detectors and observables itself, so
    counts_to_measurement_array() and extract_syndromes() are not needed here.

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token 
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  same keys as extract_syndromes() from qiskit function
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables)
    """
    import os
    from iqm.qrisp_iqm import IQMBackend
    from stim_to_qrisp import stim_to_detector_experiment

    os.environ["IQM_TOKEN"] = token or TOKEN

    # Build circuits, exchange for make_qrisp_experiment() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    # CREATE AN EXPERIMENT TO BE EXECUTED ON HARDWARE
    experiment = []
    #experiment = stim_to_detector_experiment(stim_circ, no_reset=no_reset)
    #experiment = make_qrisp_experiment(distance, rounds, no_reset=no_reset)

    # Connect to IQM Resonance
    # Layout: think about the pass_manager. iqm.qrisp_iqm.vf2pp_layout finds an
    # exact embedding on the coupling map (zero SWAPs) or fails.
    backend    = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
    # further arguments
    )

    # Run the experiment on the backend and collect detection events + observable flips
    # TO BE DONE
    det_events, obs_flips = [], []

    return {"det_events": det_events, "obs_flips": obs_flips}



def decode_hardware_results(
    syndromes: dict,
    distance: int,
    rounds: int,
    noise: Optional[dict] = None,
) -> tuple[float, float]:
    """
    Decodes the output of run_qiskit() (or run_qrisp()) with PyMatching.

    The decoder needs a noise model to know which errors are likely. It is built from a
    NOISY Stim copy of the same no-reset circuit that ran, so the detectors match.
    noise=None -> DEFAULT_NOISE (uniform guess). Better: values from calibration data.

    Returns
    -------
    p   : float  logical error probability for this round count
    err : float  1-sigma statistical error, sqrt(p(1-p)/shots)
    """
    noisy   = make_stim_circuit(distance, rounds, no_reset=True, noise=noise or DEFAULT_NOISE)
    dem     = noisy.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    det_events = syndromes["det_events"]
    obs_flips  = syndromes["obs_flips"]
    pred = np.asarray(matcher.decode_batch(det_events))

    p   = float(np.mean(pred[:, 0] != obs_flips[:, 0]))
    err = float(np.sqrt(p * (1 - p) / len(det_events)))
    return p, err
