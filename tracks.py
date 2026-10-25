"""
tracks.py — the two ways to build and run the circuit (--track qiskit | qrisp).

Both tracks run the SAME code: a d=3 rotated surface code memory experiment without
mid-circuit reset, on the same 17 qubits of Garnet. They differ in how the circuit is
written and compiled:

  qiskit   Stim's standard circuit (CX + H), converted to Qiskit; IQM's Qiskit
           transpiler turns it into native gates and places it (surface_code.make_stim_circuit,
           make_qiskit_circuit).
  qrisp    The Qrisp branch's generator: native CZ + one-pulse SQRT_Y gates with frame
           tracking, a barrier after every round's readout, and a calibration-ranked
           placement pinned by hand (surface_code.make_qrisp_experiment, garnet_layout.py).

Everything after the circuit is shared (run_experiment.py): noise model, decoder,
fit, saving, plots. A track only has to provide:

  twin(distance, rounds, basis)          noiseless Stim circuit with the DETECTOR rules,
                                         numbered like the track's detection events
  detector_info(distance, rounds, basis) per detector: round, check type (X/Z), position
  run(distance, rounds_list, basis, shots, quantum_computer, fake, dd, herald)
                                         all round counts as ONE job; per round count a dict:
                                           rounds, counts, det_events, obs_flips,
                                           qubit_map (twin qubit -> "QB7"), job_id,
                                           native_ops, swap_free

All round counts go in one job, so they run back to back on the same calibration.
"""

import functools

import numpy as np
from qiskit import QuantumCircuit, transpile

from surface_code import make_qiskit_circuit, make_qrisp_experiment, make_stim_circuit, \
    rotated_surface_code_layout


def _fake_backend():
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    return IQMFakeGarnet()


def _swap_free_layout(qc, backend):
    """Find chip qubits for every circuit qubit so each CX lands on a real coupler.

    Qiskit's own search (VF2Layout) gives up on big codes (d=5 on Emerald), so this
    searches directly: the code's interaction graph as a subgraph of the chip's
    coupling graph. Returns initial_layout (list: circuit qubit i -> chip qubit),
    or raises if the code does not fit at all.
    """
    import networkx as nx
    from networkx.algorithms import isomorphism

    code = nx.Graph()
    code.add_nodes_from(range(qc.num_qubits))
    for inst in qc.data:
        if len(inst.qubits) == 2:
            code.add_edge(*(qc.find_bit(q).index for q in inst.qubits))
    chip = nx.Graph(list(backend.coupling_map.get_edges()))
    match = next(isomorphism.GraphMatcher(chip, code).subgraph_monomorphisms_iter(), None)
    if match is None:
        raise RuntimeError(f"the code ({qc.num_qubits} qubits) has no SWAP-free placement on this chip")
    inverse = {c: p for p, c in match.items()}
    return [inverse[i] for i in range(qc.num_qubits)]


def compilation_options(dd: bool = False, herald: bool = False):
    """IQM server-side options for a real-hardware job (both tracks use the same ones).

    max_circuit_duration_over_t2=0.0 turns off the server's "circuit longer than T2"
    rejection, which would otherwise refuse long sweeps (6-8 rounds). It does not change
    how the circuit runs.
    dd     : dynamical decoupling during idle times
    herald : measure every qubit before the circuit and drop shots that did not start in 0
    """
    from iqm.iqm_client import CircuitCompilationOptions, DDMode, HeraldingMode
    return CircuitCompilationOptions(
        max_circuit_duration_over_t2=0.0,
        dd_mode=DDMode.ENABLED if dd else DDMode.DISABLED,
        heralding_mode=HeraldingMode.ZEROS if herald else HeraldingMode.NONE,
    )


def _job_id(job) -> str:
    try:
        return str(job.job_id())
    except TypeError:
        return str(job.job_id)


def qubit_map_from(record: dict) -> dict:
    """twin qubit index -> physical qubit name, from any saved run (either track)."""
    if record.get("qubit_map"):
        return {int(k): v for k, v in record["qubit_map"].items()}
    if record.get("placement"):                              # Qrisp branch run files
        return {int(k): v for k, v in record["placement"].items()}
    if record.get("layout"):                                 # earlier Qiskit results files
        return {int(l["stim_qubit"]): l["physical"] for l in record["layout"]}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Qiskit track
# ─────────────────────────────────────────────────────────────────────────────

class QiskitTrack:
    name = "qiskit"

    def twin(self, distance, rounds, basis="Z"):
        return make_stim_circuit(distance, rounds, basis=basis, no_reset=True, noise=None)

    def detector_info(self, distance, rounds, basis="Z"):
        twin = self.twin(distance, rounds, basis)
        qcoords = {tuple(map(float, v)): k for k, v in twin.get_final_qubit_coordinates().items()}
        first_h = next(i for i in twin.flattened() if i.name == "H")
        x_anc = {t.value for t in first_h.targets_copy()}
        info = []
        for k, (x, y, t) in sorted(twin.get_detector_coordinates().items()):
            anc = qcoords[(x, y)]
            info.append({"round": int(t), "type": "X" if anc in x_anc else "Z", "ancilla": anc, "pos": (x, y)})
        return info

    def events_from_counts(self, distance, rounds, basis, counts):
        """Saved counts -> (det_events, obs_flips), e.g. to re-analyse a results file."""
        from internal_helpers import counts_to_measurement_array
        twin = self.twin(distance, rounds, basis)
        raw = counts_to_measurement_array(counts, twin.num_measurements, sum(counts.values()))
        return twin.compile_m2d_converter().convert(measurements=raw.astype(bool), separate_observables=True)

    def run(self, distance, rounds_list, basis, shots, quantum_computer="garnet", fake=False, dd=False,
            herald=False, allow_swaps=False):
        """allow_swaps: only if NO SWAP-free placement exists, let the transpiler add SWAPs
        (Sabre routing). Each SWAP is 3 extra CZs; the extra count is printed and saved."""
        from run_on_hardware import get_backend, get_physical_layout
        from internal_helpers import counts_to_measurement_array
        from extract_syndromes import extract_syndromes

        from qiskit.transpiler.exceptions import TranspilerError

        backend = _fake_backend() if fake else get_backend(quantum_computer)
        prepared = []
        placement = None    # found once (if Qiskit's own search fails), reused for every round count
        for r in rounds_list:
            qc = make_qiskit_circuit(distance, r, basis=basis)
            twin = self.twin(distance, r, basis)
            # routing_method="none": the transpiler may NOT add SWAPs; it raises instead.
            try:
                if placement is not None:
                    raise TranspilerError("use the placement found earlier")
                qc_t = transpile(qc, backend, optimization_level=3, seed_transpiler=1, routing_method="none")
            except TranspilerError:
                if placement is None:
                    try:
                        placement = _swap_free_layout(qc, backend)
                        print(f"[qiskit] Qiskit found no SWAP-free placement; found one directly "
                              f"({len(placement)} chip qubits)")
                    except RuntimeError:
                        if not allow_swaps:
                            raise RuntimeError(
                                f"d={distance} has no SWAP-free placement on {quantum_computer} right now; "
                                f"rerun with --allow-swaps to let the transpiler add SWAPs") from None
                        placement = "swaps"
                        print(f"[qiskit] no SWAP-free placement on this chip; --allow-swaps: "
                              f"letting the transpiler add SWAPs")
                if placement == "swaps":
                    qc_t = transpile(qc, backend, optimization_level=3, seed_transpiler=1)
                else:
                    qc_t = transpile(qc, backend, optimization_level=3, seed_transpiler=1,
                                     routing_method="none", initial_layout=placement)
            n_cx, n_cz = qc.count_ops().get("cx", 0), qc_t.count_ops().get("cz", 0)
            layout = get_physical_layout(qc_t, twin, backend)
            prepared.append((r, qc_t, twin, layout, n_cx == n_cz, n_cz - n_cx))
            extra = f" (+{n_cz - n_cx} CZ from SWAPs, about {(n_cz - n_cx) / 3:.0f} SWAPs)" if n_cz > n_cx else ""
            print(f"[qiskit] d={distance} {basis} rounds={r}: CX={n_cx} -> CZ={n_cz}{extra}, depth {qc_t.depth()}")

        options = {} if fake else {"circuit_compilation_options": compilation_options(dd, herald)}
        job = backend.run([p[1] for p in prepared], shots=shots, **options)
        print(f"[qiskit] submitted job {_job_id(job)} ({len(prepared)} circuits x {shots} shots); waiting ...")
        result = job.result()

        out = []
        for i, (r, qc_t, twin, layout, swap_free, extra_cz) in enumerate(prepared):
            counts = result.get_counts(i)
            raw = counts_to_measurement_array(counts, twin.num_measurements, sum(counts.values()))
            s = extract_syndromes(raw, twin, print_summary=False)
            out.append({
                "rounds": r, "counts": counts,
                "det_events": s["det_events"], "obs_flips": s["obs_flips"],
                "qubit_map": {l["stim_qubit"]: l["physical"] for l in layout},
                "layout": layout, "job_id": _job_id(job),
                "native_ops": {k: int(v) for k, v in qc_t.count_ops().items()},
                "depth": int(qc_t.depth()), "swap_free": bool(swap_free), "extra_cz": int(extra_cz),
            })
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Qrisp track
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=64)
def _qrisp_twin(distance, rounds, basis):
    # Building a Qrisp experiment takes a few seconds; the twin never changes, so cache it.
    return make_qrisp_experiment(distance, rounds, basis=basis, no_reset=True, noise=None).to_stim()


def qrisp_to_qiskit(qc) -> QuantumCircuit:
    """A traced Qrisp circuit -> Qiskit, gate for gate (SQRT_Y = RY(+pi/2), SQRT_Y_DAG = RY(-pi/2)).
    Classical bit i of the result = qc.clbits[i]."""
    qubit_index = {qb: i for i, qb in enumerate(qc.qubits)}
    clbit_index = {cb: i for i, cb in enumerate(qc.clbits)}
    out = QuantumCircuit(len(qc.qubits), len(qc.clbits))
    for instr in qc.data:
        name = instr.op.name
        qs = [qubit_index[q] for q in instr.qubits]
        if name == "sy":
            out.ry(np.pi / 2, qs[0])
        elif name == "sy_dg":
            out.ry(-np.pi / 2, qs[0])
        elif name == "cz":
            out.cz(*qs)
        elif name == "measure":
            out.measure(qs[0], clbit_index[instr.clbits[0]])
        elif name == "barrier":
            out.barrier(*qs)
        elif name in ("parity", "qb_alloc", "qb_free"):
            continue                                   # bookkeeping only: allocation, parity in post-processing
        else:
            raise NotImplementedError(f"qrisp_to_qiskit: no conversion for '{name}'")
    return out


class QrispTrack:
    name = "qrisp"

    def twin(self, distance, rounds, basis="Z"):
        return _qrisp_twin(distance, rounds, basis).copy()

    def detector_info(self, distance, rounds, basis="Z"):
        # Same order as make_qrisp_experiment appends its parity() detectors.
        layout = rotated_surface_code_layout(distance)
        x_set = set(layout.x_ancillas)
        memory = layout.z_ancillas if basis == "Z" else layout.x_ancillas
        pos = dict(enumerate(layout.coords))

        def entry(k, a):
            return {"round": k, "type": "X" if a in x_set else "Z", "ancilla": a, "pos": tuple(pos[a])}
        info = []
        for k in range(rounds):
            for a in layout.ancillas:
                if k > 0 or a in memory:
                    info.append(entry(k, a))
        info += [entry(rounds, a) for a in memory]
        return info

    def _prepare(self, distance, r, basis):
        import jax
        from qrisp.jasp import make_jaspr
        exp = make_qrisp_experiment(distance, r, basis=basis, no_reset=True)
        jaspr = make_jaspr(exp._func)()
        return exp, jaspr.to_qc()[-1], jax.jit(jaspr.extract_post_processing())

    @staticmethod
    def _events(post_processing, counts, reverse):
        det_rows, obs_rows, weights = [], [], []
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            bits = bits[::-1] if reverse else bits
            vals = np.asarray(post_processing(np.array([c == "1" for c in bits])), dtype=bool)
            det_rows.append(vals[:-1])
            obs_rows.append(vals[-1:])
            weights.append(count)
        return (np.repeat(np.array(det_rows), weights, axis=0),
                np.repeat(np.array(obs_rows), weights, axis=0))

    def events_from_counts(self, distance, rounds, basis, counts):
        """Saved counts -> (det_events, obs_flips), e.g. to re-analyse a results file."""
        _, _, post = self._prepare(distance, rounds, basis)
        return self._events(post, counts, reverse=False)

    def run(self, distance, rounds_list, basis, shots, quantum_computer="garnet", fake=False, dd=False,
            herald=False):
        if fake:
            return self._run_fake(distance, rounds_list, basis, shots)

        # Real hardware: the Qrisp branch's runner (calibration-ranked placement, pinned
        # with fixed_layout, every CZ checked against the couplers, one job for all).
        import os
        from run_on_hardware import SERVER_URL, run_qrisp_batch
        os.environ.setdefault("IQM_SERVER_URL", SERVER_URL)
        options = compilation_options(dd, herald)
        configs = [{"distance": distance, "rounds": r, "basis": basis} for r in rounds_list]
        results = run_qrisp_batch(configs, shots=shots, quantum_computer=quantum_computer,
                                  compilation_options=options, save_dir=None)
        return [{
            "rounds": res["rounds"], "counts": res["counts"],
            "det_events": res["det_events"], "obs_flips": res["obs_flips"],
            "qubit_map": {int(k): v for k, v in res["placement"].items()},
            "job_id": res["job_id"], "native_ops": res["native_ops"],
            "swap_free": True,                     # fixed_layout raises on any non-native CZ
            "budget_per_round": res.get("budget_per_round"),
        } for res in results]

    def _run_fake(self, distance, rounds_list, basis, shots):
        # Local test: same circuit and placement logic, run on IQM's fake Garnet via Qiskit.
        from garnet_layout import rank_placements
        from noise_models import calibration_from_fake

        backend = _fake_backend()
        cal = calibration_from_fake(backend)
        prepared = []
        for r in rounds_list:
            exp, qc, post = self._prepare(distance, r, basis)
            placement = rank_placements(exp.layout, cal)[0].placement
            qk = qrisp_to_qiskit(qc)
            initial = [backend.qubit_name_to_index(placement[i]) for i in range(qk.num_qubits)]
            qc_t = transpile(qk, backend, initial_layout=initial, routing_method="none",
                             optimization_level=1, seed_transpiler=1)
            n_cz_in, n_cz = qk.count_ops().get("cz", 0), qc_t.count_ops().get("cz", 0)
            prepared.append((r, qc_t, post, placement, n_cz_in == n_cz))
            print(f"[qrisp] d={distance} {basis} rounds={r}: CZ={n_cz_in} -> {n_cz}, depth {qc_t.depth()}")

        job = backend.run([p[1] for p in prepared], shots=shots)
        result = job.result()
        out = []
        for i, (r, qc_t, post, placement, swap_free) in enumerate(prepared):
            counts = result.get_counts(i)
            # Qiskit's count keys already come in the order Qrisp's post-processing expects
            # (checked: a noiseless run gives 0 detections this way, ~all firing reversed).
            det, obs = self._events(post, counts, reverse=False)
            out.append({
                "rounds": r, "counts": counts, "det_events": det, "obs_flips": obs,
                "qubit_map": dict(placement), "job_id": _job_id(job),
                "native_ops": {k: int(v) for k, v in qc_t.count_ops().items()},
                "depth": int(qc_t.depth()), "swap_free": bool(swap_free),
            })
        return out


TRACKS = {"qiskit": QiskitTrack(), "qrisp": QrispTrack()}
