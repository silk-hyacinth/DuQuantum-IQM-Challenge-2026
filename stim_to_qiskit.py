import stim
from qiskit import QuantumCircuit

from internal_helpers import get_qubit_lists

# ─────────────────────────────────────────────────────────────────────────────
#  STIM → QISKIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

# Instructions that exist only in Stim's simulation model — skip them when
# building the Qiskit circuit for hardware.
_STIM_SKIP = frozenset({
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR",
    "ELSE_CORRELATED_ERROR", "DETECTOR", "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS", "SHIFT_COORDS", "TICK",
})


def stim_to_qiskit(stim_circuit: stim.Circuit) -> tuple[QuantumCircuit, dict, list]:
    """
    Converts a (noiseless) Stim circuit into a Qiskit QuantumCircuit.

    The Stim circuit is first flattened to unroll any REPEAT blocks,
    then noise/annotation instructions are dropped. The resulting
    Qiskit circuit is suitable for transpilation onto IQM Resonance.

    Parameters
    ----------
    stim_circuit : stim.Circuit  (use make_stim_circuit(..., noise=None))

    Returns
    -------
    qc            : QuantumCircuit  with dense qubit indices 0..N-1
    stim_to_dense : dict  stim_qubit_index → dense_qiskit_index
    meas_order    : list[int]  stim qubit indices in measurement order
                    Use this to map hardware bitstrings back to Stim's
                    measurement record for detection event conversion.
    """
    flat = stim_circuit.flattened()

    # Build dense re-indexing: Stim uses sparse indices (e.g. 1,3,5,8,…,25)
    # Qiskit needs contiguous 0..N-1 indices.
    data_q, anc_q = get_qubit_lists(stim_circuit)
    all_stim_q    = sorted(set(data_q + anc_q))
    stim_to_dense = {sq: i for i, sq in enumerate(all_stim_q)}

    n_qubits = len(all_stim_q)
    n_clbits = stim_circuit.num_measurements # one classical bit per measured qubit
    qc = QuantumCircuit(n_qubits, n_clbits)

    meas_idx  = 0
    meas_order = []

    for instr in flat:
        name = instr.name

        if name in _STIM_SKIP:
            continue

        # Extract qubit targets (ignore measurement-record targets in M gates)
        targets = [t.value for t in instr.targets_copy() if t.is_qubit_target]
        dense   = [stim_to_dense[t] for t in targets]

        if name == "R":
            # NOTE: if transpile() rejects 'reset' on Emerald, drop the first R per qubit
            # (qubits start in |0>), as stim_to_qrisp.py does.
            for d in dense:
                qc.reset(d)

        elif name == "H":
            for d in dense:
                qc.h(d)

        elif name == "CX":
            for i in range(0, len(dense), 2):
                qc.cx(dense[i], dense[i + 1])

        elif name == "CZ":
            for i in range(0, len(dense), 2):
                qc.cz(dense[i], dense[i + 1])

        elif name == "X":
            for d in dense:
                qc.x(d)

        elif name == "RX":
            # prepare |+>: reset to |0>, then H
            for d in dense:
                qc.reset(d)
                qc.h(d)

        elif name == "MX":
            # measure in the X basis: H, then an ordinary Z measurement
            for stim_q, d in zip(targets, dense):
                qc.h(d)
                qc.measure(d, meas_idx)
                meas_order.append(stim_q)
                meas_idx += 1

        elif name in ("M", "MR"):
            for stim_q, d in zip(targets, dense):
                qc.measure(d, meas_idx)
                meas_order.append(stim_q)
                meas_idx += 1
            # MR = measure-then-reset (mid-circuit reset for multi-round)
            # MR only appears in with-reset circuits, which Emerald does not support.
            if name == "MR":
                for d in dense:
                    qc.reset(d)

        else:
            # Warn about genuinely unhandled gates (not noise/annotations)
            raise NotImplementedError(
                f"[stim_to_qiskit] Unhandled gate '{name}'. "
                f"If this is a noise/annotation instruction, add it to _STIM_SKIP. "
                f"If it is a gate instruction, implement it explicitly."
                )

    return qc, stim_to_dense, meas_order