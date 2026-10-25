import stim
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
#  These may be useful for working with Stim, when converting to qiskit
#  and when converting measurement result back into readable files
# ─────────────────────────────────────────────────────────────────────────────

"""


"""


def get_qubit_lists(circuit: stim.Circuit) -> tuple[list, list]:
    """
    For Stim circuits: Returns (data_qubit_stim_indices, ancilla_qubit_stim_indices).

    NOTE: PLEASE DOUBLE CHECK FUNCTIONALITY!

    Supports both standard circuits (ancillas use MR) and no-reset circuits
    (ancillas use M). When QUBIT_COORDS annotations are present, distinguishes
    by coordinate parity:
        data qubits   → both x and y coordinates are odd
        ancilla qubits → at least one coordinate is even

    """
    coords = circuit.get_final_qubit_coordinates()

    # Collect all measured qubit indices
    all_measured: set[int] = set()
    for instr in circuit.flattened():
        if instr.name in ("M", "MR", "MX"):
            for t in instr.targets_copy():
                if t.is_qubit_target:
                    all_measured.add(t.value)

    data_q, anc_q = [], []
    for q in sorted(all_measured):
        if q in coords:
            x, y = coords[q]
            # data qubits sit at odd-x, odd-y grid positions
            if int(x) % 2 == 1 and int(y) % 2 == 1:
                data_q.append(q)
            else:
                anc_q.append(q)
        else:
            # No coordinates — conservative: treat as data
            data_q.append(q)
    return data_q, anc_q


    return data_q, anc_q


def get_meas_order(circuit: stim.Circuit) -> list:
    """Returns list of Stim qubit indices in measurement order."""
    order = []
    for instr in circuit.flattened():
        if instr.name in ("M", "MR", "MX"):
            for t in instr.targets_copy():
                if t.is_qubit_target:
                    order.append(t.value)
    return order



def counts_to_measurement_array(
    counts: dict,
    num_measurements: int,
    total_shots: int,
) -> np.ndarray:
    """
    QISKIT TRACK ONLY. On the Qrisp track, the DetectorExperiment returns
    detection events and observable flips directly.

    Converts a Qiskit counts dict to a (total_shots, num_measurements)
    boolean numpy array in Stim's measurement order.

    Qiskit bitstrings are RIGHT-TO-LEFT: rightmost character = clbit 0.
    Stim's measurement order corresponds to clbit 0, 1, 2, … in sequence.
    """
    result = np.zeros((total_shots, num_measurements), dtype=bool)
    row = 0
    for bitstring, count in counts.items():
        # Reverse so index 0 = clbit 0
        bits = bitstring[::-1]
        meas = np.array([int(b) for b in bits[:num_measurements]], dtype=bool)
        for _ in range(count):
            result[row] = meas
            row += 1
    return result