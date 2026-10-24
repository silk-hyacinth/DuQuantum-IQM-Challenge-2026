# -*- coding: utf-8 -*-

from internal_helpers import *

"""
surface_code.py

Circuit generation for a rotated surface code memory experiment (Z or X basis)
on IQM Emerald. By default distance 3: 9 data + 8 ancilla = 17 qubits.

Pipeline via Qrisp OR Qiskit
------------------------------------
  Qrisp  : circuit → Qrisp DetectorExperiment → IQM Emerald
           → detection events → decoder
  Qiskit : circuit → Qiskit circuit → IQM Emerald → raw measurements
           → extract_syndromes() → decoder


Generators (this file): pick ONE starting point
-----------------------------------------------
  make_stim_circuit()      For both Qrisp and Qiskit. Use one of the converters (VERIFY their funcionality FIRST!) to run on hardware, as Stim circuits dont work.
                           (stim_to_qiskit.py / stim_to_qrisp.py)
  make_qiskit_circuit()    Qiskit-native alternative
  make_qrisp_experiment()  Qrisp-native alternative

Where the other stages live
---------------------------
  1. Circuit generation              this file
  2. Stim → Qiskit / Qrisp           stim_to_qiskit.py, stim_to_qrisp.py
  3. OPTIONAL Emerald qubit mapping  build_emerald_qubit_rotated.py  (only needed for Qiskit)
  4. Simulation                      TBD (section 4 below)
  5. Hardware execution              run_on_hardware.py: run_qiskit() / run_qrisp()
  6. Syndromes → Decoder             extract_syndromes.py (only needed for Qiskit),
                                     run_on_hardware.py: decode_hardware_results()
  Helpers                            internal_helpers.py

Usage
-----
  from surface_code import make_stim_circuit, DEFAULT_NOISE
  from run_on_hardware import run_qiskit, run_qrisp, decode_hardware_results


"""

import numpy as np
import stim
import pymatching
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
#  1.  STIM CIRCUIT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

#: Default noise model matching IQM Emerald typical gate error rates (~0.1-0.5%).
#: Adjust these based on your latest calibration data from Resonance.
DEFAULT_NOISE = dict(
    after_clifford_depolarization  = 0.003,
    after_reset_flip_probability   = 0.003,
    before_measure_flip_probability= 0.003,
    before_round_data_depolarization = 0.003,
)


def make_stim_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    noise: Optional[dict] = None,
) -> stim.Circuit:
    """
    Generates a Stim circuit for the rotated surface code.

    Shared starting point for BOTH tracks. A Stim circuit does not run on hardware, you need to convert it first
    first:
        Qiskit track : stim_to_qiskit(stim_circ)               (stim_to_qiskit.py)
        Qrisp track  : stim_to_detector_experiment(stim_circ)  (stim_to_qrisp.py)
    Alternatively, you can start from make_qiskit_circuit() / make_qrisp_experiment().

    Parameters
    ----------
    distance : code distance
    rounds   : number of stabilizer rounds
    basis    : "Z" or "X" memory experiment NOTE: Stim's X-memory circuits use RX / MX,
             which the provided converters do not handle. Possible avenue for extension.
    no_reset : True -> ancillas are measured without reset (required on Emerald)
    noise    : None for the hardware circuit. For for simulation / DEM it can be  e.g. DEFAULT_NOISE or sophisticated choices.

    NOTE: Emerald does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant.
    With reset: ancilla starts fresh each round. Raw measurement m_k = syndrome s_k of round k. Detection event = s_k XOR s_(k-1) = m_k XOR m_(k-1).
    Without reset: ancilla accumulates. Raw measurement m_k = s_k XOR m_(k-1), i.e. the XOR of all syndromes up to round k.
    The detection event is the same quantity (s_k XOR s_(k-1)), but in raw measurements it becomes m_k XOR m_(k-2)  (with m_0 = 0).
    So the DETECTOR definitions must be rewritten for the no-reset circuit. stim_to_qrisp.py does this rewrite for you (PLEASE VERIFY!); on the Qiskit track it is on you.
    Trade-off: no-reset avoids reset errors (X_ERROR after R), but the ancilla stays "live" (potentially decohering) throughout, which causes amplitude damping.


    Returns
    -------
    stim.Circuit
    """

    return

def get_circuit_info(circuit: stim.Circuit) -> dict:
    """
    Returns a summary dict of qubit counts and measurement structure.
    Useful for sanity-checking before hardware runs.
    """
    data_q, anc_q = get_qubit_lists(circuit)
    return {
        "num_qubits"      : circuit.num_qubits,
        "num_data_qubits" : len(data_q),
        "num_anc_qubits"  : len(anc_q),
        "num_measurements": circuit.num_measurements,
        "num_detectors"   : circuit.num_detectors,
        "num_observables" : circuit.num_observables,
        "data_stim_indices": sorted(data_q),
        "anc_stim_indices" : sorted(anc_q),
        "meas_order"      : get_meas_order(circuit),
    }


def make_qiskit_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
) -> QuantumCircuit:
    """
    QISKIT TRACK, native alternative.
    Generates a Qiskit circuit for the rotated surface code.

    Pros (vs. Stim + converter)
    ---------------------------
    + Full control over gate basis (e.g. IQM native) and order (e.g. qubit placement). 
    + No converter before it is passed to the transpiler.

    Cons
    ----
    - No detectors or logical observables: Tracking needs to be done by user, across rounds and
      for the no-reset case.
    - No detector error model: for decoding you still need a Stim version of your circuit
      (or build the decoding graph by hand). extract_syndromes() only works with a
      Stim circuit that matches yours.

    NOTE: Emerald does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant.
    With reset: ancilla starts fresh each round. Raw measurement m_k = syndrome s_k of round k. Detection event = s_k XOR s_(k-1) = m_k XOR m_(k-1).
    Without reset: ancilla accumulates. Raw measurement m_k = s_k XOR m_(k-1), i.e. the XOR of all syndromes up to round k.
    The detection event is the same quantity (s_k XOR s_(k-1)), but in raw measurements it changes to m_k XOR m_(k-2)  (with m_0 = 0).


    Returns
    -------
    qiskit.QuantumCircuit
    """

    return

#: CZ order per ancilla type, as (dx, dy) offsets to the data qubit touched in
#: each of the four layers. Same hook-safe schedule as Stim's rotated generator.
_X_ORDER = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_Z_ORDER = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class SurfaceCodeLayout:
    """Rotated surface code on abstract coordinates.

    Data qubits sit at odd (x, y), ancillas at even (x, y). Every coupling is a
    diagonal step, which is a nearest-neighbour edge of a square lattice rotated
    by 45 degrees (see ``lattice_coords``). Qubit index i is position i of the
    Qrisp QuantumArray and of the exported Stim circuit: data first, then X
    ancillas, then Z ancillas.
    """

    distance: int
    coords: tuple            # (x, y) per qubit index
    data: tuple              # qubit indices
    x_ancillas: tuple
    z_ancillas: tuple
    support: dict            # ancilla index -> sorted data indices it checks
    layers: tuple            # 4 layers of (ancilla, data) CZ pairs
    logical_z: tuple         # data indices of a Z string commuting with X checks
    logical_x: tuple         # data indices of an X string commuting with Z checks

    @property
    def num_qubits(self) -> int:
        return len(self.coords)

    @property
    def ancillas(self) -> tuple:
        return self.x_ancillas + self.z_ancillas

    def lattice_coords(self) -> list:
        """Integer square-lattice coordinates (u, v); every CZ pair is Manhattan distance 1."""
        return [((x + y) // 2, (x - y) // 2) for x, y in self.coords]


def rotated_surface_code_layout(distance: int) -> SurfaceCodeLayout:
    """Geometry, stabilizer supports, CZ schedule and logical operators for a distance-d rotated code."""
    if distance < 2:
        raise ValueError("distance must be at least 2")
    edge = 2 * distance

    data_xy = [(x, y) for y in range(1, edge, 2) for x in range(1, edge, 2)]
    data_set = set(data_xy)

    x_xy, z_xy = [], []
    for y in range(0, edge + 1, 2):
        for x in range(0, edge + 1, 2):
            kind = "X" if ((x + y) // 2) % 2 == 1 else "Z"
            # X checks terminate on the top/bottom boundaries, Z checks on left/right.
            if y in (0, edge) and kind != "X":
                continue
            if x in (0, edge) and kind != "Z":
                continue
            neighbours = [(x + dx, y + dy) for dx, dy in _X_ORDER if (x + dx, y + dy) in data_set]
            if len(neighbours) < 2:
                continue
            (x_xy if kind == "X" else z_xy).append((x, y))

    coords = tuple(data_xy + x_xy + z_xy)
    index = {xy: i for i, xy in enumerate(coords)}
    n_data = len(data_xy)
    data = tuple(range(n_data))
    x_anc = tuple(range(n_data, n_data + len(x_xy)))
    z_anc = tuple(range(n_data + len(x_xy), len(coords)))

    support, layers = {}, [[] for _ in range(4)]
    for a in x_anc + z_anc:
        ax, ay = coords[a]
        order = _X_ORDER if a in x_anc else _Z_ORDER
        touched = []
        for layer, (dx, dy) in enumerate(order):
            d = index.get((ax + dx, ay + dy))
            if d is not None and d < n_data:
                layers[layer].append((a, d))
                touched.append(d)
        support[a] = tuple(sorted(touched))

    def commutes(string, checks):
        return all(len(set(string) & set(support[a])) % 2 == 0 for a in checks)

    rows = [tuple(i for i in data if coords[i][1] == y) for y in range(1, edge, 2)]
    cols = [tuple(i for i in data if coords[i][0] == x) for x in range(1, edge, 2)]
    logical_z = next(s for s in rows + cols if commutes(s, x_anc))
    logical_x = next(s for s in rows + cols if commutes(s, z_anc))
    assert len(set(logical_z) & set(logical_x)) % 2 == 1

    return SurfaceCodeLayout(
        distance=distance,
        coords=coords,
        data=data,
        x_ancillas=x_anc,
        z_ancillas=z_anc,
        support=support,
        layers=tuple(tuple(layer) for layer in layers),
        logical_z=logical_z,
        logical_x=logical_x,
    )


def _resolve_noise(noise: Union[None, float, dict]) -> dict:
    """Map None / a single strength / a DEFAULT_NOISE-style dict onto the four noise channels."""
    if noise is None:
        return dict.fromkeys(DEFAULT_NOISE, 0.0)
    if isinstance(noise, (int, float)):
        return dict.fromkeys(DEFAULT_NOISE, float(noise))
    unknown = set(noise) - set(DEFAULT_NOISE)
    if unknown:
        raise ValueError(f"unknown noise keys {sorted(unknown)}; expected keys of DEFAULT_NOISE")
    return {k: float(noise.get(k, 0.0)) for k in DEFAULT_NOISE}


def _sqrt_y_gate(dagger: bool = False):
    """SQRT_Y (or its inverse): one PRX(±pi/2, pi/2) pulse on IQM hardware, and a Clifford Stim understands."""
    import qrisp.circuit.standard_operations as std_ops

    op = std_ops.RYGate(-np.pi / 2 if dagger else np.pi / 2)
    op.name = "sy_dg" if dagger else "sy"
    op.params = []
    op.permeability = {0: False}
    op.is_qfree = False
    return op


def make_qrisp_experiment(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    noise: Union[None, float, dict] = None,
    idle_time: float = 0.0,
):
    """
    QRISP TRACK. Rotated surface code memory experiment as a Qrisp DetectorExperiment.

    Each stabilizer round is SQRT_Y on every ancilla, four CZ layers in Stim's
    hook-safe order, SQRT_Y_DAG on every ancilla, then ancilla readout. Data
    qubits that meet an X check in a layer are rotated into the X frame
    (SQRT_Y_DAG) so the CZ acts as a CNOT from the ancilla; frames persist
    between layers and rounds, so a data qubit only pays a pulse when its frame
    changes. Every single-qubit gate is SQRT_Y or SQRT_Y_DAG, i.e. one native PRX
    on Garnet, and every two-qubit gate is CZ on a square-lattice edge.

    Without reset the ancilla keeps its previous outcome, so the raw readout is
    m_k = s_k XOR m_(k-1). The detector s_k XOR s_(k-1) is therefore
    m_k XOR m_(k-2), with m_0 = 0. With reset (simulation only) it is
    m_k XOR m_(k-1).

    Parameters
    ----------
    distance  : code distance (3 -> 17 qubits)
    rounds    : number of stabilizer rounds, at least 1
    basis     : "Z" or "X" memory
    no_reset  : True for the hardware circuit (Garnet has no mid-circuit reset)
    noise     : None, one strength for every channel, or a dict with DEFAULT_NOISE
                keys. Becomes stim_noise annotations for the DEM; hardware ignores it.
    idle_time : seconds of explicit delay on the ancillas after each readout
                (between rounds). 0 emits only the round barrier.

    Returns
    -------
    iqm.qrisp_iqm.DetectorExperiment
        Returns (detectors, [observable]). ``experiment.layout`` holds the
        SurfaceCodeLayout used to build it.
    """
    from iqm.qrisp_iqm import DetectorExperiment
    from iqm.qrisp_iqm.custom_pulse_operations import delay
    from qrisp import QuantumArray, QuantumBool, barrier, cz, measure, parity, reset
    from qrisp.core.gate_application_functions import append_operation
    from qrisp.misc.stim_tools import stim_noise

    basis = basis.upper()
    if basis not in ("Z", "X"):
        raise ValueError("basis must be 'Z' or 'X'")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if idle_time < 0:
        raise ValueError("idle_time must be non-negative")

    layout = rotated_surface_code_layout(distance)
    p = _resolve_noise(noise)
    ancillas = layout.ancillas
    x_set = set(layout.x_ancillas)
    memory_checks = layout.z_ancillas if basis == "Z" else layout.x_ancillas
    logical = layout.logical_z if basis == "Z" else layout.logical_x
    sy, sy_dg = _sqrt_y_gate(), _sqrt_y_gate(dagger=True)

    def experiment():
        q = QuantumArray(qtype=QuantumBool(), shape=(layout.num_qubits,))

        def gate1(op, i):
            append_operation(op, [q[i]])
            if p["after_clifford_depolarization"]:
                stim_noise("DEPOLARIZE1", p["after_clifford_depolarization"], q[i])

        def set_frame(d, want):
            # Physical |0> in the X frame is logical |+>, and a Z readout there is an X readout.
            if frame[d] != want:
                gate1(sy_dg if want == "X" else sy, d)
                frame[d] = want

        def noise1(name, key, qubits):
            if p[key]:
                for i in qubits:
                    stim_noise(name, p[key], q[i])

        frame = dict.fromkeys(layout.data, basis)
        raw = {a: [] for a in ancillas}
        detectors = []

        for k in range(1, rounds + 1):
            if k > 1 and not no_reset:
                for a in ancillas:
                    reset(q[a])
                noise1("X_ERROR", "after_reset_flip_probability", ancillas)
            noise1("DEPOLARIZE1", "before_round_data_depolarization", layout.data)

            for a in ancillas:
                gate1(sy, a)
            for layer in layout.layers:
                for a, d in layer:
                    set_frame(d, "X" if a in x_set else "Z")
                for a, d in layer:
                    cz(q[a], q[d])
                    if p["after_clifford_depolarization"]:
                        stim_noise("DEPOLARIZE2", p["after_clifford_depolarization"], q[a], q[d])
            for a in ancillas:
                gate1(sy_dg, a)

            noise1("X_ERROR", "before_measure_flip_probability", ancillas)
            for a in ancillas:
                raw[a].append(measure(q[a]))
            barrier(q)
            if idle_time and k < rounds:
                for a in ancillas:
                    delay(q[a], duration=float(idle_time))

            for a in ancillas:
                m = raw[a]
                if k == 1:
                    if a in memory_checks:
                        detectors.append(parity(m[0], expectation=0))
                elif no_reset:
                    terms = (m[-1], m[-3]) if k >= 3 else (m[-1],)
                    detectors.append(parity(*terms, expectation=0))
                else:
                    detectors.append(parity(m[-1], m[-2], expectation=0))

        for d in layout.data:
            set_frame(d, basis)
        noise1("X_ERROR", "before_measure_flip_probability", layout.data)
        data_meas = {d: measure(q[d]) for d in layout.data}

        for a in memory_checks:
            m = raw[a]
            last = (m[-1], m[-2]) if (no_reset and rounds >= 2) else (m[-1],)
            detectors.append(parity(*[data_meas[d] for d in layout.support[a]], *last, expectation=0))

        observable = parity(*[data_meas[d] for d in logical], expectation=0, observable=True)
        return detectors, [observable]

    exp = DetectorExperiment(experiment)
    exp.layout = layout
    return exp


##########################
# Below the necessary steps. No need to put everything in this file. 
##########################

# ─────────────────────────────────────────────────────────────────────────────
#  2.  STIM → Qrisp/QISKIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  3.  EMERALD QUBIT MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  4.  STIM/AER SIMULATION  (no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  5.  Hardware execution
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  6.  Offline Decoding
# ─────────────────────────────────────────────────────────────────────────────

# TBD