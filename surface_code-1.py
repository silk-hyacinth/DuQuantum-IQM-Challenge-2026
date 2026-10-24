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
from qiskit import QuantumCircuit, transpile
from typing import Optional


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
    Rotated surface code memory experiment (Z basis).

    no_reset=False : Stim's built-in circuit (ancillas reset with MR every round).
    no_reset=True  : same gates, but ancillas are measured with M (no reset), and every
                     DETECTOR / OBSERVABLE is rewritten so it still means "did this check change?".
                     Trick: Stim's detectors are written in terms of each ancilla's per-round answer.
                     Without reset, answer_k = m_k XOR m_(k-1), so each ancilla measurement in a
                     detector is replaced by {m_k, m_(k-1)}; XOR-ing as sets makes repeats cancel,
                     which yields the m_k XOR m_(k-2) rule from the README.
    noise=None for hardware; e.g. DEFAULT_NOISE for simulation / building the decoder.
    """
    if basis != "Z":
        raise NotImplementedError("Only Z-basis memory is supported for now")

    circ = stim.Circuit.generated("surface_code:rotated_memory_z",
                                  distance=distance, rounds=rounds, **(noise or {}))
    if not no_reset:
        return circ

    out = stim.Circuit()
    n = 0          # number of measurements so far
    last = {}      # qubit -> index of its most recent measurement
    prev = {}      # ancilla measurement index -> index of that ancilla's previous measurement
    for inst in circ.flattened():
        assert isinstance(inst, stim.CircuitInstruction)
        if inst.name in ("M", "MR"):
            for t in inst.targets_copy():
                if inst.name == "MR":
                    prev[n] = last.get(t.value)
                last[t.value] = n
                n += 1
            inst = stim.CircuitInstruction("M", inst.targets_copy(), inst.gate_args_copy())
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            recs = set()
            for t in inst.targets_copy():
                j = n + t.value                  # t.value is negative ("k measurements ago")
                recs ^= {j}
                if prev.get(j) is not None:      # answer = this reading XOR previous reading
                    recs ^= {prev[j]}
            inst = stim.CircuitInstruction(inst.name, [stim.target_rec(j - n) for j in sorted(recs)],
                                           inst.gate_args_copy())
        out.append(inst)
    return out

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
    Builds the Stim circuit (noiseless), converts it with stim_to_qiskit, and drops the
    reset instructions (only present at the very start when no_reset=True; hardware
    qubits already start in |0>, and IQM's native gate set has no 'reset').
    """
    from stim_to_qiskit import stim_to_qiskit

    stim_circ = make_stim_circuit(distance, rounds, basis=basis, no_reset=no_reset, noise=None)
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)

    clean = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name != "reset":
            clean.append(inst)
    return clean

def make_qrisp_experiment(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    p: Optional[float] = None,
):
    """
    QRISP TRACK, native alternative to make_stim_circuit() + stim_to_detector_experiment().
    Generates a Qrisp DetectorExperiment for the rotated surface code.

    Pros (vs. Stim + converter)
    ---------------------------
    + Detectors and observables are part of the experiment (qrisp.parity). Qrisp derives the Stim circuit and DEM from it.
    + Noise model per gate via stim_noise annotations: easy to adjust according to calibration data.
    + Full control over gate choice, CX/CZ schedule and layout.

    Cons
    ----
    - Geometry, CX schedule and detector bookkeeping: Qrisp has no
      surface-code generator.
    - Expectation values must be correct. Wrong choice is Invisible in a
      noiseless Stim simulation, but breaks the hardware results.
    - The no-reset detector algebra (which raw measurements form a detector) needs to be implemented.

    Parameters
    ----------
    distance, rounds, basis, no_reset : as in make_stim_circuit()
    p        : noise strength for stim_noise annotations (decoder weights only;
               ignored on hardware). None -> noiseless.

    Returns
    -------
    iqm.qrisp_iqm.DetectorExperiment
    """

    return


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