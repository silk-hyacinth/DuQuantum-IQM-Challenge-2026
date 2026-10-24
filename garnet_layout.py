# -*- coding: utf-8 -*-
"""
garnet_layout.py

Choose which physical Garnet qubits run the rotated surface code.

Every placement is an exact embedding of the code's CZ graph into the device
coupling graph, so no placement needs a SWAP. Placements are ranked by the
expected number of faults per stabilizer round, built from the current
calibration set:

    CZ error          on each of the 24 CZ pairs (each used once per round)
    PRX error         per single-qubit pulse, counted from the actual schedule
    readout error     on each ancilla (measured once per round)
    idle dephasing    on data qubits while the ancillas are read out

Usage
-----
    from surface_code import rotated_surface_code_layout
    from garnet_layout import fetch_calibration, rank_placements

    cal = fetch_calibration()
    best = rank_placements(rotated_surface_code_layout(3), cal)[0]
    print(best.placement, best.budget)
"""

import ast
import math
import os
from dataclasses import dataclass

import networkx as nx

#: Nominal Garnet durations (seconds). Only the readout time matters for the
#: score; replace with calibrated values when pulse-level access is available.
PRX_TIME = 40e-9
CZ_TIME = 60e-9
READOUT_TIME = 1.5e-6


def fetch_calibration(quantum_computer: str = None, server_url: str = None) -> dict:
    """Per-qubit and per-pair error rates of the current calibration set.

    Reads IQM_TOKEN, IQM_SERVER_URL and IQM_QUANTUM_COMPUTER from the
    environment. Errors are 1 - fidelity; t1/t2 are in seconds.
    """
    from iqm.qiskit_iqm import IQMProvider
    from iqm.qubit_selector.qubit_selector import CalibrationDataManager

    server_url = server_url or os.environ["IQM_SERVER_URL"]
    quantum_computer = quantum_computer or os.environ.get("IQM_QUANTUM_COMPUTER", "garnet")
    backend = IQMProvider(server_url, quantum_computer=quantum_computer).get_backend()
    raw = CalibrationDataManager().get_calibration_fidelities(backend)

    cz = {}
    for key, fidelity in raw["CZ"].items():
        a, b = ast.literal_eval(key)
        cz[frozenset((a, b))] = 1.0 - fidelity

    return {
        "quantum_computer": quantum_computer,
        "qubits": [backend.index_to_qubit_name(i) for i in range(backend.num_qubits)],
        "cz": cz,
        "prx": {q: 1.0 - f for q, f in raw["1Q"].items()},
        "readout": {q: 1.0 - f for q, f in raw["readout"].items()},
        "t1": {q: t * 1e-6 for q, t in raw["t1"].items()},
        "t2": {q: t * 1e-6 for q, t in raw["t2"].items()},
    }


def code_graph(layout) -> nx.Graph:
    """Graph on code qubit indices with one edge per CZ pair."""
    g = nx.Graph()
    g.add_nodes_from(range(layout.num_qubits))
    g.add_edges_from((a, d) for layer in layout.layers for a, d in layer)
    return g


def device_graph(cal: dict) -> nx.Graph:
    """Coupling graph of the device, restricted to calibrated CZ pairs."""
    g = nx.Graph()
    g.add_nodes_from(cal["qubits"])
    g.add_edges_from(tuple(pair) for pair in cal["cz"])
    return g


def prx_per_round(layout) -> dict:
    """Steady-state PRX pulses per round for every code qubit.

    Ancillas pay two (into and out of the X basis). A data qubit pays one pulse
    each time consecutive checks touching it differ in type; frames persist into
    the next round, so the sequence is cyclic.
    """
    x_set = set(layout.x_ancillas)
    seq = {d: [] for d in layout.data}
    for layer in layout.layers:
        for a, d in layer:
            seq[d].append("X" if a in x_set else "Z")
    counts = {a: 2 for a in layout.ancillas}
    for d, s in seq.items():
        counts[d] = sum(s[i] != s[i - 1] for i in range(len(s))) if len(s) > 1 else 0
    return counts


@dataclass(frozen=True)
class ScoredPlacement:
    placement: dict    # code qubit index -> physical qubit name
    budget: dict       # expected faults per round, by source, plus "total"

    @property
    def qubits(self) -> list:
        return [self.placement[i] for i in sorted(self.placement)]


def round_error_budget(layout, placement: dict, cal: dict) -> dict:
    """Expected number of faults per stabilizer round, split by source."""
    pulses = prx_per_round(layout)
    cz = sum(cal["cz"][frozenset((placement[a], placement[d]))]
             for layer in layout.layers for a, d in layer)
    prx = sum(n * cal["prx"][placement[i]] for i, n in pulses.items())
    readout = sum(cal["readout"][placement[a]] for a in layout.ancillas)
    idle = sum(0.5 * (1.0 - math.exp(-READOUT_TIME / cal["t2"][placement[d]]))
               for d in layout.data)
    budget = {"cz": cz, "prx": prx, "readout": readout, "idle": idle}
    budget["total"] = sum(budget.values())
    return budget


def rank_placements(layout, cal: dict, exclude=()) -> list:
    """All zero-SWAP placements of the code on the device, best first."""
    device = device_graph(cal)
    device.remove_nodes_from(exclude)
    matcher = nx.algorithms.isomorphism.GraphMatcher(device, code_graph(layout))
    scored = []
    for dev_to_code in matcher.subgraph_monomorphisms_iter():
        placement = {c: q for q, c in dev_to_code.items()}
        scored.append(ScoredPlacement(placement, round_error_budget(layout, placement, cal)))
    if not scored:
        raise ValueError("the code graph does not embed in the device coupling graph")
    scored.sort(key=lambda s: s.budget["total"])
    return scored


def fixed_layout(placement: dict, device_qubits: list, connectivity: list):
    """Qrisp pass that pins code qubit i of the traced circuit onto placement[i].

    ``device_qubits`` is the backend's qubit order (IQMBackend._dqa.qubits), so
    circuit position p runs on device_qubits[p]. Raises if any two-qubit gate
    lands on a pair that is not a native coupler, which makes the result a
    checked zero-SWAP layout.
    """
    from qrisp import Qubit
    from qrisp.circuit.pass_management.circuit_pass import CircuitPass

    position = {name: p for p, name in enumerate(device_qubits)}
    couplers = {frozenset(e) for e in connectivity}

    @CircuitPass
    def _fixed_layout(qc):
        if qc.num_qubits() != len(placement):
            raise ValueError(f"circuit has {qc.num_qubits()} qubits, placement has {len(placement)}")
        new_qc = qc.copy()
        while new_qc.num_qubits() < len(device_qubits):
            new_qc.add_qubit(Qubit(f"amended_qb_{new_qc.num_qubits()}"))
        slot = [None] * len(device_qubits)
        for code_index, name in placement.items():
            slot[position[name]] = code_index
        spare = iter(range(len(placement), len(device_qubits)))
        new_qc.qubits = [new_qc.qubits[s if s is not None else next(spare)] for s in slot]

        index = {qb: p for p, qb in enumerate(new_qc.qubits)}
        for instr in new_qc.data:
            if len(instr.qubits) == 2 and instr.op.name not in ("barrier",):
                pair = frozenset(index[qb] for qb in instr.qubits)
                if pair not in couplers:
                    names = sorted(device_qubits[p] for p in pair)
                    raise ValueError(f"{instr.op.name} on {names} is not a native coupler")
        return new_qc

    _fixed_layout.__name__ = "fixed_layout"
    return _fixed_layout
