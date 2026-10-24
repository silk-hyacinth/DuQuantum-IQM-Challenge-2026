# -*- coding: utf-8 -*-
"""
garnet_layout.py — decide WHICH physical qubits on IQM Garnet run the surface code.

Big picture
-----------
`surface_code.make_qrisp_experiment` builds a circuit on 17 *abstract* qubits
numbered 0..16 (0-8 = data, 9-12 = X-check ancillas, 13-16 = Z-check ancillas).
The real chip has 20 *physical* qubits called QB1..QB20, and a CZ gate can only
run between physically wired neighbours ("couplers"). If two qubits that must
interact are not neighbours, a compiler has to insert SWAP gates to move them
next to each other, which adds a lot of noise. The judges reward avoiding SWAPs
(30% of the score), so this file:

  1. Reads the chip's current calibration (how good every qubit and coupler is).
  2. Finds EVERY way to place the 17 abstract qubits on 17 physical qubits such
     that every CZ in the circuit lands on a real coupler (zero SWAPs). This is
     a "subgraph monomorphism" search: does the code's interaction graph fit
     inside the chip's wiring graph?
  3. Scores each placement by the expected number of faults per round, using
     the calibration numbers, and sorts best-first.
  4. Provides a Qrisp compiler pass (`fixed_layout`) that pins the circuit onto
     the chosen qubits and REFUSES to continue if any CZ is not on a coupler.
     That refusal is our machine-checked proof of "zero SWAPs".

What we learned from running it (Oct 24 2026)
---------------------------------------------
On Garnet only ONE set of 17 qubits fits the d=3 code: it always leaves out
QB3, QB17 and QB18. The search returns 4 placements, but they are rotations /
reflections of the same set, so they all score the same. In other words the
search proves that no better qubit set exists, rather than picking one.

Typical use
-----------
    from surface_code import rotated_surface_code_layout
    from garnet_layout import fetch_calibration, rank_placements

    cal = fetch_calibration()                       # live, needs IQM_TOKEN
    cal = fetch_calibration("results/dashboard/calibration-....json")   # offline
    best = rank_placements(rotated_surface_code_layout(3), cal)[0]
    print(best.qubits)    # physical qubit for code qubit 0, 1, 2, ...
    print(best.budget)    # expected faults per round, split by cause
"""

import json
import math
import os
import re
from dataclasses import dataclass

import networkx as nx

# Nominal Garnet operation durations in seconds. Only READOUT_TIME affects the
# score today (it sets how long data qubits sit idle while the ancillas are
# measured). These are estimates, not calibrated values: with pulse-level access
# (PulLA) they should be replaced with the real durations from the calibration.
PRX_TIME = 40e-9
CZ_TIME = 60e-9
READOUT_TIME = 1.5e-6


# ─────────────────────────────────────────────────────────────────────────────
#  1. Calibration data
# ─────────────────────────────────────────────────────────────────────────────
#
# IQM publishes a "quality metric set" for every calibration: a flat list of
# (field name, value) pairs such as
#     "metrics.irb.cz.slepian_crf.QB2__QB5.fidelity:par=d2" -> 0.9958
#     "characterization.model.QB2.t2_time"                  -> 5.2e-06
# The dashboard's "calibration data" download is the same list as JSON.
#
# We read the fields by their exact names. (IQM's helper
# `iqm.qubit_selector.CalibrationDataManager` matches names by substring, which
# picks the *heralded measure_fidelity* readout number instead of the plain
# `measure` our circuit uses, and can return either Ramsey or echo T2 depending
# on field order. That made QB16 look like an 8.9% readout error when the
# relevant number is 2.8%.)

_FIELDS = {
    # name in our dict : regex over the IQM field name. Group(s) = qubit name(s).
    "cz":      r"metrics\.irb\.cz\.[^.]+\.(QB\d+)__(QB\d+)\.fidelity(:par=d2)?$",   # interleaved RB CZ fidelity
    "prx":     r"metrics\.rb\.prx\.[^.]+\.(QB\d+)\.fidelity(:par=d2)?$",            # RB single-qubit fidelity
    "readout": r"metrics\.ssro\.measure\.constant\.(QB\d+)\.assignment_error$",     # P(wrong readout), unheralded
    "e01":     r"metrics\.ssro\.measure\.constant\.(QB\d+)\.error_0_to_1$",         # prepared 0, read 1
    "e10":     r"metrics\.ssro\.measure\.constant\.(QB\d+)\.error_1_to_0$",         # prepared 1, read 0 (decay during readout)
    "leakage": r"metrics\.rrc\.measure\.constant\.(QB\d+)\.leakage$",               # P(readout pushes qubit out of {0,1})
    "t1":      r"characterization\.model\.(QB\d+)\.t1_time$",                       # energy relaxation time, seconds
    "t2":      r"characterization\.model\.(QB\d+)\.t2_time$",                       # Ramsey dephasing time (no echo), seconds
    "t2_echo": r"characterization\.model\.(QB\d+)\.t2_echo_time$",                  # dephasing time with one echo pulse, seconds
}


def _parse_observations(pairs, calibration_set_id=None, quantum_computer=None) -> dict:
    """Turn (field name, value) pairs into the calibration dict used below.

    Output keys
    -----------
    qubits   : ["QB1", ..., "QB20"] sorted by number
    cz       : {frozenset({"QB2","QB5"}): CZ error, ...}  (error = 1 - fidelity)
    prx      : {"QB1": single-qubit gate error, ...}
    readout  : {"QB1": probability a readout returns the wrong bit, ...}
    e01, e10 : the two directions of readout error separately
    leakage  : {"QB1": probability readout leaves the qubit in |2>, ...}
    t1, t2, t2_echo : {"QB1": seconds, ...}
    """
    cal = {k: {} for k in _FIELDS}
    patterns = {k: re.compile(p) for k, p in _FIELDS.items()}
    for field, value in pairs:
        for key, pattern in patterns.items():
            m = pattern.search(field)
            if not m:
                continue
            if key == "cz":
                cal["cz"][frozenset((m.group(1), m.group(2)))] = 1.0 - value
            elif key == "prx":
                cal["prx"][m.group(1)] = 1.0 - value
            else:
                cal[key][m.group(1)] = float(value)
            break
    cal["qubits"] = sorted(cal["t1"], key=lambda q: int(q[2:]))
    cal["calibration_set_id"] = calibration_set_id
    cal["quantum_computer"] = quantum_computer
    return cal


def fetch_calibration(path: str = None, quantum_computer: str = None, server_url: str = None) -> dict:
    """Get the calibration numbers, either live from IQM Resonance or from a file.

    path             : a dashboard "calibration data" JSON file. If given, no
                       network or token is needed (useful for teammates).
    quantum_computer : defaults to $IQM_QUANTUM_COMPUTER or "garnet".
    server_url       : defaults to $IQM_SERVER_URL.

    Live mode reads IQM_TOKEN from the environment (run_qrisp loads it from .env).
    """
    if path:
        with open(path) as f:
            raw = json.load(f)
        pairs = [(o["dut_field"], o["value"]) for o in raw["observations"] if not o.get("invalid")]
        return _parse_observations(pairs, raw.get("describes_id"), quantum_computer)

    from iqm.iqm_client import IQMClient

    server_url = server_url or os.environ["IQM_SERVER_URL"]
    quantum_computer = quantum_computer or os.environ.get("IQM_QUANTUM_COMPUTER", "garnet")
    client = IQMClient(server_url, quantum_computer=quantum_computer)
    metrics = client.get_quality_metric_set()
    pairs = [(o.dut_field, o.value) for o in metrics.observations if not getattr(o, "invalid", False)]
    calset = client.get_dynamic_quantum_architecture(None).calibration_set_id
    return _parse_observations(pairs, str(calset), quantum_computer)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Graphs: what the code needs vs. what the chip offers
# ─────────────────────────────────────────────────────────────────────────────

def code_graph(layout) -> nx.Graph:
    """Nodes = the 17 abstract code qubits; edge = "these two do a CZ together".

    For d=3 this has 24 edges (each weight-4 check touches 4 data qubits, each
    weight-2 check touches 2: 4*4 + 4*2 = 24).
    """
    g = nx.Graph()
    g.add_nodes_from(range(layout.num_qubits))
    g.add_edges_from((a, d) for layer in layout.layers for a, d in layer)
    return g


def device_graph(cal: dict) -> nx.Graph:
    """Nodes = physical qubits; edge = a calibrated CZ coupler (30 on Garnet)."""
    g = nx.Graph()
    g.add_nodes_from(cal["qubits"])
    g.add_edges_from(tuple(pair) for pair in cal["cz"])
    return g


# ─────────────────────────────────────────────────────────────────────────────
#  3. Scoring a placement
# ─────────────────────────────────────────────────────────────────────────────

def prx_per_round(layout) -> dict:
    """How many single-qubit pulses (PRX) each code qubit gets per round.

    Ancillas always pay 2: one to rotate into the X basis before the CZs, one
    to rotate back before measuring. A data qubit pays one pulse each time it
    switches between being checked by an X check and a Z check (see the frame
    trick in surface_code.make_qrisp_experiment). Frames carry over to the next
    round, so the sequence of check types is treated as a loop. For d=3 every
    qubit ends up with exactly 2.
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
    placement: dict    # code qubit index (0..16) -> physical qubit name ("QB2")
    budget: dict       # expected faults per round, by source, plus "total"

    @property
    def qubits(self) -> list:
        """Physical qubit for code qubit 0, 1, 2, ... in order."""
        return [self.placement[i] for i in sorted(self.placement)]


def round_error_budget(layout, placement: dict, cal: dict) -> dict:
    """Expected number of faults in ONE stabilizer round, split by cause.

    This is a simple "add up the error probabilities" estimate. It is not a
    simulation; it tells us which kind of error dominates, which tells us what
    to fix first.

      cz      : sum of CZ errors over the 24 CZs in a round
      prx     : sum over qubits of (pulses per round x that qubit's PRX error)
      readout : sum of readout errors over the 8 ancillas (measured every round)
      idle    : the 9 data qubits sit idle for READOUT_TIME while ancillas are
                read out; probability of a phase flip in that time is
                (1 - exp(-t / T2)) / 2, using Ramsey T2 (no dynamical decoupling)

    A total around 1 means "about one fault somewhere on the patch per round".
    """
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
    """Every zero-SWAP placement of the code on the chip, best (lowest total) first.

    exclude : physical qubits to forbid (e.g. ("QB12",)). On Garnet d=3 there is
              only one possible qubit set, so excluding any of its 17 qubits
              makes this raise.
    """
    device = device_graph(cal)
    device.remove_nodes_from(exclude)
    # GraphMatcher(big, small).subgraph_monomorphisms_iter() yields every map
    # physical -> code such that each code edge lands on a physical edge.
    matcher = nx.algorithms.isomorphism.GraphMatcher(device, code_graph(layout))
    scored = []
    for dev_to_code in matcher.subgraph_monomorphisms_iter():
        placement = {c: q for q, c in dev_to_code.items()}
        scored.append(ScoredPlacement(placement, round_error_budget(layout, placement, cal)))
    if not scored:
        raise ValueError("the code graph does not embed in the device coupling graph")
    scored.sort(key=lambda s: s.budget["total"])
    return scored


# ─────────────────────────────────────────────────────────────────────────────
#  4. Pinning the circuit to the chosen qubits
# ─────────────────────────────────────────────────────────────────────────────

def fixed_layout(placement: dict, device_qubits: list, connectivity: list):
    """Qrisp compiler pass: put code qubit i onto physical qubit placement[i].

    How Qrisp decides which physical qubit runs which circuit wire: the IQM
    converter sends circuit wire number p to device_qubits[p] (for Garnet,
    wire 0 -> QB1, wire 1 -> QB2, ...). So "placing" a circuit just means
    re-ordering its wires (and padding with unused wires up to 20). This is the
    same trick IQM's own `vf2pp_layout` pass uses, except we choose the order
    from our calibration-ranked placement instead of the first match found.

    device_qubits : backend._dqa.qubits, the backend's wire order
    connectivity  : backend.connectivity, list of (wire, wire) couplers

    After re-ordering, every two-qubit gate is checked against the couplers;
    a non-native pair raises an error instead of silently needing a SWAP.
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
        # Pad with idle wires so wire numbers cover every physical qubit.
        while new_qc.num_qubits() < len(device_qubits):
            new_qc.add_qubit(Qubit(f"amended_qb_{new_qc.num_qubits()}"))
        # slot[p] = which code qubit sits on physical wire p (None = unused).
        slot = [None] * len(device_qubits)
        for code_index, name in placement.items():
            slot[position[name]] = code_index
        spare = iter(range(len(placement), len(device_qubits)))
        new_qc.qubits = [new_qc.qubits[s if s is not None else next(spare)] for s in slot]

        # Zero-SWAP proof: every 2-qubit gate must be on a native coupler.
        index = {qb: p for p, qb in enumerate(new_qc.qubits)}
        for instr in new_qc.data:
            if len(instr.qubits) == 2 and instr.op.name != "barrier":
                pair = frozenset(index[qb] for qb in instr.qubits)
                if pair not in couplers:
                    names = sorted(device_qubits[p] for p in pair)
                    raise ValueError(f"{instr.op.name} on {names} is not a native coupler")
        return new_qc

    _fixed_layout.__name__ = "fixed_layout"
    return _fixed_layout
