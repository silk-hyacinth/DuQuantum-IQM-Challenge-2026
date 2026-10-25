"""
check_surface_code.py — offline self-test for Build 1. No IQM token, no credits.

    python check_surface_code.py        (~2 minutes)

Why this exists
---------------
A surface code circuit can be wrong in ways that still "run": a detector that
fires even without noise, a CZ order that quietly halves the code's strength, a
gate the hardware can't do. On hardware those bugs look exactly like noise, so
we prove the circuit is right in simulation first.

How to read the output
----------------------
One line per configuration, for example

    ok  d=3 r=3 Z no-reset   qubits=17 detectors=24 CZ=72

  ok         every assertion below passed for this configuration
  d / r      code distance / number of stabilizer rounds
  Z / X      which kind of information is protected (bit value / phase)
  no-reset   the hardware variant (ancillas never cleared); "reset" is a
             simulation-only comparison
  qubits     9 data + 8 ancilla = 17 for d=3
  detectors  number of alarm bits the decoder gets. Formula:
             (#Z or #X checks) x 2 + (#ancillas) x (rounds - 1)
  CZ         two-qubit gates: 24 per round for d=3, the minimum possible

If anything is wrong the script stops with an AssertionError naming the
configuration and what broke.

What is checked, per configuration
----------------------------------
  * qubit counts
  * only SQRT_Y / SQRT_Y_DAG / CZ / M (+ R with reset) appear, and every CZ is a
    data-ancilla pair that are neighbours on a square grid (Garnet's wiring)
  * detectors are deterministic: with NO noise, every detector and the logical
    observable read 0 in raw samples. (Raw matters: the hardware result is
    raw bits, nobody subtracts a reference for us.)
  * "graphlike distance" of the noisy circuit equals d: the smallest number of
    faults that can flip the logical value without the decoder noticing. If the
    CZ order were bad (hook errors) this would come out smaller than d.
Then once: the optional idle delay survives into the hardware circuit, each
single-qubit gate becomes exactly one PRX pulse, the code fits Garnet with zero
SWAPs (using the saved dashboard calibration), and a simulated LER.
"""

import glob

import numpy as np
import stim

from surface_code import make_qrisp_experiment

# Gates we allow in the exported Stim circuit (TICK = layer separator).
ALLOWED = {"SQRT_Y", "SQRT_Y_DAG", "CZ", "M", "TICK"}
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR"}


def record_sets(circuit: stim.Circuit):
    """For every DETECTOR / OBSERVABLE, the absolute indices of the measurements it XORs.

    Stim stores detectors as "rec[-3]"-style relative offsets; converting to
    absolute indices lets us XOR the right columns of a raw sample table.
    """
    n_meas, dets, obs = 0, [], {}
    for inst in circuit.flattened():
        if inst.name == "M":
            n_meas += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            dets.append([n_meas + t.value for t in inst.targets_copy()])
        elif inst.name == "OBSERVABLE_INCLUDE":
            k = int(inst.gate_args_copy()[0])
            obs.setdefault(k, []).extend(n_meas + t.value for t in inst.targets_copy())
    return dets, obs


def check(distance: int, rounds: int, basis: str, no_reset: bool, shots: int = 256) -> None:
    """Run every structural and correctness check for one configuration."""
    exp = make_qrisp_experiment(distance, rounds, basis=basis, no_reset=no_reset)
    layout = exp.layout
    circuit = exp.to_stim()
    tag = f"d={distance} r={rounds} {basis} {'no-reset' if no_reset else 'reset'}"

    # Right number of qubits: d² data + (d² - 1) ancillas.
    assert circuit.num_qubits == distance**2 + (distance**2 - 1), tag
    assert len(layout.data) == distance**2 and len(layout.ancillas) == distance**2 - 1, tag

    # Only gates that map 1:1 onto Garnet's native operations.
    allowed = ALLOWED | ({"R"} if not no_reset else set())
    names = {inst.name for inst in circuit.flattened()} - {"DETECTOR", "OBSERVABLE_INCLUDE"}
    assert names <= allowed, f"{tag}: unexpected gates {names - allowed}"

    # Every CZ joins one data and one ancilla qubit that are grid neighbours
    # (Manhattan distance 1 on the rotated lattice) -> embeddable with no SWAPs.
    data, lattice = set(layout.data), layout.lattice_coords()
    n_cz = 0
    for inst in circuit.flattened():
        if inst.name != "CZ":
            continue
        t = [x.value for x in inst.targets_copy()]
        for a, b in zip(t[::2], t[1::2]):
            assert (a in data) != (b in data), f"{tag}: CZ {a}-{b} is not data-ancilla"
            (ua, va), (ub, vb) = lattice[a], lattice[b]
            assert abs(ua - ub) + abs(va - vb) == 1, f"{tag}: CZ {a}-{b} is not a lattice edge"
            n_cz += 1
    # Exactly one CZ per (ancilla, data neighbour) per round: nothing wasted.
    weights = sum(len(s) for s in layout.support.values())
    assert n_cz == rounds * weights, f"{tag}: {n_cz} CZs, expected {rounds * weights}"

    # Detector count: memory-basis checks in round 1 and at the end, every
    # ancilla in rounds 2..r.
    expected_dets = len(layout.z_ancillas) * 2 + len(layout.ancillas) * (rounds - 1)
    assert circuit.num_detectors == expected_dets, f"{tag}: {circuit.num_detectors} detectors"
    assert circuit.num_observables == 1, tag

    circuit.detector_error_model()  # raises on a non-deterministic detector or observable

    # Noise-free raw samples: every detector and the observable must XOR to 0.
    dets, obs = record_sets(circuit)
    raw = circuit.compile_sampler().sample(shots)
    for i, s in enumerate(dets):
        assert not np.any(np.bitwise_xor.reduce(raw[:, s], axis=1)), f"{tag}: detector {i} fires noiselessly"
    for k, s in obs.items():
        assert not np.any(np.bitwise_xor.reduce(raw[:, s], axis=1)), f"{tag}: observable {k} flips noiselessly"
    ref = circuit.reference_sample()
    assert not np.bitwise_xor.reduce(ref[obs[0]]), f"{tag}: observable disagrees with reference_sample()"
    assert not circuit.compile_detector_sampler().sample(shots).any(), f"{tag}: detector sampler fired"

    # Code strength: the fewest faults that fool the decoder must equal d.
    noisy = make_qrisp_experiment(distance, rounds, basis=basis, no_reset=no_reset, noise=1e-3).to_stim()
    found = len(noisy.shortest_graphlike_error())
    assert found == distance, f"{tag}: graphlike distance {found}"

    print(f"ok  {tag:<24} qubits={circuit.num_qubits} detectors={circuit.num_detectors} CZ={n_cz}")


if __name__ == "__main__":
    # d=3 in both bases, both reset modes, several round counts; d=5 to show
    # the generator scales (49 qubits: does not fit Garnet, simulation only).
    for basis in ("Z", "X"):
        for no_reset in (True, False):
            for rounds in (1, 2, 3, 5):
                check(3, rounds, basis, no_reset)
    check(5, 3, "Z", True)
    check(5, 3, "X", True)

    from qrisp import PassManager, convert_to_prx

    # idle_time adds an iqm.delay on each ancilla between rounds (2 gaps x 8
    # ancillas for 3 rounds). Stim ignores it, so the detectors are unchanged.
    # Separately: lowering to native gates must give exactly one PRX per
    # SQRT_Y / SQRT_Y_DAG (no hidden extra pulses).
    idle = make_qrisp_experiment(3, 3, basis="X", idle_time=200e-9)
    ops = [i.op.name for i in idle.to_qc().data]
    gaps = 8 * 2
    assert ops.count("iqm.delay") == gaps, "ancilla idle delay missing from the hardware circuit"
    assert idle.to_stim().num_detectors == 24, "idle delay changed the Stim export"
    pm = PassManager()
    pm.add_pass(convert_to_prx)
    lowered = [i.op.name for i in pm.run(idle.to_qc()).data]
    assert lowered.count("prx") == ops.count("sy") + ops.count("sy_dg"), "SQRT_Y did not lower to one PRX"
    print(f"ok  idle delay: {gaps} ancilla delays kept for hardware, {lowered.count('prx')} single-qubit gates -> one PRX each")

    # Garnet placement from a saved dashboard calibration file (offline): the
    # code graph must embed in Garnet's coupler graph, i.e. zero SWAPs.
    cal_files = sorted(glob.glob("results/dashboard/calibration-*.json"))
    if cal_files:
        from garnet_layout import fetch_calibration, rank_placements
        cal = fetch_calibration(cal_files[-1])
        ranked = rank_placements(make_qrisp_experiment(3, 1).layout, cal)
        sets = {frozenset(s.qubits) for s in ranked}
        unused = sorted(set(cal["qubits"]) - set(ranked[0].qubits), key=lambda q: int(q[2:]))
        print(f"ok  garnet placement: {len(ranked)} zero-SWAP placements, {len(sets)} qubit set(s), "
              f"unused {unused}, expected faults/round {ranked[0].budget['total']:.2f}")

    # Simulated logical error probability at 0.1% uniform noise. A sanity
    # number for the pipeline, NOT a prediction for Garnet (Garnet's CZ error
    # is ~0.5%). Random sampling: expect it to wobble around 0.0010-0.0018.
    exp = make_qrisp_experiment(3, 3, noise=1e-3)
    print(f"stim LER, d=3 r=3 Z no-reset, p=1e-3: {exp.compute_LER(shots=20_000):.4f}")
    print("all checks passed")
