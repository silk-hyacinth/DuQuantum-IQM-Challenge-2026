"""
Local checks for make_qrisp_experiment (build 1). No Resonance token needed.

    python check_surface_code.py

For every basis and reset mode it verifies, on the Stim export of the Qrisp experiment:
  * qubit counts (d=3: 17 = 9 data + 8 ancilla)
  * only SQRT_Y / SQRT_Y_DAG / CZ / M (+ R with reset) appear, and every CZ is a
    data-ancilla square-lattice edge
  * detectors are deterministic: Stim builds a DEM, and on raw noiseless samples
    every detector and the observable equal their declared expectation (0).
    Raw parities matter on hardware, where no Stim reference sample is subtracted.
  * graphlike distance of the noisy twin equals the code distance
"""

import numpy as np
import stim

from surface_code import make_qrisp_experiment

ALLOWED = {"SQRT_Y", "SQRT_Y_DAG", "CZ", "M", "TICK"}
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR"}


def record_sets(circuit: stim.Circuit):
    """Absolute measurement indices of every DETECTOR and OBSERVABLE_INCLUDE."""
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
    exp = make_qrisp_experiment(distance, rounds, basis=basis, no_reset=no_reset)
    layout = exp.layout
    circuit = exp.to_stim()
    tag = f"d={distance} r={rounds} {basis} {'no-reset' if no_reset else 'reset'}"

    assert circuit.num_qubits == distance**2 + (distance**2 - 1), tag
    assert len(layout.data) == distance**2 and len(layout.ancillas) == distance**2 - 1, tag

    allowed = ALLOWED | ({"R"} if not no_reset else set())
    names = {inst.name for inst in circuit.flattened()} - {"DETECTOR", "OBSERVABLE_INCLUDE"}
    assert names <= allowed, f"{tag}: unexpected gates {names - allowed}"

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
    weights = sum(len(s) for s in layout.support.values())
    assert n_cz == rounds * weights, f"{tag}: {n_cz} CZs, expected {rounds * weights}"

    expected_dets = len(layout.z_ancillas) * 2 + len(layout.ancillas) * (rounds - 1)
    assert circuit.num_detectors == expected_dets, f"{tag}: {circuit.num_detectors} detectors"
    assert circuit.num_observables == 1, tag

    circuit.detector_error_model()  # raises on a non-deterministic detector or observable

    dets, obs = record_sets(circuit)
    raw = circuit.compile_sampler().sample(shots)
    for i, s in enumerate(dets):
        assert not np.any(np.bitwise_xor.reduce(raw[:, s], axis=1)), f"{tag}: detector {i} fires noiselessly"
    for k, s in obs.items():
        assert not np.any(np.bitwise_xor.reduce(raw[:, s], axis=1)), f"{tag}: observable {k} flips noiselessly"
    ref = circuit.reference_sample()
    assert not np.bitwise_xor.reduce(ref[obs[0]]), f"{tag}: observable disagrees with reference_sample()"
    assert not circuit.compile_detector_sampler().sample(shots).any(), f"{tag}: detector sampler fired"

    noisy = make_qrisp_experiment(distance, rounds, basis=basis, no_reset=no_reset, noise=1e-3).to_stim()
    found = len(noisy.shortest_graphlike_error())
    assert found == distance, f"{tag}: graphlike distance {found}"

    print(f"ok  {tag:<24} qubits={circuit.num_qubits} detectors={circuit.num_detectors} CZ={n_cz}")


if __name__ == "__main__":
    for basis in ("Z", "X"):
        for no_reset in (True, False):
            for rounds in (1, 2, 3, 5):
                check(3, rounds, basis, no_reset)
    check(5, 3, "Z", True)
    check(5, 3, "X", True)

    from qrisp import PassManager, convert_to_prx

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

    exp = make_qrisp_experiment(3, 3, noise=1e-3)
    print(f"stim LER, d=3 r=3 Z no-reset, p=1e-3: {exp.compute_LER(shots=20_000):.4f}")
    print("all checks passed")
