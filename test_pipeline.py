"""
test_pipeline.py — sanity tests for the surface code pipeline (no QPU time needed).

Run from the repo root:   python test_pipeline.py

Tests
-----
1. Stim simulation, with reset vs. no reset (d=3, rounds 1..8)
2. No-reset detectors are valid for several sizes (Stim builds a DEM without errors)
3. Stim -> Qiskit conversion is correct (noiseless Qiskit run => 0 detectors fire, 0 logical flips)
4. Transpiling onto IQM fake backends inserts 0 SWAPs
5. Full pipeline on IQM's noisy fake Garnet (circuit -> transpile -> run -> syndromes -> decode)
6. Qrisp track: its circuit, converted for a local simulator, gives 0 detections without noise
7. Noise models and decoders: every built-in model gives a valid decoder map on both tracks,
   and a decoder can be named as module:function

Slow parts: test 5 runs a noisy 20-qubit simulation, test 6 traces Qrisp circuits (~1-2 min each).
The Qrisp branch's own, deeper self-test is check_surface_code.py.
"""

import numpy as np
import pymatching
from qiskit import transpile
from qiskit_aer import AerSimulator
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from iqm.qiskit_iqm import IQMFakeAphrodite

from surface_code import make_stim_circuit, make_qiskit_circuit, DEFAULT_NOISE
from internal_helpers import counts_to_measurement_array
from extract_syndromes import extract_syndromes


def decode(det_events, obs_flips, noisy_stim_circuit):
    """PyMatching decode. Returns (P_error, 1-sigma error bar, undecoded flip rate)."""
    dem = noisy_stim_circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    pred = np.asarray(matcher.decode_batch(det_events))
    p = np.mean(pred[:, 0] != obs_flips[:, 0])
    return p, np.sqrt(p * (1 - p) / len(det_events)), obs_flips[:, 0].mean()


def hardware_counts_to_syndromes(counts, distance, rounds, shots, basis="Z"):
    """Qiskit counts -> detection events, using the noisy Stim twin of the circuit that ran."""
    stim_noisy = make_stim_circuit(distance, rounds, basis=basis, no_reset=True, noise=DEFAULT_NOISE)
    raw = counts_to_measurement_array(counts, stim_noisy.num_measurements, shots)
    return extract_syndromes(raw, stim_noisy, print_summary=False), stim_noisy


# ─────────────────────────────────────────────────────────────────────────────
def test_stim_simulation(distance=3, rounds_list=(1, 2, 3, 5, 8), shots=20_000):
    print(f"\n[1] Stim simulation, uniform noise 0.3%  (d={distance}, {shots} shots)")
    for no_reset in (False, True):
        print(f"  no_reset = {no_reset}")
        for r in rounds_list:
            circ = make_stim_circuit(distance, r, no_reset=no_reset, noise=DEFAULT_NOISE)
            det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
            p, err, raw = decode(det, obs, circ)
            print(f"    rounds={r}  P_error={p:.4f} ± {err:.4f}   (no correction: {raw:.3f})")


def test_detectors_valid(distances=(3, 5), rounds_list=(1, 2, 3, 5, 8)):
    print("\n[2] No-reset detector validity (Stim raises if any detector is non-deterministic)")
    for basis in ("Z", "X"):
      for d in distances:
        for r in rounds_list:
            circ = make_stim_circuit(d, r, basis=basis, no_reset=True, noise=None)
            circ.detector_error_model()                     # raises on a wrong detector
            assert "MR" not in str(circ)
            n_anc = d * d - 1
            assert circ.num_detectors == n_anc * r, (circ.num_detectors, n_anc * r)
            print(f"    {basis}-memory d={d} rounds={r}: OK  ({circ.num_detectors} detectors, no MR)")


def test_qiskit_noiseless(distance=3, rounds_list=(1, 2, 3), shots=2_000):
    print("\n[3] Stim -> Qiskit conversion, noiseless Aer run (expect 0 detectors, 0 flips)")
    sim = AerSimulator(method="stabilizer")
    for basis in ("Z", "X"):
        for r in rounds_list:
            qc = make_qiskit_circuit(distance, r, basis=basis)
            counts = sim.run(qc, shots=shots).result().get_counts()
            s, _ = hardware_counts_to_syndromes(counts, distance, r, shots, basis)
            n_det, n_obs = int(s["det_events"].sum()), int(s["obs_flips"].sum())
            assert n_det == 0 and n_obs == 0
            print(f"    {basis}-memory rounds={r}: detectors fired={n_det}, logical flips={n_obs}  OK")


def test_zero_swaps(rounds=3):
    print("\n[4] Transpile onto IQM fake backends (each CX -> exactly 1 CZ means 0 SWAPs)")
    for backend, distances in ((IQMFakeGarnet(), (3,)), (IQMFakeAphrodite(), (3, 5))):
        for d in distances:
            qc = make_qiskit_circuit(d, rounds)
            t = transpile(qc, backend, optimization_level=3, seed_transpiler=1)
            cx, cz = qc.count_ops()["cx"], t.count_ops()["cz"]
            assert cx == cz
            print(f"    {backend.name} ({backend.num_qubits}q) d={d}: CX={cx} -> CZ={cz}, "
                  f"depth {t.depth()}  => 0 SWAPs")


def test_fake_garnet_pipeline(distance=3, rounds_list=(1, 2, 3), shots=2_000):
    print("\n[5] Full pipeline on noisy fake Garnet (decoder uses uniform 0.3% noise guess)")
    backend = IQMFakeGarnet()
    for r in rounds_list:
        qc = make_qiskit_circuit(distance, r)
        t = transpile(qc, backend, optimization_level=3, seed_transpiler=1)
        counts = backend.run(t, shots=shots).result().get_counts()
        s, stim_noisy = hardware_counts_to_syndromes(counts, distance, r, shots)
        p, err, raw = decode(s["det_events"], s["obs_flips"], stim_noisy)
        print(f"    rounds={r}  P_error={p:.3f} ± {err:.3f}   (no correction: {raw:.3f})")


def test_qrisp_noiseless(rounds_list=(1, 2), shots=200):
    print("\n[6] Qrisp track, noiseless simulation (expect 0 detections, 0 flips)")
    from tracks import TRACKS, qrisp_to_qiskit
    sim = AerSimulator()
    track = TRACKS["qrisp"]
    for basis in ("Z", "X"):
        for r in rounds_list:
            _, qc, post = track._prepare(3, r, basis)
            counts = sim.run(qrisp_to_qiskit(qc), shots=shots).result().get_counts()
            det, obs = track._events(post, counts, reverse=False)
            assert det.shape[1] == track.twin(3, r, basis).num_detectors
            assert len(track.detector_info(3, r, basis)) == det.shape[1]
            assert int(det.sum()) == 0 and int(obs.sum()) == 0
            print(f"    {basis}-memory rounds={r}: detectors fired=0, logical flips=0  OK")


def test_noise_models_and_decoders(rounds=3, shots=5_000):
    print("\n[7] Noise models x decoders on both tracks (Stim simulation)")
    from tracks import TRACKS
    from noise_models import NOISE_MODELS, NoiseContext, add_noise, calibration_from_fake, calibration_to_json
    from decoding import DECODERS, get_decoder, logical_error
    cal = calibration_to_json(calibration_from_fake(IQMFakeGarnet()))
    qubit_map = {q: f"QB{q + 1}" for q in range(20)}       # any placement works for this check
    assert get_decoder("decoding:pymatching_decoder") is DECODERS["pymatching"]
    for tname, track in TRACKS.items():
        twin = track.twin(3, rounds, "Z")
        sample = add_noise(twin, "uniform:0.003", NoiseContext())
        det, obs = sample.compile_detector_sampler().sample(shots, separate_observables=True)
        for spec in ["uniform", "uniform:0.01", "calibration"]:
            noisy = add_noise(twin, spec, NoiseContext(calibration=cal, qubit_map=qubit_map))
            for dec in DECODERS:
                p, err, _ = logical_error(det, obs, noisy, dec)
                assert p < 0.1
                print(f"    {tname:6} noise={spec:13} decoder={dec}: sim P_error={p:.4f} +- {err:.4f}")


if __name__ == "__main__":
    test_stim_simulation()
    test_detectors_valid()
    test_qiskit_noiseless()
    test_zero_swaps()
    test_fake_garnet_pipeline()
    test_qrisp_noiseless()
    test_noise_models_and_decoders()
    print("\nAll tests passed.")
