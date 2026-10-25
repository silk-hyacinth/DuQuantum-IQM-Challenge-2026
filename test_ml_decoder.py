"""
test_ml_decoder.py: checks that ml_decoder is really exact, and what it buys you.

    python test_ml_decoder.py            # all checks (about a minute)

1. EXACTNESS: on a small circuit, build the full table P(syndrome, observable)
   by brute force (no sweep, no retiring, no ordering tricks) and check that
   ml_decoder makes the same decision for EVERY possible syndrome.
2. NEVER WORSE THAN MATCHING: on the same sampled shots, ML must make no more
   mistakes than PyMatching (it can only win or tie, up to a few shots of
   floating-point ties).
3. SPEED: time on the d=3 sizes you run on hardware.
"""

import time

import numpy as np
import pymatching
import stim

from ml_decoder import dem_mechanisms, ml_decoder


def noisy_surface_code(rounds, p, basis="z"):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=3, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def brute_force_table(dem):
    """P(syndrome, observable) for ALL syndromes: apply every mechanism to the full table."""
    n = dem.num_detectors
    table = np.zeros(2 ** (n + 1))
    table[0] = 1.0                      # index = syndrome << 1 | observable
    for p, dets, obs in dem_mechanisms(dem):
        sig = obs
        for d in dets:
            sig |= 1 << (d + 1)
        table = (1 - p) * table + p * table[np.arange(len(table)) ^ sig]
    return table.reshape(2 ** n, 2)


def check_exact():
    circuit = noisy_surface_code(rounds=2, p=0.02)
    dem = circuit.detector_error_model(decompose_errors=True)
    n = dem.num_detectors
    table = brute_force_table(dem)
    want = table[:, 1] > table[:, 0]

    syndromes = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    got = ml_decoder(syndromes, dem)[:, 0]
    assert abs(table.sum() - 1.0) < 1e-9
    close = np.abs(table[:, 1] - table[:, 0]) < 1e-12 * table.sum(axis=1)   # genuine ties
    bad = (got != want) & ~close
    print(f"1. exactness: {n} detectors, all {2 ** n} syndromes, "
          f"{int(bad.sum())} disagreements with brute force")
    assert not bad.any()


def compare_with_matching(rounds, p, shots=20_000, basis="z"):
    circuit = noisy_surface_code(rounds, p, basis)
    dem = circuit.detector_error_model(decompose_errors=True)
    det, obs = circuit.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)

    t = time.time()
    matching = np.asarray(pymatching.Matching.from_detector_error_model(dem).decode_batch(det),
                          dtype=bool)
    t_match = time.time() - t
    t = time.time()
    ml = ml_decoder(det, dem, verbose=True)
    t_ml = time.time() - t

    e_match = np.mean(np.any(matching != obs, axis=1))
    e_ml = np.mean(np.any(ml != obs, axis=1))
    print(f"   d=3 r={rounds} p={p}: no correction {obs.mean():.4f}   "
          f"PyMatching {e_match:.4f} ({t_match:.1f}s)   exact ML {e_ml:.4f} ({t_ml:.1f}s)")
    return e_match, e_ml, shots


if __name__ == "__main__":
    check_exact()

    print("2. ML vs PyMatching, same shots, same error model:")
    for rounds, p in [(1, 0.02), (3, 0.02), (3, 0.04)]:
        e_match, e_ml, shots = compare_with_matching(rounds, p)
        assert e_ml <= e_match + 5 / shots, "ML should never lose to matching"

    print("3. timing at hardware sizes (error model only, shots sampled from it):")
    for rounds in (2, 4, 6):
        compare_with_matching(rounds, 0.03, shots=2_000)
    print("all checks passed")
