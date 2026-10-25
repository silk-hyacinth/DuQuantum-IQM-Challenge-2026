import numpy as np
import pymatching
from surface_code import make_stim_circuit, DEFAULT_NOISE

SHOTS = 20_000
DISTANCE = 3


def simulate(distance, rounds, no_reset, noise, shots):
    """Simulate the noisy circuit and decode. Returns (P_error, error_bar, undecoded)."""
    circ = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=noise)

    # 1. Run the noisy circuit many times
    det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True) # run this on iqm to get det, obs

    # 2. Build the decoder from the circuit's list of possible errors
    dem = circ.detector_error_model(decompose_errors=True)   # crashes here if comparisons are wrong
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # 3. Decode: guess whether each shot's logical qubit got flipped
    pred = np.asarray(matcher.decode_batch(det))

    # 4. Logical error = guess was wrong
    p = np.mean(pred[:, 0] != obs[:, 0])
    return p, np.sqrt(p * (1 - p) / shots), obs[:, 0].mean()


for no_reset in [False, True]:
    print(f"\n=== no_reset = {no_reset} ===")
    for rounds in [1, 2, 3, 5, 8]:
        p, err, raw = simulate(DISTANCE, rounds, no_reset, DEFAULT_NOISE, SHOTS)
        print(f"rounds={rounds}  P_error={p:.4f} ± {err:.4f}   (no correction: {raw:.3f})")