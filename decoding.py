"""
decoding.py — decoders (--decoder) and the logical error probability.

A decoder looks at which detectors fired in each shot and predicts whether the
logical value got flipped. Every decoder here has the same shape:

    my_decoder(det_events, dem) -> predictions

    det_events  : bool array (shots, num_detectors)  True = that detector fired
    dem         : stim.DetectorErrorModel             which errors are possible, how likely,
                                                       which detectors each one fires
    predictions : bool array (shots, num_observables)  True = "the logical value flipped"

Built-in
--------
  pymatching         minimum-weight perfect matching (PyMatching)
  qubit_calibrated   the same matching, with each physical qubit weighted by its
                     own calibration. prepare_twin writes those weights before matching.
  lookup_ml          maximum likelihood by lookup table: for every detector pattern, the
                     more likely logical value under the noise model. Exact for small
                     circuits, estimated from simulated shots for bigger ones.

Adding a decoder
----------------
In this file:

    @register("union_find")
    def union_find(det_events, dem):
        ...
        return predictions

or in any other file, without touching this one: --decoder my_file:my_function
(module name : function name, the function having the same shape).

Note: the separate decoders.py (Qrisp branch) holds decoders for sweep.py, which take
a whole saved run instead; they are not used by run_experiment.py.
"""

import importlib
import math

import numpy as np
import stim

from noise_models import READOUT_TIME, NoiseContext, Rates, annotate

DECODERS: dict = {}


def register(name: str):
    """Decorator: make a decoder available as --decoder <name>."""
    def wrap(fn):
        DECODERS[name] = fn
        return fn
    return wrap


def get_decoder(spec: str):
    """'pymatching' -> the registered decoder; 'module:function' -> that function."""
    if spec in DECODERS:
        return DECODERS[spec]
    if ":" in spec:
        module, _, func = spec.partition(":")
        return getattr(importlib.import_module(module), func)
    raise ValueError(f"unknown decoder '{spec}'. Available: {', '.join(sorted(DECODERS))} "
                     f"(or module:function)")


def qubit_calibrated_rates(ctx: NoiseContext) -> Rates:
    """One error rate per physical qubit, from that run's calibration snapshot.

    The idle term is half the estimated phase error during one readout, because
    the 1.5 us readout duration is an assumption. Echo T2 when the run used
    dynamical decoupling, Ramsey T2 otherwise. Measured on the saved 17:47 and
    17:49 Garnet shots (no new QPU time), README-style per-round error:

        weights              Z no DD   X no DD   Z with DD   X with DD
        uniform 0.3%         0.126     0.099     0.078       0.073
        every qubit, full    0.098     0.098     0.084       0.091
        worst 4 qubits only  0.128
        half idle (this one) 0.109     0.096     0.079       0.073

    Full idle and worst-four (QB2, QB6, QB11, QB12) each made a basis worse.
    Half idle helps the runs without decoupling and leaves the decoupling runs
    where they were. It does not beat 7.8% (Z) and 7.3% (X).
    """
    cal = ctx.calibration or {}
    if not cal.get("readout") or not cal.get("cz"):
        raise ValueError("qubit_calibrated needs a calibration snapshot with per-qubit readout and CZ errors")

    def qname(q: int):
        return ctx.physical(q)

    def gate1(q: int) -> float:
        return (cal.get("prx") or {}).get(qname(q), 0.001)

    def gate2(a: int, b: int) -> float:
        pair = "__".join(sorted((str(qname(a)), str(qname(b)))))
        return (cal.get("cz") or {}).get(pair, 0.005)

    def readout(q: int) -> float:
        return (cal.get("readout") or {}).get(qname(q), 0.02)

    def after_readout(q: int) -> float:
        name = qname(q)
        qnd = (cal.get("qndness") or {}).get(name)
        if qnd is not None:
            return max(1.0 - qnd, 0.0)
        return (cal.get("leakage") or {}).get(name, 0.0)

    def idle(q: int):
        name = qname(q)
        table = cal.get("t2_echo") if ctx.dd else cal.get("t2")
        t2 = (table or {}).get(name) or (cal.get("t2") or {}).get(name, 10e-6)
        phase = 0.5 * (1.0 - math.exp(-READOUT_TIME / t2))
        return (0.0, 0.0, 0.5 * phase)

    return Rates(gate1=gate1, gate2=gate2, readout=readout, reset=lambda q: 0.0,
                 after_readout=after_readout, idle=idle)


def qubit_calibrated_circuit(noiseless_twin: stim.Circuit, ctx: NoiseContext) -> stim.Circuit:
    """Noiseless twin plus one error rate per physical qubit."""
    return annotate(noiseless_twin, qubit_calibrated_rates(ctx))


@register("pymatching")
def pymatching_decoder(det_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Minimum-weight perfect matching: the most likely set of errors that explains the shot."""
    import pymatching
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return np.asarray(matcher.decode_batch(det_events), dtype=bool)


@register("qubit_calibrated")
def qubit_calibrated_decoder(det_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Matching on the graph prepare_twin built from this run's calibration."""
    return pymatching_decoder(det_events, dem)


qubit_calibrated_decoder.prepare_twin = qubit_calibrated_circuit


# ── Maximum likelihood by lookup table ──────────────────────────────────────
#
# For a given detector pattern (syndrome) s, the best possible guess is the logical
# value L with the larger P(s, L) under the noise model. Matching approximates this;
# a lookup table gets it exactly, but has 2^(detectors) rows, so:
#   exact   : <= EXACT_MAX_BITS detectors+observables (d=3: up to 2 rounds).
#             P(s, L) for every s and L, built one error mechanism at a time:
#             each error with probability p either doesn't happen or XORs its
#             detectors and observables into the state.
#   sampled : bigger circuits. Estimate the table from LOOKUP_SAMPLES simulated
#             shots of the same noise model; patterns never seen fall back to matching.

EXACT_MAX_BITS = 21          # 2^21 states x 8 bytes = 16 MB
LOOKUP_SAMPLES = 1_000_000


def _error_masks(dem: stim.DetectorErrorModel):
    """(probability, bitmask) per error mechanism; bits 0..n-1 detectors, then observables."""
    n = dem.num_detectors
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        mask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                mask ^= 1 << (n + t.val)
        out.append((inst.args_copy()[0], mask))
    return out


def _exact_table(dem: stim.DetectorErrorModel) -> np.ndarray:
    """Most likely logical value(s), as an int, for every syndrome index."""
    n, k = dem.num_detectors, dem.num_observables
    prob = np.zeros(1 << (n + k))
    prob[0] = 1.0
    idx = np.arange(prob.size)
    for p, mask in _error_masks(dem):
        prob = (1.0 - p) * prob + p * prob[idx ^ mask]
    return prob.reshape(1 << k, 1 << n).argmax(axis=0)        # row = observable value


def _syndrome_index(det_events: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(det_events.shape[1], dtype=np.int64))
    return det_events.astype(np.int64) @ weights


@register("lookup_ml")
def lookup_ml_decoder(det_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Maximum-likelihood decoding by lookup table (exact when small, sampled otherwise)."""
    det_events = np.asarray(det_events, dtype=bool)
    n, k = dem.num_detectors, dem.num_observables
    if n + k <= EXACT_MAX_BITS:
        best = _exact_table(dem)[_syndrome_index(det_events)]
        return ((best[:, None] >> np.arange(k)) & 1).astype(bool)

    # Sampled table: count, per syndrome, how often each logical value occurred.
    sim_det, sim_obs, _ = dem.compile_sampler().sample(LOOKUP_SAMPLES)
    sim_keys = np.packbits(sim_det, axis=1)
    keys, inverse = np.unique(sim_keys, axis=0, return_inverse=True)
    obs_val = (sim_obs.astype(np.int64) << np.arange(k)).sum(axis=1)
    counts = np.zeros((len(keys), 1 << k), dtype=np.int64)
    np.add.at(counts, (inverse.ravel(), obs_val), 1)
    table = {kb.tobytes(): int(c.argmax()) for kb, c in zip(keys, counts)}

    out = pymatching_decoder(det_events, dem)                  # fallback for unseen patterns
    for i, kb in enumerate(np.packbits(det_events, axis=1)):
        v = table.get(kb.tobytes())
        if v is not None:
            out[i] = (v >> np.arange(k)) & 1
    return out


def logical_error(det_events, obs_flips, noisy_twin: stim.Circuit, decoder: str = "pymatching"):
    """Decode every shot and count the decoder's mistakes.

    noisy_twin : the Stim twin of the circuit that ran, with a noise model added
                 (noise_models.add_noise). Its detectors must match det_events.
    Returns (P_error, 1-sigma error bar, predictions).
    """
    det_events = np.asarray(det_events, dtype=bool)
    obs_flips = np.asarray(obs_flips, dtype=bool)
    if det_events.shape[1] != noisy_twin.num_detectors:
        raise ValueError(f"{det_events.shape[1]} detectors in the data, "
                         f"{noisy_twin.num_detectors} in the circuit twin")
    dem = noisy_twin.detector_error_model(decompose_errors=True)
    predictions = np.asarray(get_decoder(decoder)(det_events, dem), dtype=bool).reshape(obs_flips.shape)
    failures = np.any(predictions != obs_flips, axis=1)
    p = float(failures.mean())
    return p, float(np.sqrt(p * (1 - p) / len(failures))), predictions
