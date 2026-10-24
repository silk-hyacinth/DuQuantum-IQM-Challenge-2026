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
  pymatching         minimum-weight perfect matching, using whatever noise model
                     built the detector error model (--noise)
  qubit_calibrated   the same matching, but each physical qubit is weighted by
                     its own calibration (see below). Both tracks. Needs the
                     run's calibration snapshot and qubit map.

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
    """Per-qubit error rates for the matching graph.

    Every qubit the circuit used is weighted by its own numbers from the
    calibration snapshot. This is not a placement change: the shots stay the
    same, and the decoder is told which physical qubits to trust less.

    What was measured, on the saved 17:47/17:49 Garnet sweeps (same shots,
    re-decoded, no new QPU time). README-fit epsilon_L:

        weights              Z no DD   X no DD   Z with DD   X with DD
        uniform 0.3%         0.126     0.099     0.078       0.073
        every qubit, full    0.098     0.098     0.084       0.091
        worst 4 qubits only  0.128       --        --          --
        half idle (this one) 0.109     0.096     0.079       0.073

    Using each qubit's full estimated idle error helps Z with no decoupling
    and makes X with decoupling worse. Trusting only the worst four qubits
    (QB2, QB6, QB11, QB12 on that calibration) did not help. Halving the idle
    term, because the 1.5 us readout duration is an estimate, is the weighting
    that lowers Z, slightly lowers X, and leaves the decoupling runs unchanged.
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
        # How often a measurement disturbs the qubit. Prefer QNDness; fall
        # back to readout leakage for snapshots that did not record it.
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
    """The noiseless twin with one error rate per physical qubit, for matching."""
    return annotate(noiseless_twin, qubit_calibrated_rates(ctx))


@register("pymatching")
def pymatching_decoder(det_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Minimum-weight perfect matching: the most likely set of errors that explains the shot."""
    import pymatching
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return np.asarray(matcher.decode_batch(det_events), dtype=bool)


def logical_error(det_events, obs_flips, noisy_twin: stim.Circuit, decoder: str = "pymatching",
                  context: NoiseContext = None):
    """Decode every shot and count the decoder's mistakes.

    noisy_twin : the Stim twin of the circuit that ran. For pymatching it must
                 already contain the noise model. For qubit_calibrated pass the
                 NOISELESS twin plus context (calibration snapshot and qubit map);
                 the per-qubit weights are added here.
    Returns (P_error, 1-sigma error bar, predictions).
    """
    if decoder == "qubit_calibrated":
        if context is None:
            raise ValueError("decoder qubit_calibrated needs a NoiseContext (calibration and qubit_map)")
        noisy_twin = qubit_calibrated_circuit(noisy_twin, context)
        decoder = "pymatching"
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
