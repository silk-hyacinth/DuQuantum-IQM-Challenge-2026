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
  pymatching   minimum-weight perfect matching (PyMatching)

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

import numpy as np
import stim

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


@register("pymatching")
def pymatching_decoder(det_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Minimum-weight perfect matching: the most likely set of errors that explains the shot."""
    import pymatching
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return np.asarray(matcher.decode_batch(det_events), dtype=bool)


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
