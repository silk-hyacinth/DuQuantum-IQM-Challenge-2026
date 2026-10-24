# -*- coding: utf-8 -*-
"""
decoders.py — alternative decoders to compare against the baseline. No credits needed.

The rule
--------
A decoder here is any function

    my_decoder(syndromes: dict) -> (logical_error, sigma)

where `syndromes` is one saved hardware run (from load_hardware_results):
    syndromes["det_events"]  bool (shots, n_detectors)   which alarms fired
    syndromes["obs_flips"]   bool (shots, 1)             did the logical value flip
    syndromes["rounds"], ["basis"], ["distance"], ["no_reset"]
    syndromes["placement"]   code qubit index (str) -> physical qubit ("QB2")
    syndromes["calibration"] per-qubit numbers of the qubits used
                             (readout, e01, e10, leakage, t1, t2, t2_echo, prx, cz)

Try one on the saved baseline (re-decodes the same hardware shots):

    python sweep.py --load results/sweeps/<baseline>.json --decoder decoders:garnet_uniform
    python sweep.py --compare results/sweeps/<baseline>.json results/sweeps/<new>.json

Because both use identical shots, any eps_L difference is caused by the decoder.
"""

from run_on_hardware import decode_hardware_results


def garnet_uniform(syndromes: dict):
    """Same PyMatching decoder, but weights from Garnet's typical error sizes.

    The baseline assumes 0.3% for every error type. On Garnet the median CZ
    error is ~0.5%, readout ~2%, and data qubits idling through one ancilla
    readout pick up a few % of phase error, so readout and idle errors are
    much more likely than the baseline assumes. Telling the decoder that
    changes which explanation it picks for a given alarm pattern.

    Result on the Oct 24 baseline (job 01a16fb3): WORSE. eps_L went from
    0.0833 to 0.0880 (X) and 0.0855 to 0.0880 (Z). Plausible reason: these
    blanket numbers over-weight idle/readout errors on qubits that are fine
    and under-weight the bad ones. Lesson: decoder weights must be tested,
    not assumed; per-qubit weights from syndromes["calibration"] are the
    natural next attempt.
    """
    noise = dict(
        after_clifford_depolarization=0.005,     # CZ / PRX
        before_measure_flip_probability=0.02,    # readout
        before_round_data_depolarization=0.04,   # data idling during ancilla readout
        after_reset_flip_probability=0.0,        # no resets on hardware
    )
    return decode_hardware_results(syndromes, noise=noise)
