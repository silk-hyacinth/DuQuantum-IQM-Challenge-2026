## What this adds

A second decoder, `qubit_calibrated`, selectable with the existing `--decoder` flag (including `--load`, so it costs no QPU time on saved runs):

```bash
python run_experiment.py --load results/<run>.json --decoder qubit_calibrated
```

It is the same minimum-weight perfect matching as `pymatching`, but the matching graph is weighted per physical qubit from that run's calibration snapshot:

- after each 1-qubit gate: that qubit's PRX error
- after each CZ: that pair's CZ error
- each readout: that qubit's readout error
- after a mid-circuit readout: 1 − QNDness (leakage if QNDness is missing)
- idle qubits during a readout: half the estimated phase error over the assumed 1.5 µs readout (echo T2 with DD, Ramsey T2 without)

## How it plugs in

- `decoding.py`: `qubit_calibrated_rates` / `qubit_calibrated_circuit`, and the registered decoder. A decoder can now carry a `prepare_twin(twin, ctx)` hook that builds its own noisy twin.
- `run_experiment.py`: `decode_record` uses `prepare_twin` when the decoder has one, otherwise `--noise` as before. Real-hardware runs now call `_load_credentials()` before the calibration snapshot, so a `.env` token is picked up.
- `Details.md`: one bullet documenting the decoder and its results.

## Results (saved 17:47 / 17:49 Garnet shots, Qrisp track, simple fit)

| weights | Z no DD | X no DD | Z with DD | X with DD |
| --- | --- | --- | --- | --- |
| uniform 0.3% | 0.126 | 0.099 | 0.078 | 0.073 |
| qubit_calibrated | 0.109 | 0.096 | 0.079 | 0.073 |

Helps the runs without DD; neutral with DD. Smoke-tested on the 20:17 Qiskit-track DD run: runs end to end (simple fit 6.7% → 7.0%, amplitude fit 14.3% → 14.7%, i.e. no gain there either).

## Notes for review

- With `--decoder qubit_calibrated`, `--noise` is ignored, but the `--load` output file name still contains the `--noise` value (e.g. `_redecode_uniform_qubit_calibrated_`).
- It overlaps with `--noise calibration` (also per-qubit); the differences are the halved idle term, no T1 decay term, and the leakage fallback. Worth comparing the two on the same shots.
- The docstring table says "README-style"; that fit is now called the "simple fit".
