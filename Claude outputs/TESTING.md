# Pipeline testing — circuit generation & Qiskit conversion

All tests run locally, no QPU time or Resonance account needed.

```bash
python test_pipeline.py
```

## What was implemented (`surface_code.py`)

**`make_stim_circuit(distance, rounds, basis="Z", no_reset=True, noise=None)`**
- `no_reset=False`: Stim's built-in `surface_code:rotated_memory_z` circuit (ancillas reset with `MR` every round).
- `no_reset=True` (required on Emerald/Garnet): same gates, ancillas measured with `M` only, and every `DETECTOR` / `OBSERVABLE_INCLUDE` rewritten.
  - Without reset, an ancilla's raw reading is the running XOR of its answers: $m_k = s_k \oplus m_{k-1}$, so $s_k = m_k \oplus m_{k-1}$.
  - Stim's original detectors are written in terms of answers $s_k$. Each ancilla measurement inside a detector is replaced by the pair $\{m_k, m_{k-1}\}$; combining these as sets with XOR cancels repeats, giving the README's $m_k \oplus m_{k-2}$ rule automatically (round 1: $m_1$; round 2: $m_2$; final detectors gain $m_{\text{last}-1}$).
- `noise=None` for hardware; `DEFAULT_NOISE` (uniform 0.3%) for simulation and for the decoder's error model.
- Only Z-basis supported (the provided converters don't handle `RX`/`MX`).

**`make_qiskit_circuit(distance, rounds, basis="Z", no_reset=True)`**
- Builds the noiseless Stim circuit, converts it with `stim_to_qiskit`, and removes the initial `reset` instructions (qubits start in |0⟩; IQM's native gate set has no `reset`).
- Contains no SWAPs; SWAPs could only be added by `transpile()` (see test 4).

## Tests and results

### 1. Stim simulation: with reset vs. no reset
d=3, uniform 0.3% noise, 20,000 shots, PyMatching decoder.

| rounds | P_error (reset) | P_error (no reset) | no correction (no reset) |
|---|---|---|---|
| 1 | 0.0024 ± 0.0003 | 0.0024 ± 0.0003 | 0.034 |
| 2 | 0.0041 ± 0.0005 | 0.0033 ± 0.0004 | 0.050 |
| 3 | 0.0061 ± 0.0006 | 0.0069 ± 0.0006 | 0.067 |
| 5 | 0.0106 ± 0.0007 | 0.0099 ± 0.0007 | 0.094 |
| 8 | 0.0175 ± 0.0009 | 0.0157 ± 0.0009 | 0.133 |

**Takeaway:** no-reset matches with-reset within error bars, P_error grows with rounds, and decoding is ~10× better than no correction.

### 2. No-reset detector validity
d=3 and d=5, rounds 1, 2, 3, 5, 8. For each: Stim builds a detector error model without error (it raises if any detector is non-deterministic, i.e. would fire without an error), no `MR` remains, and the number of detectors is $(d^2-1)\times\text{rounds}$. **All passed.**

The no-reset detectors were also cross-checked against an independent, hand-written implementation of the $m_k \oplus m_{k-2}$ rule: identical detector definitions for d=3, 5 and rounds 1–8.

### 3. Stim → Qiskit conversion (noiseless)
`make_qiskit_circuit` run on Qiskit Aer's noiseless stabilizer simulator, then `counts_to_measurement_array` → `extract_syndromes` (the same path hardware results take).

| rounds | detectors fired | logical flips |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |

**Takeaway:** the conversion, Qiskit bit ordering, and detector definitions all line up.

### 4. Zero SWAPs after transpiling
`transpile(qc, backend, optimization_level=3, seed_transpiler=1)`, 3 rounds. Each CX becomes exactly one CZ, so CX count = CZ count means no SWAPs (each SWAP would add 3 CZs).

| backend | d | CX before | CZ after | depth | SWAPs |
|---|---|---|---|---|---|
| IQM fake Garnet (20q) | 3 | 72 | 72 | 25 | 0 |
| IQM fake Aphrodite (54q) | 3 | 72 | 72 | 25 | 0 |
| IQM fake Aphrodite (54q) | 5 | 240 | 240 | 27 | 0 |

Qiskit's transpiler first searches for a perfect placement (VF2Layout) and finds one, so no hand-written qubit map is needed to avoid SWAPs. A hand-picked map can still help by choosing the best-calibrated qubits.

### 5. Full pipeline on noisy fake Garnet
`make_qiskit_circuit` → `transpile` → run on `IQMFakeGarnet` (local simulator with Garnet's layout and a realistic noise model) → `extract_syndromes` → PyMatching. 2,000 shots. Decoder uses the uniform 0.3% noise guess.

| rounds | P_error | no correction |
|---|---|---|
| 1 | 0.025 ± 0.003 | 0.103 |
| 2 | 0.057 ± 0.005 | 0.150 |
| 3 | 0.070 ± 0.006 | 0.187 |

**Takeaway:** the whole hardware path works end to end. Decoding helps ~3–4×, even though the decoder's noise model doesn't match the backend's. Matching the decoder's noise model to the device (calibration data) is an obvious next improvement.

## Notes
- Fake backends: `from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet` and `from iqm.qiskit_iqm import IQMFakeAphrodite`. No fake Emerald is included in `iqm-client` 35.0.3; Aphrodite is a 54-qubit square-lattice stand-in.
- Results vary slightly run to run (random sampling).
- Environment: Python 3.11+, packages from `requirements.txt` (stim 1.16.0, pymatching 2.4.0, qiskit 2.1.2, qiskit-aer, iqm-client[qiskit] 35.0.3).
