# Running the pipeline (fake backend or real IQM hardware)

## Files
| File | What it does |
|---|---|
| `surface_code.py` | `make_stim_circuit()` (no-reset circuit + detectors), `make_qiskit_circuit()` (Qiskit version for hardware) |
| `run_on_hardware.py` | `run_qiskit()` (transpile → run → detection events), `decode_hardware_results()` (PyMatching → P_error) |
| `run_experiment.py` | Loops over round counts, decodes, fits ε_L, saves JSON + plot to `results/` |
| `test_pipeline.py` | Sanity tests, no QPU needed (see `TESTING.md`) |

## 1. Test locally first (free, no account)
```bash
python run_experiment.py --fake --rounds 1 2 3 --shots 2000
```
Uses `IQMFakeGarnet`, a local simulator with Garnet's qubit layout and a realistic noise model. Slow: rounds 1–5 at 2,000 shots took ~10 minutes (it simulates 20 noisy qubits). Use fewer rounds/shots for quick checks.

## 2. Real hardware
Set your Resonance API token (from the Resonance dashboard) once per terminal:

```powershell
# Windows PowerShell
$env:IQM_TOKEN = "paste-token-here"
```
```bash
# macOS / Linux
export IQM_TOKEN="paste-token-here"
```

Then:
```bash
python run_experiment.py --qc garnet --rounds 1 2 3 4 5 --shots 2000
python run_experiment.py --qc garnet --rounds 1 2 3 4 5 --shots 2000 --dd     # with dynamical decoupling
python run_experiment.py --qc emerald ...                                    # when Emerald is back
```

Each round count is a separate job. Check your QPU time quota on the Resonance dashboard before big runs.

## 3. Outputs (all in `results/`)
| File | Contents | When it's written |
|---|---|---|
| `<name>_calibration.json` | Device error rates at the time of the run | First, before any job |
| `<name>.json` | Settings; per round count: raw counts, P_error, physical qubits used (`layout`); fitted ε_L at the end | **After every job** (`"status": "in_progress"` until all finish, then `"complete"`) |
| `<name>.png` | P_error vs rounds with the fit | At the end |

Raw counts are saved so you can re-decode later (e.g. with a calibration-based noise model) without re-running on hardware. If a run dies halfway (crash, quota, Wi-Fi), the jobs that finished are already in the JSON.

**Calibration file format**
- Real hardware (`"source": "resonance"`): `fidelities["CZ"]["[QB1, QB2]"]`, `["1Q"]["QB1"]`, `["readout"]["QB1"]`, `["readout_qndness"]`, … are **fidelities** (error = 1 − fidelity); `["t1"]`, `["t2"]` are coherence times. Fetched with `iqm-qubit-selector` (same as the repo's `iqm_utils/get_calibration_data.py`).
- Fake backend (`"source": "fake_error_profile"`): the noise the simulator uses: CZ / single-qubit error rates, readout errors, T1/T2 in ns.
- If fetching fails, the run continues and the file says `"source": "unavailable"` with the error.

**`layout`** (per run) lists every code qubit with its role (data/ancilla), position in the code, and the physical qubit it ran on (e.g. `QB12`). Combined with the calibration file, this tells you the error rate of every gate and qubit your circuit actually used.

## 4. What `run_qiskit` does, step by step
1. `make_qiskit_circuit()` builds the circuit; `make_stim_circuit(noise=None)` builds its Stim twin, which defines the detectors.
2. `transpile()` rewrites the circuit into the device's native gates (`r`, `cz`) and picks physical qubits. It prints CX vs CZ counts: equal means 0 SWAPs.
3. `backend.run()` submits the job and waits for counts.
4. `counts_to_measurement_array()` → `extract_syndromes()` turn the bitstrings into detection events and logical flips.

`decode_hardware_results()` then builds the decoder from a **noisy** Stim copy of the same circuit (default: uniform 0.3% noise guess) and returns P_error ± error bar.

## 5. Fit
$$P_\text{error}(n) = \tfrac{1}{2}\left[1 - (1 - 2\varepsilon_L)^n\right]$$
fitted over all round counts with `scipy.optimize.curve_fit`; ε_L is the logical error rate per round.

## 6. Example result (fake Garnet, d=3, 2,000 shots per round count)
`python run_experiment.py --fake --rounds 1 2 3 4 5 --shots 2000`

| rounds | CX → CZ | depth | P_error | no correction |
|---|---|---|---|---|
| 1 | 24 → 24 | 9 | 0.022 ± 0.003 | 0.113 |
| 2 | 48 → 48 | 17 | 0.058 ± 0.005 | 0.153 |
| 3 | 72 → 72 | 25 | 0.073 ± 0.006 | 0.190 |
| 4 | 96 → 96 | 32 | 0.112 ± 0.007 | 0.201 |
| 5 | 120 → 120 | 39 | 0.129 ± 0.008 | 0.243 |

**Fitted ε_L = 0.0280 ± 0.0010** per round, 0 SWAPs at every round count.
Files: `results/fakegarnet_d3_20261024_153700.json`, `results/fakegarnet_d3_20261024_153700.png`.
This is a simulator baseline; real Garnet/Emerald numbers will differ.
