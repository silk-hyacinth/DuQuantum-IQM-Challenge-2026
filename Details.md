# Surface Code Pipeline: Step by Step

Oct 24, 2026 · @Ben

The pipeline stores one logical bit on 17 qubits, checks it for errors for a number of rounds, and reports how often the decoder still gets it wrong: the logical error rate per round, ε\_L.

## Overview

&#91;embedded content: pipeline · 7 steps\]

The circuit from step 1 is used three times: converted for the chip (top row), as the XOR rules that turn raw bits into detections (step 5), and as a noisy copy that tells the decoder which errors are likely (step 6).

| File | What it holds |
| --- | --- |
| `run_experiment.py` | The one command: runs every step for a list of round counts, fits, saves, plots |
| `tracks.py` | The two ways to build and run the circuit (`--track qiskit` / `qrisp`) |
| `surface_code.py` | Circuit generators: `make_stim_circuit`, `make_qiskit_circuit` (Qiskit track), `make_qrisp_experiment` (Qrisp track) |
| `noise_models.py` | The decoder's noise model (`--noise`) and the calibration snapshot format |
| `decoding.py` | Decoders (`--decoder`) and P\_error |
| `run_on_hardware.py` | Connecting to Resonance, calibration snapshot, the Qrisp hardware runner, `decode_hardware_results` for any saved run |
| `garnet_layout.py` | Qrisp track: calibration parser, zero-SWAP placement search, pinning pass |
| `stim_to_qiskit.py`, `extract_syndromes.py`, `internal_helpers.py` | Repo helpers used by the Qiskit track |
| `diagnose_run.py` | After a run: detector firing per round and which error type fits |
| `test_pipeline.py`, `check_surface_code.py` | Self-tests, no QPU time (both tracks; the second is the Qrisp branch's own) |
| `sweep.py`, `decoders.py` | The Qrisp branch's own sweep tool, kept working |

## Two tracks

Both tracks build the same d=3 rotated surface code without reset, on the same 17 Garnet qubits; they differ only in steps 1–3 below. Steps 4–7 are shared.

|  | `--track qiskit` (default) | `--track qrisp` |
| --- | --- | --- |
| Circuit | Stim's standard circuit (CX + H), converted to Qiskit | Native CZ + one-pulse SQRT\_Y gates, tracking each data qubit's rotation |
| Placement | IQM's Qiskit transpiler finds a SWAP-free placement | Calibration-ranked search over every SWAP-free placement, pinned by hand |
| Between rounds | Compiler may merge gates across a readout | Barrier after every round's readout |
| Detection events | Stim's detector rules applied to the bits | Qrisp's post-processing of its parity checks |

The Stim **twin** of each track's circuit (noiseless, with the detector rules) is what steps 5–6 use; each track supplies its own, so the detector numbering always matches.

## 1. Build the circuit in Stim

`make_stim_circuit(distance, rounds, basis, no_reset, noise)` in `surface_code.py`

*Qrisp track:* `make_qrisp_experiment(distance, rounds, basis)` builds the native circuit and its detectors in one go, replacing steps 1–2; steps 1–3 below describe the Qiskit track.

- **In:** code size (3), number of rounds, `"Z"` or `"X"` memory, `no_reset=True`, noise (`None` for hardware, `DEFAULT_NOISE` for the decoder).
- **Does:** takes Stim's built-in rotated surface code, swaps every ancilla `MR` for `M` (Garnet and Emerald can't reset mid-circuit), and rewrites every `DETECTOR` so it still means "did this check change?" (each reading becomes this reading XOR the previous one).
- **Out:** a `stim.Circuit`: the gates, plus the `DETECTOR` and `OBSERVABLE_INCLUDE` rules.
- **Next:** the noiseless copy goes to step 2; the same circuit is reused in step 5 (detector rules) and a noisy copy in step 6 (decoder map).

## 2. Convert to Qiskit

`make_qiskit_circuit(distance, rounds, basis, no_reset)` in `surface_code.py`, using `stim_to_qiskit.py`

- **In:** the same settings as step 1.
- **Does:** builds the noiseless Stim circuit, converts it gate by gate (`H`, `CX`, `M`; `RX` = reset + H, `MX` = H + measure), then deletes the `reset` lines (they only appear at the start, and qubits already start in 0).
- **Out:** a `qiskit.QuantumCircuit` with 17 qubits and one classical bit per measurement, in the same order as Stim's measurements.
- **Next:** step 3. The matching measurement order is what lets step 5 apply Stim's rules to Qiskit's bits.

## 3. Transpile onto the chip

`transpile(qc, backend, optimization_level=3, routing_method="none")`, in `QiskitTrack.run` (`tracks.py`)

- **In:** the Qiskit circuit and the backend (real Garnet or `IQMFakeGarnet`).
- **Does:** rewrites `h`/`cx` into the chip's native gates (`r`, `cz`) and assigns each circuit qubit to a physical qubit. `routing_method="none"` forbids SWAPs: if no SWAP-free placement existed, transpile would raise instead of adding them.
- **Out:** the transpiled circuit. It prints `CX=72 -> CZ=72`; equal counts confirm it.
- **Qrisp track:** `garnet_layout.rank_placements` scores every SWAP-free placement with live calibration and pins the best one; its pass raises on any non-native CZ.
- **Next:** step 4.

## 4. Run it

`track.run(...)` in `tracks.py` (Qrisp on hardware: `run_qrisp_batch` in `run_on_hardware.py`)

- **In:** one circuit per round count, a number of shots (e.g. 2,000), optional dynamical decoupling (`--dd`).
- **Does:** submits **all round counts as one job**, so they run back to back on the same calibration, then waits.
- **Out:** `counts` per circuit, a dictionary `{bitstring: times seen}`. Each bitstring holds every measurement of one shot, e.g. 33 bits for 3 rounds (8 ancillas × 3 + 9 data).
- **Next:** step 5. The raw counts are saved as soon as the job returns, so you can re-decode later without new QPU time.

## 5. Counts to detection events

Qiskit track: `counts_to_measurement_array()` then `extract_syndromes()`; Qrisp track: Qrisp's post-processing. Both in `tracks.py`.

- **In:** the counts, and the Stim circuit from step 1 (built with the same settings as the one that ran).
- **Does:** turns counts into a table (one row per shot, one column per measurement, Qiskit's reversed bit order fixed), then applies every `DETECTOR` and `OBSERVABLE_INCLUDE` rule: each is an XOR of a few bits.
- **Out:**
  - `det_events`: shots × detectors, `True` = that check reported a change (a hint that an error happened nearby).
  - `obs_flips`: shots × 1, `True` = the stored logical bit came out flipped.
- **Next:** step 6.

## 6. Decode with PyMatching

`noise_models.add_noise` then `decoding.logical_error`, called by `run_experiment.py` (`--noise`, `--decoder`)

- **In:** `det_events` and `obs_flips` from step 5, and the track's noiseless twin.
- **Does:**
  1. **Noise model** (`--noise`) adds the decoder's guess of the noise to a copy of the twin:
     - `uniform` (default): every gate, readout and idle period fails with 0.3%; `uniform:0.005` sets your own value.
     - `calibration`: per qubit, from the run's calibration snapshot: CZ and PRX errors, readout error, readout disturbance (1 − QNDness), and decay plus dephasing of idle qubits during each readout (echo T2 when `--dd` is on). Readout time is assumed to be 1.5 µs.
  2. Stim turns that copy into the detector error model (DEM): every possible error, how likely it is, which detectors it lights up, whether it flips the logical bit.
  3. **Decoder** (`--decoder`, default `pymatching`) predicts, per shot, whether the logical bit flipped. A shot is a logical error when the prediction disagrees with `obs_flips`.
- **Out:** `P_error` and its error bar, √(P(1−P)/shots), plus how often each detector fired.
- **Add your own:** a noise model is one `@register("name")` function in `noise_models.py`; a decoder is one `@register("name")` function in `decoding.py` taking `(det_events, dem)` and returning predictions, or any `module:function` passed to `--decoder`.
- **`qubit_calibrated`:** same matching, but each physical qubit is weighted from that run's calibration (`prepare_twin` in `decoding.py`). Re-decode saved shots with `python run_experiment.py --load results/<run>.json --decoder qubit_calibrated`. On the 17:47 shots it moved Z from 12.6% to 10.9% per round and X from 9.9% to 9.6%. With decoupling already on it stayed at 7.9% and 7.3%, so it does not replace the decoupling number.
- **Next:** step 7.

## 7. Repeat over rounds and fit the logical error rate

`run_experiment.py`

- **In:** a list of round counts (e.g. 1 2 3 4 5).
- **Does:** runs steps 1–6 for all of them (one job), saving as it goes, then fits P\_error against rounds two ways. The README fit assumes every round costs the same:

```latex
P_\text{error}(n) = \tfrac{1}{2}\left[1 - (1 - 2\varepsilon_L)^n\right]
```

- The amplitude fit adds a factor A for the one-off cost of state preparation and final readout, so ε\_L is the cost of each extra round. On Garnet round 1 is far cheaper than later rounds, so this is the number to report. It needs at least 3 round counts:

```latex
P_\text{error}(n) = \tfrac{1}{2}\left[1 - A\,(1 - 2\varepsilon_L)^n\right]
```

- **Out:**
  - ε\_L from both fits, with uncertainties.
  - `results/<name>.json`: settings (track, noise, decoder, DD), the calibration, and per round count: raw counts, P\_error, detector firing rate, which physical qubits were used.
  - `results/<name>_calibration.json`: the chip's error rates when the run started.
  - `results/<name>.png`: P\_error vs rounds with both fits.
- **Next:** `diagnose_run.py` on the JSON shows where the errors come from (either track).

## Results

On real Garnet, DD lowers the per-round logical error by about 30% in both tracks; even with DD each round after the first costs about 9–14%, against 3% on the simulator.

&#91;embedded content: results/\*.json · fake Garnet 15:37, Garnet no DD 17:14, Garnet with DD 17:12, Oct 24 2026\]

Every run had 0 SWAPs. Each pair of runs was taken back to back on the same qubits. Round 1 is far cheaper than later rounds, so the README fit (one rate for every round) understates the cost of a round: it gives 5.6% → 4.3% for these runs. The amplitude fit separates the one-off first-round cost; with only 3 round counts per run, treat its error bars as optimistic.

The Qrisp track (teammate's branch, 17:47 and 17:50, 5,000 shots, rounds 1, 2, 3, 4, 6, 8) confirms it with more data:

| Qrisp track, amplitude fit | No DD | With DD | Change |
| --- | --- | --- | --- |
| Z-memory ε\_L per round | 12.6% ± 0.3% | 8.6% ± 0.2% | −32% |
| X-memory ε\_L per round | 11.0% ± 0.2% | 9.0% ± 0.2% | −18% |

The tracks differ at round 1 (Qiskit 1.4% vs Qrisp 13.3% P\_error, no DD) but the Qrisp circuit adds less error per later round. They were run 30 minutes apart, so a back-to-back `--track` comparison is still needed before calling either circuit better.

## What we learned

On real Garnet the code works for 1 round, then breaks at the first mid-circuit readout; DD cuts the per-round error by about 30% but doesn't fix it.

- **The pipeline is correct.** A noiseless simulation gives 0 detections and 0 logical errors for Z and X memory on both tracks, and fake Garnet behaves like a working code.
- **Zero SWAPs.** Qiskit track: the transpiler is not allowed to add SWAPs (`routing_method="none"`) and CX count = CZ count. Qrisp track: the circuit is pinned by hand and the compiler pass raises on any non-native CZ. Garnet has exactly one 17-qubit patch that fits the code (it skips QB3, QB17, QB18).
- **Round 1 works on hardware:** P\_error 1.1–1.4% on the Qiskit track, and decoding beats no correction about 6×.
- **From round 2 the checks fire 22–37% of the time** (round 1: about 7%). Error correction needs a few percent. Round 2 is the first round that follows a mid-circuit readout.
- **Why the readout hurts:**
  1. Readout takes about 1–2 µs; data qubits idle and lose phase (Ramsey T2 only 5–10 µs on Garnet).
  2. Measuring a checker disturbs it 2–5% of the time (IQM readout QNDness 0.95–0.98), and with no reset the error carries forward.
  3. Readout reports the wrong bit 2–5% of the time.
- **Report the amplitude fit, not the README fit.** Round 1 is nearly free, so one rate for all rounds understates each round's cost about 3× (Qiskit track, no DD: 5.6% vs 20.4% per round).
- **DD helps but doesn't fix it.** Back to back on the same qubits: Qiskit track 20.4% → 14.3% per round, Qrisp track 12.6% → 8.6% (Z) and 11.0% → 9.0% (X). Round 1 is unchanged, as expected.
- **A better decoder can't rescue it.** Re-decoding with a fitted readout-idle noise model made P\_error slightly worse, and the Qrisp branch's "garnet\_uniform" weights also made it worse: the data is above the code's threshold.
- **Open question:** Z-checks fire about as often as X-checks on the Qiskit track, though idle dephasing should mostly trip X-checks. Likely the transpiler leaves some data qubits rotated while they idle; the Qrisp track tracks those rotations explicitly.

## Next steps

1. **Back-to-back track comparison:** `--track qiskit` then `--track qrisp`, same rounds, shots and DD setting, a few minutes apart. Tells whether the native-CZ circuit with barriers really adds less error per round.
2. **X-memory on Garnet** on the Qiskit track (`--basis X`), with and without DD.
3. **Calibration-weighted decoding:** run with `--noise calibration` and compare against `--noise uniform` on the same shots (re-decoding a saved file still needs a small `--load` option).
4. **Tiny diagnostic circuits:** one qubit measured twice (readout disturbance), and one data qubit idling while a neighbour is read out (idle decay).
5. **New decoders:** e.g. the union-find decoder from the `error_testing` branch, registered in `decoding.py`.
6. **Ask IQM mentors** whether mid-circuit reset or feedback is available; it would remove the no-reset problem.
7. **Emerald when it's back:** more qubits to choose from, and a chip-to-chip comparison.

## Commands

Set the token once per terminal (PowerShell: `$env:IQM_TOKEN = "..."`), never in a `.py` file.

In Command Prompt / Anaconda Prompt use `set IQM_TOKEN=...` with no quotes. Don't also pass the token in code: IQM's client refuses a token given both ways.

| Command | Uses QPU time? | What it does |
| --- | --- | --- |
| `python test_pipeline.py` | No | Checks both tracks: circuit, conversion, zero SWAPs, noise models, decoders |
| `python check_surface_code.py` | No | The Qrisp branch's self-test of its generator and Garnet placement |
| `python run_experiment.py --fake --rounds 1 2 3` | No | Full pipeline on IQM's local fake Garnet |
| `python run_experiment.py --qc garnet --rounds 1 2 3 4 5` | Yes | Real run; all round counts in one job |
| `python diagnose_run.py results/<file>.json` | No | Detector firing per round and which error type fits |

| Option | Default | What it does |
| --- | --- | --- |
| `--track qiskit\|qrisp` | `qiskit` | How the circuit is built and compiled |
| `--noise uniform\|uniform:P\|calibration` | `uniform` | The decoder's noise model |
| `--decoder NAME\|module:function` | `pymatching` | The decoder |
| `--qc NAME` | `garnet` | Quantum computer on Resonance |
| `--fake` | off | IQM's local fake Garnet instead of hardware (free, no token) |
| `--rounds N N …` | `1 2 3 4 5` | Round counts to run |
| `--shots N` | `2000` | Shots per round count |
| `--basis Z\|X` | `Z` | Z-memory (bit flips) or X-memory (phase flips) |
| `--dd` | off | IQM's dynamical decoupling (hardware only) |
| `--distance N` | `3` | Code size; 5 needs 49 qubits (Emerald) |
| `--tag TEXT` | none | Extra label in the output file names |

The Qrisp track also needs `networkx` (`pip install -r requirements.txt` installs it).
