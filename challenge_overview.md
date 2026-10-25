# Surface Code on IQM Emerald — Challenge Overview

## Quantum Error correction - Memory experiment

Quantum computers are fundamentally noise-limited: every gate, every idle moment, every measurement introduces errors that accumulate faster than useful computation can proceed. 
Quantum error correction addresses this by encoding a single logical qubit across many physical qubits and continuously measuring stabilizer operators — parity checks that reveal error syndromes without collapsing the encoded information. 
Surface codes have emerged as the leading practical approach.
A surface code is defined, via the number of physical qubits it needs, the distance it covers, and the number of logical qubits.
They map directly onto the square-lattice connectivity that superconducting processors like IQM Emerald provide.
Still, writing out a whole a whole pipeline is notoriously hard. 
The challenge you will be facing today will have you operating on the forefront of science! 

## The QEC Pipeline

You are building and completing a proof-of-concept quantum error correction pipeline that runs a surface code on real superconducting hardware and produces a logical error rate. There are two equal ways of implementing the pipeline, Qiskit and Qrisp and you should pick one. The full chain is:

```
Qiskit :  (Stim circuit  →)  Qiskit  →  IQM Emerald (Resonance)  →  Syndrome extraction  →  Decoder  →   P_L(rounds) → fit → LER
Qrisp  :  (Stim circuit  →)  Qrisp DetectorExperiment  →  IQM Emerald (Resonance)  →  Decoder  →   P_L(rounds) → fit → LER
```

Each stage has a clear role. 

* Stim, Qiskit or Qrisp defines the surface code circuit including stabilizer structure, qubit coordinates, detectors, and the logical observable. Stim is an optional package tool, specialized for creating surface code experiments. While you cannot directly run the Stim circuits on the hardware, it makes postprocessing a lot easier. 
* The Qiskit or Qrisp layer translates that abstract circuit into something the hardware can execute. You can also build the surface code experiment in this layer of the stack. 
* IQM Resonance runs the transpiled circuit and returns shot-by-shot measurement bitstrings. 
* Syndrome extraction converts those raw bitstrings into detection events, which are what the decoder actually needs, not the raw measurements themselves. If you write your code in Qrisp, the `DetectorExperiment` returns detection events directly.
* The decoder then uses those events to infer whether a logical error occurred.

In real hardware, you would then feed the information on logical errors back into the QPU and correct errors on the fly. But real hardware is limited, so this is something we cant do (yet). This is called a memory experiment.

**This pipeline is offline**. Decoding happens after all shots are collected. There is no requirement to feed correction results back to the QPU during coherence time.

---

## Judging criteria

1) Theoretical correctness. Does the experiment actually set up a surface correctly? Is the pipeline fully functional? (30%)
2) Sophistication of the implementation. Have you optimized your circuits for the performance/architecture of the quantum computer? Can you prove a low number of SWAPS? (30%)
3) Logical error rate statistics. (20%)
4) Flexibility of the experiment and integration of advanced functionalities - Circuit compilation techniques, Noise Model improvements, Pulse-level compilations,... (20%)


Bonus points will be awarded (at our discretion) for:
- Hardware flexibility.
- Integration of advanced functionalities.
- Experimentally identifying the most important reasons for improving LER.
- Other particularly cool and surprising ideas

## Submissions

Submit your code and any other key results through [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSeKTmoGUzxhrc4bPtLJ5Vp3VtoFSrtoes1UP3_6IoLgZDLIxg/viewform?usp=publish-editor). 


# Useful Resources

**Please see the ``UsefulResources.md`` markdown for an overview over useful resources!** Such as, links to **IQM PulLA**, client-side libraries such as **Dynamical Decoupling**, access links to IQM Resonance!
**To access PulLA you need to be a part of a paid organisation, please approach your mentors about this!**

### Related papers

* [Quantum error correction below the surface code threshold](https://www.nature.com/articles/s41586-024-08449-y) (Google Quantum AI, Nature 2024): the current state of the art for what you are doing in this challenge. Too long to read in full during the event, but some details may help you.
* [Suppressing quantum errors by scaling a surface code logical qubit](https://www.nature.com/articles/s41586-022-05434-1) (Google Quantum AI, Nature 2023): older, more basic and clearer. Fig. 3 shows the logical error probability vs. number of rounds, which is the plot you should produce (see section 2 below).
* [Exponential suppression of bit or phase errors with cyclic error correction](https://www.nature.com/articles/s41586-021-03588-y#Sec4) (Google Quantum AI, Nature 2021): source of the fit function used to extract the logical error rate.




---

## Provided functionality

**All the provided functionality is optional!** Should you wish to build from scratch, feel free to do so! 

Install everything you should need with the requirements file of this repo: `pip install -r requirements.txt` (Python ≥ 3.11).

### Circuit translation (`stim_to_qiskit.py`, `stim_to_qrisp.py`)

Should you wish to use Stim, the full Stim-to-Qiskit and Stim-to-Qrisp translation layer is implemented. Functionality of `stim_to_qiskit`:

* Stim uses sparse, non-contiguous qubit indices whereas Qiskit requires a contiguous `0..N-1` range, so the converter builds a dense re-indexing map. 
* strips all noise and annotation instructions (which exist only in Stim's simulation model and have no hardware equivalent), 
* handles mid-circuit reset by translating Stim's `MR` instruction into a measure-then-reset pair (only relevant for with-reset circuits, since our Emerald cloud quantum computer does not support mid-circuit reset), 
* preserves measurement order exactly.

`stim_to_detector_experiment` (`stim_to_qrisp.py`) translates a Stim circuit such that it can be embedded as a Qrisp `DetectorExperiment`, including detectors and observables. With `no_reset=True` it disregards the mid-circuit resets and rewrites the detectors accordingly (please verify!).

Neither converter handles X-basis gates (`RX`, `MX`). Extending them can be understood as a part of the challenge.

### Simulation baseline (`raw_stim_d3_surface_code.py`)

We provide a d=3 rotated surface code memory experiment in Stim. It models the hardware with uniform noise at circuit level and decodes the syndromes with PyMatching. You can use to first familiarize yourself with how an experiment is written, how to set up decoding and get to an expected LER. The memory experiment uses a with-reset reference: the hardware runs the no-reset variant, so its LER is not directly comparable to your hardware results.

### Syndrome extraction (`extract_syndromes.py`)

This only needed for the Qiskit implementation. The full syndrome extraction step is implemented using Stim's ``compile_m2d_converter()``, which converts raw measurement bitstrings into detection events and observable flips in a single call. If you are working within a pure Qiskit workflow, you can replace this with a manual implementation: detection events are the XOR of consecutive ancilla measurement rounds with reset. Without reset, the XOR of the measurements two rounds apart. (a stabilizer that changes between rounds indicates an error), and the logical observable is recovered from the parity of the final data qubit measurements. Either way, what matters for the decoder is the same output shape: a [shots, num_detectors]-shaped boolean array of detection events and a [shots, num_observables] array of logical outcomes. The dictionary also carries per-shot syndrome weights and per-detector firing rates as diagnostics.

### Hardware execution (*`run_on_hardware.py`*)

* `run_qiskit()`: The hardware connection is **almost** ready for you be used. It connects to IQM Resonance via `IQMProvider`, retrieves the backend, transpiles the Qiskit circuit (you should think about whether adjustments in the transpilation step are suitable), submits the job, and converts the returned Qiskit counts dictionary into the numpy array format the syndrome extractor expects. The Qiskit bitstring reversal (Qiskit is right-to-left, Stim is left-to-right) is handled in the helper `counts_to_measurement_array`.
**Attention here!** Automatic SWAP insertion for pairs that aren't natively adjacent on the hardware grid! You can cirumvent this by writing your own stim/qiskit/qrisp-to-emerald hardware mapping.

* `run_qrisp()`: Similarly, the  connects to IQM Resonance via `IQMBackend`. Running the `DetectorExperiment` and collecting the detection events is up to you. Compiler passes do some transpilation for you, but should think about whether adjustments in the transpilation step are suitable `vf2pp_layout` 

Both return the same syndromes dict, so `decode_hardware_results()` works for either implementation. Emerald does not support mid-circuit reset: use the no-reset circuit (`no_reset=True`, the default).

### Utilities (*`internal_helpers.py`*)

Three helper functions support the pipeline, mostly for stim to qiskit and postprocessing functionality.

* `get_qubit_lists` classifies qubits as data or ancilla using the coordinate parity convention Stim uses for the rotated surface code. 
* `get_meas_order` returns qubit indices in the order they are measured to align hardware bitstrings with Stim's measurement record. 
* `counts_to_measurement_array` does the Qiskit-to-numpy conversion including the endianness correction (Qiskit implementation only). 

Note the docstrings reminding you, that you should verify this logic carefully before fully trusting it.

---

## What Needs To Be Built

### 1. A way to generate the surface code circuit

The pipeline currently has no entry point. You need to build a surface code generator in the framework of your choice! 

Concretely, you need a function that accepts:

* a code distance
* a number of stabilizer rounds, 
* which memory experiment we are looking at (Z or X)
* whether ancillas are reset between rounds (`no_reset`)
* optionally a noise model, 
* and returns a `stim.Circuit` or a `qiskit.QuantumCircuit`, a Qrisp `DetectorExperiment`, or ...

`surface_code.py` has three stubs with these signatures. You can pick one starting point: `make_stim_circuit()` (shared by both tracks, needs one of the converters to run on hardware), `make_qiskit_circuit()` or `make_qrisp_experiment()`. Their docstrings list the pros and cons.

You do not have to build this in Stim or Qiskit if you prefer to use another framework! 

On Emerald, you must implement a no-reset version of the experiment: it does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant. Most surface codes default to mid-circuit resets on ancilla qubits after a measurement (`MR`), so each round starts with a fresh ancilla and each raw measurement directly gives that round's syndrome. An alternative is to omit the reset: the ancilla then accumulates across rounds, and each raw measurement is the ``XOR`` of all syndromes up to that point. The detection events mean the same in both cases, but in raw measurements a no-reset detector is the XOR of the measurements two rounds apart (m_k XOR m_(k-2)) instead of consecutive ones, so the detector definitions must be rewritten. In terms of hardware performance for the experiment, an interesting trade-off occurs: The no-reset version avoids the qubit reset error, leaving the ancilla live and decohering between rounds.

### 2. A way to decode the syndromes and compute the logical error rate

`decode_hardware_results` does not do anything in its current form. The detection events and observable flips are already correctly formatted when they arrive here — `det_events` is a `(shots, num_detectors)` boolean array, `obs_flips` is `(shots, num_observables)`. What is missing is the actual decoding step and the statistical aggregation into a logical error rate and its uncertainty. If you decode with a detector error model (DEM), its detectors must match the circuit that you executed on hardware

Options are plenty, but PyMatching (specifically the MWPM decoder) will give you the fastest out-of-the-box result. 

__Logical error probability vs. logical error rate.__ What one decoded experiment gives you is the logical error probability `P_error` after `n_rounds` rounds, not the logical error rate. It is normal for it to approach 50% after a few rounds. To get the logical error rate per round `ε_L` you need to: 

* Run the memory experiment for a number of round counts, e.g. `rounds = 2, 3, 4, …`, each as a separate job.
* Decode each one and plot the logical error probability against the number of rounds (compare Fig. 3 of the 2023 paper in "Related papers").
* Fit the related function to your data. It describes the exponential decay of the logical fidelity `F = 1 - 2 P_error` with the number of rounds. The fitted `ε_L` is your logical error rate.

$$2P_\text{error} = 1 - (1 - 2\varepsilon_L)^{n_\text{rounds}}$$

For a single round count this solves to `ε_L = ½ [1 - (1 - 2 P_error)^(1/n_rounds)]`, but the fit over many round counts is more robust.


### 3. (Optional but Recommended) A simulation baseline

`raw_stim_d3_surface_code.py` gives you a starting point, but it simulates a with-reset circuit. Before spending QPU time, it is strongly advisable to run the same circuit (i.e. your no-reset circuit) through a simulator with a noise model and validate that the pipeline is working correctly end to end. A simulation baseline also gives you an expected LER to compare against the hardware result, even though there will be big discrepancies.

### 4. (Optional) Custom hardware qubit placement

`build_emerald_qubit_map` returns an empty dictionary. The Qiskit transpiler handles qubit placement automatically and will find a valid mapping, but it does so without knowledge of the surface code's structure. The rotated surface code can be aligned to Emerald's square grid, eliminating SWAP overhead entirely. If you are using Stim, the `print_qubit_map` function will then show you the mapping and flag any non-native pairs, so you can verify before submitting. 
In Qiskit this verification is straight forward. On the Qrisp track, `vf2pp_layout` does the placement for you.

Since the code is somewhat convoluted, we again recommend extensive investigation or writing these functions yourselfes. 

---

## Extensions

### Pulse-level compilation with PulLA

The current pipeline operates at the gate level and relies on Qiskit's transpiler to map to the hardware's native gate set. A more direct path is to compile the stabilizer circuit to pulses using **PulLA**, IQM's pulse-level compilation framework. This bypasses the gate abstraction entirely and can produce significantly shallower circuits by fusing or reshaping pulses that would otherwise be separate gates. For a noise-limited experiment like this, reducing circuit depth directly translates to fewer errors before the decoder sees the syndrome, which is the most direct lever on logical error rate.
**Also activate advanced circuit compilation techniques, such as dynamical decoupling (DD), readily available within IQMs stack.**


### Different decoders and advanced functionalities

PyMatching (minimum-weight perfect matching) is the natural starting point, but it is not the only option. **Union-Find decoding** is asymptotically faster and nearly as accurate near threshold. **Belief propagation** decoders can incorporate more detailed noise information. The MWPM graph itself can be weighted by calibration data from Resonance, which typically improves decoding accuracy on real hardware. **There are many more, so feel to investigate!** 
In particular using calibration data and setting up experiments for extended noise model characterization are of very high value! These can massively improve the decoding pipeline. 

For a more substantial extension, the **NVIDIA Ising Predecoder** offers a hardware-accelerated decoding path. This route requires working through the Ising model's `MemoryCircuit` class for circuit generation (which handles native X-basis preparation and measurement and a different Stim-to-Qiskit translation), and is therefore more involved. Please approach us if you are interested! 

### Different hardware patches and code distances

The pipeline as set up targets a distance-3 rotated surface code (17 qubits) on a single 5×5 region of Emerald. Several natural extensions exist: running the same code on different 5×5 sub-regions of the chip and comparing logical error rates across regions reveals spatial variation in hardware quality. Trying distance-5 (49 qubits) would require a larger patch but fits on Emerald's full qubit count. Trying the distance-3 code on Garnet also gives you an idea on how a different QPU behaves. 

