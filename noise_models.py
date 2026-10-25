"""
noise_models.py — the decoder's guess of how likely each error is (--noise).

The decoder needs to know which errors are likely and which are rare. We tell it by
taking the NOISELESS Stim twin of the circuit that ran (its gates + DETECTOR rules)
and adding noise annotations to a copy. Stim turns that copy into the detector error
model (DEM), which is the decoder's map. None of this touches the hardware: the real
noise comes from the chip, this is only the decoder's assumption about it.

Built-in models
---------------
  uniform          every gate, readout and idle period fails with p = 0.003
  uniform:0.005    the same with your own p
  calibration      per-qubit numbers from the run's calibration snapshot:
                     after each 1-qubit gate  DEPOLARIZE1(PRX error of that qubit)
                     after each 2-qubit gate  DEPOLARIZE2(CZ error of that pair)
                     each readout             flips with that qubit's readout error
                     after a mid-circuit readout
                       measured qubits        flip with 1 - QNDness (readout disturbs them)
                       every other qubit      idles for READOUT_TIME: decays with T1 and
                                              dephases with T2 (echo T2 when --dd is on)

Adding a model
--------------
Write a function that receives a NoiseContext and returns a Rates object, and
register it under a name:

    @register("my_model")
    def my_model(ctx: NoiseContext) -> Rates:
        return Rates(gate1=lambda q: 0.001, gate2=lambda a, b: 0.01, readout=lambda q: 0.02)

Then run with --noise my_model (or --noise my_model:<arg>, which arrives as ctx.arg).
Every probability function receives qubit indices of the Stim twin; ctx.physical(q)
gives the physical qubit name ("QB7") it ran on.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import stim

# Nominal Garnet readout duration in seconds (same assumption as garnet_layout.py).
# Not calibrated: replace with the real value if IQM gives it to you.
READOUT_TIME = 1.5e-6

_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}
_MEASUREMENTS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
_RESETS = {"R", "RX", "RY"}


@dataclass
class NoiseContext:
    """Everything a noise model may use."""
    arg: Optional[str] = None            # the text after ':' in --noise name:arg
    calibration: Optional[dict] = None   # calibration snapshot (calibration_to_json format)
    qubit_map: dict = field(default_factory=dict)  # twin qubit index -> physical name ("QB7")
    dd: bool = False                     # dynamical decoupling was on

    def physical(self, q: int) -> Optional[str]:
        return self.qubit_map.get(q)


@dataclass
class Rates:
    """Error probabilities per place in the circuit. Any field left out = no noise there."""
    gate1: Callable[[int], float] = lambda q: 0.0                 # after a 1-qubit gate
    gate2: Callable[[int, int], float] = lambda a, b: 0.0         # after a 2-qubit gate
    readout: Callable[[int], float] = lambda q: 0.0               # wrong measurement result
    reset: Callable[[int], float] = lambda q: 0.0                 # after a reset
    after_readout: Callable[[int], float] = lambda q: 0.0         # measured qubit disturbed (mid-circuit)
    idle: Callable[[int], tuple] = lambda q: (0.0, 0.0, 0.0)      # (px, py, pz) while others are read out


NOISE_MODELS: dict = {}


def register(name: str):
    """Decorator: make a noise model available as --noise <name>."""
    def wrap(fn):
        NOISE_MODELS[name] = fn
        return fn
    return wrap


def parse_spec(spec: str) -> tuple:
    """'uniform:0.005' -> ('uniform', '0.005'); 'calibration' -> ('calibration', None)."""
    name, _, arg = spec.partition(":")
    if name not in NOISE_MODELS:
        raise ValueError(f"unknown noise model '{name}'. Available: {', '.join(sorted(NOISE_MODELS))}")
    return name, (arg or None)


def add_noise(twin: stim.Circuit, spec: str, ctx: NoiseContext) -> stim.Circuit:
    """Return a copy of the noiseless twin with the noise of model `spec` added."""
    name, arg = parse_spec(spec)
    ctx.arg = arg
    return annotate(twin, NOISE_MODELS[name](ctx))


def annotate(twin: stim.Circuit, rates: Rates) -> stim.Circuit:
    """Insert noise channels into a noiseless circuit according to `rates`.

    Measurements are split into one instruction per qubit (so each can have its own
    error rate); their order is unchanged, so every DETECTOR's rec[-k] still points at
    the same measurement.
    """
    flat = twin.flattened()
    used = sorted({t.value for ins in flat for t in ins.targets_copy() if t.is_qubit_target})
    meas_positions = [i for i, ins in enumerate(flat) if ins.name in _MEASUREMENTS]
    last_meas = meas_positions[-1] if meas_positions else -1

    out = stim.Circuit()
    for i, ins in enumerate(flat):
        name = ins.name
        targets = ins.targets_copy()
        if name in _ANNOTATIONS:
            out.append(ins)
            continue
        data = stim.gate_data(name)
        if data.is_noisy_gate and name not in _MEASUREMENTS:
            continue                                   # drop any noise already in the twin

        if name in _MEASUREMENTS:
            measured = [t.value for t in targets if t.is_qubit_target]
            for t in targets:
                p = _clip(rates.readout(t.value)) if t.is_qubit_target else 0.0
                out.append(name, [t], [p] if p > 0 else [])
            if i != last_meas:                         # mid-circuit readout
                flip = "Z_ERROR" if name in ("MX", "MRX") else "X_ERROR"
                for q in measured:
                    p = _clip(rates.after_readout(q))
                    if p > 0:
                        out.append(flip, [q], p)
                others = [q for q in used if q not in set(measured)]
                for q in others:
                    px, py, pz = (_clip(v) for v in rates.idle(q))
                    if px + py + pz > 0:
                        out.append("PAULI_CHANNEL_1", [q], [px, py, pz])
            continue

        out.append(ins)
        if name in _RESETS:
            flip = "Z_ERROR" if name == "RX" else "X_ERROR"
            for t in targets:
                p = _clip(rates.reset(t.value))
                if p > 0:
                    out.append(flip, [t.value], p)
        elif data.is_unitary and data.is_single_qubit_gate:
            for t in targets:
                p = _clip(rates.gate1(t.value))
                if p > 0:
                    out.append("DEPOLARIZE1", [t.value], p)
        elif data.is_unitary and data.is_two_qubit_gate:
            qs = [t.value for t in targets]
            for a, b in zip(qs[0::2], qs[1::2]):
                p = _clip(rates.gate2(a, b))
                if p > 0:
                    out.append("DEPOLARIZE2", [a, b], p)
    return out


def _clip(p: float) -> float:
    return min(max(float(p), 0.0), 0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in models
# ─────────────────────────────────────────────────────────────────────────────

@register("uniform")
def uniform(ctx: NoiseContext) -> Rates:
    """The same probability p everywhere (default 0.003, or --noise uniform:<p>)."""
    p = float(ctx.arg) if ctx.arg else 0.003
    return Rates(
        gate1=lambda q: p,
        gate2=lambda a, b: p,
        readout=lambda q: p,
        reset=lambda q: p,
        after_readout=lambda q: p,
        idle=lambda q: (p / 3, p / 3, p / 3),
    )


@register("calibration")
def calibration(ctx: NoiseContext) -> Rates:
    """Per-qubit rates from the calibration snapshot of the run."""
    cal = ctx.calibration
    if not cal or not cal.get("prx"):
        raise ValueError("--noise calibration needs a calibration snapshot, and this run has none")

    def per_qubit(key, q, default=None):
        name = ctx.physical(q)
        table = cal.get(key) or {}
        if name in table:
            return table[name]
        values = sorted(table.values())
        return values[len(values) // 2] if values else default   # unknown qubit -> chip median

    def cz(a, b):
        pair = "__".join(sorted((str(ctx.physical(a)), str(ctx.physical(b)))))
        table = cal.get("cz") or {}
        if pair in table:
            return table[pair]
        values = sorted(table.values())
        return values[len(values) // 2] if values else 0.005

    def idle(q):
        t1 = per_qubit("t1", q, 40e-6)
        t2 = per_qubit("t2_echo" if ctx.dd else "t2", q, None) or per_qubit("t2", q, 10e-6)
        p_decay = 1.0 - math.exp(-READOUT_TIME / t1)
        p_phase = 0.5 * (1.0 - math.exp(-READOUT_TIME / t2))
        px = py = p_decay / 4
        return px, py, max(p_phase - px, 0.0)

    return Rates(
        gate1=lambda q: per_qubit("prx", q, 0.001),
        gate2=cz,
        readout=lambda q: per_qubit("readout", q, 0.02),
        reset=lambda q: 0.0,
        after_readout=lambda q: 1.0 - per_qubit("qndness", q, 1.0),
        idle=idle,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration snapshots in one format for every track and backend
# ─────────────────────────────────────────────────────────────────────────────

def calibration_to_json(cal: dict) -> dict:
    """garnet_layout.fetch_calibration() output -> plain JSON (CZ pairs as 'QB1__QB2')."""
    out = {}
    for key, value in cal.items():
        if key == "cz":
            out["cz"] = {"__".join(sorted(pair)): err for pair, err in value.items()}
        else:
            out[key] = value
    return out


def calibration_from_json(cal: dict) -> dict:
    """Inverse of calibration_to_json (CZ pairs back to frozensets), for garnet_layout."""
    out = dict(cal)
    out["cz"] = {frozenset(k.split("__")): v for k, v in (cal.get("cz") or {}).items()}
    return out


def calibration_from_fake(backend) -> dict:
    """The noise IQM's fake backend simulates, in fetch_calibration() format (seconds, errors)."""
    ep = backend.error_profile
    qubits = sorted(ep.t1s, key=lambda q: int(q[2:]))
    readout = {q: (v["0"] + v["1"]) / 2 for q, v in ep.readout_errors.items()}
    cz = {}
    for gate, pairs in ep.two_qubit_gate_depolarizing_error_parameters.items():
        for pair, err in pairs.items():
            cz[frozenset(pair)] = err
    prx = {}
    for gate, table in ep.single_qubit_gate_depolarizing_error_parameters.items():
        prx.update(table)
    return {
        "qubits": qubits,
        "cz": cz,
        "prx": prx,
        "readout": readout,
        "e01": {q: v["0"] for q, v in ep.readout_errors.items()},
        "e10": {q: v["1"] for q, v in ep.readout_errors.items()},
        "leakage": {q: 0.0 for q in qubits},
        "t1": {q: v * 1e-9 for q, v in ep.t1s.items()},
        "t2": {q: v * 1e-9 for q, v in ep.t2s.items()},
        "t2_echo": {q: v * 1e-9 for q, v in ep.t2s.items()},   # the fake backend has no echo T2
        "calibration_set_id": "fake",
        "quantum_computer": ep.name,
    }


def calibration_from_legacy(snapshot: dict) -> dict:
    """Calibration files saved before the merge (iqm-qubit-selector "fidelities" format) ->
    the current format. Approximate: that helper's readout number can be the heralded one,
    and its T2 may be Ramsey or echo (see garnet_layout.py), so treat results as a rough guide."""
    import ast
    f = snapshot.get("fidelities") or {}
    cz = {}
    for pair, fid in (f.get("CZ") or {}).items():
        a, b = ast.literal_eval(pair)
        cz["__".join(sorted((a, b)))] = 1.0 - fid
    out = {
        "source": "legacy_qubit_selector",
        "prx": {q: 1.0 - v for q, v in (f.get("1Q") or {}).items()},
        "readout": {q: 1.0 - v for q, v in (f.get("readout") or {}).items()},
        "qndness": dict(f.get("readout_qndness") or {}),
        "t1": {q: v * 1e-6 for q, v in (f.get("t1") or {}).items()},     # microseconds -> seconds
        "t2": {q: v * 1e-6 for q, v in (f.get("t2") or {}).items()},
        "cz": cz,
        "calibration_set_id": snapshot.get("calibration_set_id"),
    }
    out["t2_echo"] = dict(out["t2"])
    out["qubits"] = sorted(out["t1"], key=lambda q: int(q[2:]))
    return out
