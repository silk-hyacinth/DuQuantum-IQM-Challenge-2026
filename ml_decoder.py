"""
ml_decoder.py: exact maximum-likelihood (ML) decoder for small codes (d=3).

What it computes
----------------
The detector error model (DEM) is a list of independent error mechanisms:
"with probability p_i, flip detectors D_i and (maybe) the logical observable".
For one shot with detector pattern s, the ML decoder adds up the probability of
EVERY combination of mechanisms that produces s, separately for "logical bit
flipped" and "not flipped", and predicts whichever total is larger:

    predict flip  <=>  P(s, flip) > P(s, no flip)

PyMatching only follows the single most likely error chain. ML counts all of
them, so it is the best any decoder can do FOR THIS ERROR MODEL. If ML is not
much better than PyMatching, no better decoder (union-find, BP, ...) will help.

How it stays exact without 2^(#mechanisms) work
-----------------------------------------------
Sweep the mechanisms in time order and keep a table w[f, o]: the total
probability of all subsets of mechanisms so far whose partial detector pattern
on the "live" detectors is f and whose observable flip is o. A detector is live
from the first mechanism that touches it to the last. When a detector retires
we keep only the half of the table that matches what this shot observed, and
that bit drops out. The table size is 2^(live detectors + 1), a few hundred
thousand entries for d=3 instead of 2^(hundreds).

Usage with this repo (no edits to decoding.py needed)
-----------------------------------------------------
    python run_experiment.py ... --decoder ml_decoder:ml_decoder      # uses --noise
    python run_experiment.py ... --decoder ml_decoder:ml_calibrated   # per-qubit calibration weights
    python run_experiment.py ... --load results/<run>.json --decoder ml_decoder:ml_calibrated

IMPORTANT: ML is only optimal for the error model it is given. With the default
uniform 0.3% model it is the best decoder for a chip that is NOT uniform 0.3%.
To find a ceiling, run it with the most realistic weights you have.
"""

import warnings

import numpy as np
import stim

# Memory budget: elements per working array. The decoder runs several shots at
# once, as many as fit. 2**23 float64 = 64 MB per array, about 4 arrays alive.
DEFAULT_MAX_ELEMENTS = 2 ** 23
# Refuse to build tables bigger than 2^MAX_BITS entries for a single shot.
MAX_BITS = 26


# ─────────────────────────────────────────────────────────────────────────────
#  1. Read the DEM into a plain list of independent mechanisms
# ─────────────────────────────────────────────────────────────────────────────

def dem_mechanisms(dem: stim.DetectorErrorModel):
    """DEM -> [(p, (detector ids...), observable_flip 0/1)], one entry per distinct effect.

    * "A B ^ C D" (decomposed errors) is ONE physical error that flips A, B, C, D,
      so separators are ignored here. (The decomposition only helps matching.)
    * Mechanisms with identical effects are merged: p = p1(1-p2) + p2(1-p1).
    * Mechanisms with no effect at all are dropped.
    """
    merged: dict = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        dets: set = set()
        obs = 0
        for t in inst.targets_copy():
            if t.is_separator():
                continue
            if t.is_relative_detector_id():
                dets ^= {t.val}              # a detector named twice cancels
            elif t.is_logical_observable_id():
                if t.val != 0:
                    raise ValueError("ml_decoder supports a single logical observable")
                obs ^= 1
        key = (tuple(sorted(dets)), obs)
        if not key[0] and not key[1]:
            continue
        q = merged.get(key, 0.0)
        merged[key] = q * (1 - p) + p * (1 - q)
    return [(p, d, o) for (d, o), p in merged.items() if p > 0.0]


# ─────────────────────────────────────────────────────────────────────────────
#  2. Order the mechanisms so few detectors are live at once
# ─────────────────────────────────────────────────────────────────────────────

def _profile(mechs):
    """(max live detectors, total work) for mechanisms in this order."""
    first, last = {}, {}
    for k, (_, dets, _) in enumerate(mechs):
        for d in dets:
            first.setdefault(d, k)
            last[d] = k
    delta = np.zeros(len(mechs) + 1, dtype=int)
    for d in first:
        delta[first[d]] += 1
        delta[last[d] + 1] -= 1
    live = np.cumsum(delta)[:-1]
    return int(live.max()), float(np.sum(2.0 ** (live + 1)))


def _greedy_order(mechs):
    """Pick mechanisms one at a time, each time the one that keeps the live set smallest.

    Live = detectors already touched but still needed by a later mechanism.
    """
    remaining = {}
    for _, dets, _ in mechs:
        for d in dets:
            remaining[d] = remaining.get(d, 0) + 1
    left = set(range(len(mechs)))
    live: set = set()
    out = []
    while left:
        best = None
        for i in left:
            dets = mechs[i][1]
            grown = len(live | set(dets))
            retiring = sum(1 for d in dets if remaining[d] == 1)
            key = (grown, grown - retiring, dets[0] if dets else -1, i)
            if best is None or key < best:
                best = key
        i = best[-1]
        dets = mechs[i][1]
        live |= set(dets)
        for d in dets:
            remaining[d] -= 1
            if remaining[d] == 0:
                live.discard(d)
        out.append(mechs[i])
        left.discard(i)
    return out


def order_mechanisms(mechs):
    """Try a few sweep orders and keep the one with the least work.

    Returns (ordered mechanisms, max live detectors). The table has
    2^(live + 1) entries, so every live detector saved halves the run time.
    """
    def first(m):
        return m[1][0] if m[1] else -1

    def last(m):
        return m[1][-1] if m[1] else -1

    candidates = [
        sorted(mechs, key=lambda m: (first(m), last(m))),
        sorted(mechs, key=lambda m: (last(m), first(m))),
        sorted(mechs, key=lambda m: (-last(m), -first(m))),
        _greedy_order(mechs),
    ]
    best = None
    for ordered in candidates:
        width, work = _profile(ordered)
        if best is None or work < best[1]:
            best = (width, work, ordered)
    return best[2], best[0]


# ─────────────────────────────────────────────────────────────────────────────
#  3. The exact sweep, for a batch of shots at once
# ─────────────────────────────────────────────────────────────────────────────

def _sweep(batch: np.ndarray, ordered, first_at, last_at) -> np.ndarray:
    """Return, per shot, the table [P(s, no flip), P(s, flip)] (rescaled per shot).

    Bit 0 of a table index is the observable; bit j >= 1 is detector slots[j-1].
    """
    B = batch.shape[0]
    rows = np.arange(B)
    state = np.zeros((B, 2))
    state[:, 0] = 1.0
    slots: list = []

    for k, (p, dets, obs) in enumerate(ordered):
        # detectors first touched here join the table with partial value 0
        for d in first_at.get(k, ()):
            state = np.concatenate([state, np.zeros_like(state)], axis=1)
            slots.append(d)

        # apply the mechanism: new[i] = (1-p) * old[i] + p * old[i XOR mask]
        mask = obs
        for d in dets:
            mask |= 1 << (slots.index(d) + 1)
        idx = np.arange(state.shape[1]) ^ mask
        flipped = np.take(state, idx, axis=1)
        flipped *= p
        state *= (1.0 - p)
        state += flipped

        # detectors with no later mechanism retire: keep the observed half
        for d in last_at.get(k, ()):
            j = slots.index(d) + 1
            nbits = len(slots) + 1
            cube = state.reshape(B, 2 ** (nbits - j - 1), 2, 2 ** j)
            state = np.ascontiguousarray(cube[rows, :, batch[:, d].astype(np.intp), :])
            state = state.reshape(B, -1)
            slots.pop(j - 1)
        if last_at.get(k):
            tot = state.sum(axis=1, keepdims=True)
            state /= np.where(tot > 0, tot, 1.0)     # keep numbers in range

    assert state.shape[1] == 2, "internal error: a detector never retired"
    return state


def ml_decoder(det_events: np.ndarray, dem: stim.DetectorErrorModel,
               max_elements: int = DEFAULT_MAX_ELEMENTS, verbose: bool = False) -> np.ndarray:
    """Exact ML decoder. Same shape as every decoder in decoding.py:
    (det_events[shots, detectors] bool, dem) -> predictions[shots, 1] bool."""
    det_events = np.asarray(det_events, dtype=bool)
    shots, n_det = det_events.shape
    if dem.num_observables != 1:
        raise ValueError("ml_decoder supports a single logical observable")

    mechs = dem_mechanisms(dem)
    ordered, width = order_mechanisms(mechs)
    if width + 1 > MAX_BITS:
        raise ValueError(f"{width} detectors live at once means tables of 2^{width + 1} entries; "
                         f"too big for exact ML. Use fewer rounds, or a different decoder.")

    first_at: dict = {}
    last_at: dict = {}
    seen_first, seen_last = {}, {}
    for k, (_, dets, _) in enumerate(ordered):
        for d in dets:
            seen_first.setdefault(d, k)
            seen_last[d] = k
    for d, k in seen_first.items():
        first_at.setdefault(k, []).append(d)
    for d, k in seen_last.items():
        last_at.setdefault(k, []).append(d)

    # A detector that no mechanism touches can never fire under this model.
    untouched = np.array([d not in seen_first for d in range(n_det)], dtype=bool)
    impossible = det_events[:, untouched].any(axis=1) if untouched.any() else np.zeros(shots, bool)
    if impossible.any():
        warnings.warn(f"{int(impossible.sum())} shots fire a detector the model says can never fire; "
                      f"they are decoded as 'no flip'.")

    # identical syndromes give identical answers: decode each distinct one once
    uniq, inverse = np.unique(det_events, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    per_batch = max(1, max_elements // 2 ** (width + 1))
    if verbose:
        print(f"[ml_decoder] {len(mechs)} mechanisms, up to {width} live detectors "
              f"(tables of 2^{width + 1}), {len(uniq)} distinct syndromes of {shots} shots, "
              f"{per_batch} shots per batch")

    flip = np.zeros(len(uniq), dtype=bool)
    for start in range(0, len(uniq), per_batch):
        batch = uniq[start:start + per_batch]
        table = _sweep(batch, ordered, first_at, last_at)
        flip[start:start + len(batch)] = table[:, 1] > table[:, 0]
    predictions = flip[inverse]
    predictions[impossible] = False
    return predictions.reshape(shots, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Same decoder, but with per-qubit calibration weights (like qubit_calibrated)
# ─────────────────────────────────────────────────────────────────────────────

def ml_calibrated(det_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Exact ML on the DEM that decoding.qubit_calibrated_circuit builds."""
    return ml_decoder(det_events, dem)


try:                                    # run_experiment.py looks for this attribute
    from decoding import qubit_calibrated_circuit
    ml_calibrated.prepare_twin = qubit_calibrated_circuit
except ImportError:                     # file used outside the repo
    pass
