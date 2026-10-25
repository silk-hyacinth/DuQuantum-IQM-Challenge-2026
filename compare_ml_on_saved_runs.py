"""
compare_ml_on_saved_runs.py: re-decode saved hardware shots with PyMatching and
with exact ML, using the repo's own loader and decode path. No QPU time used.

    python compare_ml_on_saved_runs.py results/garnet_qiskit_d3_dd_20261024_201754.json
    python compare_ml_on_saved_runs.py results/*.json --max-rounds 3

Four decoders, all on identical shots:

    pymatching  + uniform 0.3%       the repo default (reference)
    exact ML    + uniform 0.3%       same error model, best possible decoder
    qubit_calibrated (matching)      per-qubit weights from the run's calibration
    exact ML    + calibration        same weights, best possible decoder

Reading the result
------------------
* ML vs matching with the SAME weights shows what a better decoding algorithm
  can give you. If it is small, union-find/BP will not help either.
* Run time grows fast with rounds (the no-reset circuit makes the tables wide:
  about 2^15 entries per shot by r=4, 2^17 by r=8). By r=6 to 8 the logical bit
  is nearly a coin flip (P_error ~ 0.45), so no decoder can gain anything there.
  --max-rounds (default 4) skips them; eps_L is then fitted on the rounds kept,
  for every decoder alike, so it will differ from the number in your results file.
* ML's number is a ceiling only for that error model. If the model is wrong for
  the chip (leakage, crosstalk, drift), a model-aware decoder can still do better.
"""

import copy
import io
import sys
from contextlib import redirect_stdout

import numpy as np

import run_experiment as rx

DECODERS = [
    ("matching, uniform",     "pymatching",               "uniform"),
    ("exact ML, uniform",     "ml_decoder:ml_decoder",    "uniform"),
    ("matching, calibrated",  "qubit_calibrated",         "uniform"),
    ("exact ML, calibrated",  "ml_decoder:ml_calibrated", "uniform"),
]


def run(path, max_rounds):
    record, events = rx.load_record(path)
    keep = [i for i, r in enumerate(record["runs"]) if r["rounds"] <= max_rounds]
    record["runs"] = [record["runs"][i] for i in keep]
    events = [events[i] for i in keep]
    rounds = [r["rounds"] for r in record["runs"]]
    print(f"\n{path}\n  {record['track']} track, {record['basis']}-memory, dd={record.get('dd')}, "
          f"rounds {rounds}, {record.get('shots')} shots, calibration "
          f"{(record.get('calibration') or {}).get('source', 'none')}")
    print(f"  {'decoder':<24}" + "".join(f"{'r=' + str(r):>9}" for r in rounds) + "   eps_L/round")
    for label, decoder, noise in DECODERS:
        rec = copy.deepcopy(record)
        try:
            with redirect_stdout(io.StringIO()):          # decode_record prints a line per round
                rx.decode_record(rec, events, noise, decoder)
                fits = rx.fit_record(rec)
        except Exception as e:                             # e.g. no calibration snapshot in this file
            print(f"  {label:<24}skipped: {e}")
            continue
        p = [r["p_error"] for r in rec["runs"]]
        fit = fits.get("amplitude") or fits.get("simple") or {}
        eps = f"{fit['eps_L']:.4f} ± {fit['eps_L_err']:.4f}" if fit else "n/a"
        print(f"  {label:<24}" + "".join(f"{x:>9.4f}" for x in p) + f"   {eps}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("files", nargs="+", help="results/<run>.json from run_experiment.py")
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="skip round counts above this (exact ML gets slow, and the data is noise anyway)")
    args = parser.parse_args()
    for path in args.files:
        run(path, args.max_rounds)
