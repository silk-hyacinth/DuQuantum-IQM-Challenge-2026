"""
plot_decoders.py — compare decoders on saved runs (no QPU time).

Each panel shows P_error vs rounds for a no-DD run and a DD run, decoded twice:
solid line + filled marker = first decoder, dashed + hollow = second decoder.

Usage (from the repo root):
    python plot_decoders.py \
        --panel "Qiskit runs" results/<qiskit_noDD>.json results/<qiskit_DD>.json \
        --panel "Qrisp runs"  results/<qrisp_noDD>.json  results/<qrisp_DD>.json \
        --decoders pymatching lookup_ml --out results/decoders_qiskit_vs_qrisp.png

A panel can have one or two files (any DD setting; the label comes from the file).
"""

import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiment import load_record, decode_record, fit_record

COLOR = {False: "#1b6e2a", True: "#1f2d6e"}          # no DD: green, DD: navy (as before)
MARKER = {False: "o", True: "s"}


def decode_all(path, decoders, noise):
    """{decoder: record with p_error per round count and fits}."""
    record, events = load_record(path)
    out = {}
    for dec in decoders:
        r = copy.deepcopy(record)
        print(f"\n{path}  decoder={dec}")
        decode_record(r, events, noise, dec)
        r["fits"] = fit_record(r)
        out[dec] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", action="append", required=True, metavar=("TITLE", "FILE"),
                    help="panel title, then 1-2 results files")
    ap.add_argument("--decoders", nargs=2, default=["pymatching", "lookup_ml"])
    ap.add_argument("--noise", default="uniform")
    ap.add_argument("--out", default="results/decoders_compare.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.panel), figsize=(6.4 * len(args.panel), 5.2), squeeze=False)
    summary = {}
    for ax, (title, *files) in zip(axes[0], args.panel):
        notes = []
        shots = set()
        for path in files:
            decoded = decode_all(path, args.decoders, args.noise)
            for k, dec in enumerate(args.decoders):
                r = decoded[dec]
                dd = bool(r.get("dd"))
                shots.add(r.get("shots"))
                R = [x["rounds"] for x in r["runs"]]
                P = [x["p_error"] for x in r["runs"]]
                E = [x["p_error_err"] for x in r["runs"]]
                ax.errorbar(R, P, yerr=E, color=COLOR[dd], marker=MARKER[dd], markersize=7,
                            linestyle="-" if k == 0 else "--", linewidth=2, capsize=3,
                            markerfacecolor=COLOR[dd] if k == 0 else "white", markeredgewidth=1.6,
                            label=f"{'Dynamic Decoupling' if dd else 'Non-Dynamic Decoupling'}, {dec}")
                f = r["fits"]
                simple = f["simple"]["eps_L"]
                amp = f.get("amplitude", {}).get("eps_L")
                notes.append(f"{'DD' if dd else 'Non-DD'} {dec}: {simple:.4f}"
                             + (f" / {amp:.4f}" if amp is not None else ""))
                summary.setdefault(title, []).append(dict(file=path, dd=dd, decoder=dec, fits=f,
                    p_error={x["rounds"]: x["p_error"] for x in r["runs"]}))
        ax.set_title(f"{title}: d=3, Garnet ({', '.join(str(s) for s in shots)} shots)", fontsize=13)
        ax.set_xlabel("QEC rounds")
        ax.set_ylabel("Logical error (p_error)")
        ax.set_ylim(0, 0.55)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax.text(0.98, 0.03, "fit ε_L/round (simple / amplitude)\n" + "\n".join(notes),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nSaved {args.out} (and the numbers in {args.out.rsplit('.', 1)[0]}.json)")


if __name__ == "__main__":
    main()
