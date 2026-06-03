"""inspect_history.py — pretty-print a run's history.json trajectory.

Usage:
    python -m scripts.inspect_history runs/v2a_effb3_bcedice/runs/v2c_mitb2_bcedice ...
    python -m scripts.inspect_history --every 5 runs/v2a_effb3_bcedice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="One or more run directories (each must contain history.json).")
    parser.add_argument("--every", type=int, default=5,
                        help="Print every Nth epoch (best epoch is always shown). Default 5.")
    args = parser.parse_args()

    cols = [
        ("train_total",                   "train_total"),
        ("val_total",                     "val_total"),
        ("val_dice",                      "val_dice"),
        ("val_iou",                       "val_iou"),
        ("val_cldice",                    "val_cldice"),
        ("val_length_ratio_skel_median",  "len_skel_med"),
        ("val_length_ratio_skel_mean",    "len_skel_mean"),
    ]

    for run in args.runs:
        hp = run / "history.json"
        if not hp.is_file():
            print(f"{run}: no history.json — skipping")
            continue
        with open(hp) as f:
            h = json.load(f)
        n = len(h["val_dice"])
        best_ep = max(range(n), key=lambda i: h["val_dice"][i])

        print(f"\n========== {run.name} ==========")
        print(f"epochs trained: {n}")
        print(f"best epoch:     {best_ep+1}, val_dice={h['val_dice'][best_ep]:.4f}")
        print(f"final epoch:    {n},          val_dice={h['val_dice'][-1]:.4f}")
        print()
        header = "epoch | " + " | ".join(f"{label:>11}" for _, label in cols)
        print(header)
        print("-" * len(header))
        seen = set()
        rows = list(range(0, n, args.every))
        if best_ep not in rows:
            rows.append(best_ep)
        rows.append(n - 1)
        for i in sorted(set(rows)):
            if i in seen:
                continue
            seen.add(i)
            vals = " | ".join(f"{h[key][i]:>11.4f}" for key, _ in cols)
            mark = "  <-- BEST" if i == best_ep else ""
            print(f"{i+1:>5} | {vals}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
