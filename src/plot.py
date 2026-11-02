"""
plot.py: quick local plots from the CSV files .

"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--runs_dir", default="./runs")
p.add_argument("--runs", nargs="*", default=None, help="filter: substrings of run names")
p.add_argument("--out", default="outputs/curves.png")
args = p.parse_args()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [("train_loss", "Train loss"), ("train_acc", "Train accuracy (%)"),
          ("val_loss", "Validation loss"), ("val_acc", "Validation accuracy (%)")]
rows = []
for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
    name = os.path.basename(run_dir)
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(csv_path) or (args.runs and not any(r in name for r in args.runs)):
        continue
    m = pd.read_csv(csv_path)
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(m["epoch"], m[col], label=name, marker="o" if len(m) < 5 else None)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        rows.append((name, s["best_val_acc"], s["test_acc"]))

axes[0, 0].legend(fontsize=7)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=120)
print(f"Saved {args.out}")

if rows:
    table = pd.DataFrame(rows, columns=["run", "best_val_acc", "test_acc"])
    print(table.sort_values("best_val_acc", ascending=False).to_string(index=False))