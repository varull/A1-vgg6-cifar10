"""
upload_runs.py: upload finished runs (metrics.csv + summary.json) to W&B.

"""

import argparse
import glob
import json
import os

import pandas as pd

CONFIG_KEYS = ["activation", "optimizer", "lr", "batch_size", "epochs", "scheduler",
               "weight_decay", "autoaugment", "seed"]
SUMMARY_KEYS = ["best_val_acc", "best_epoch", "test_acc", "test_loss"]

p = argparse.ArgumentParser()
p.add_argument("--runs_dir", default="./runs")
p.add_argument("--project", default="vgg6-cifar10")
p.add_argument("--dry_run", action="store_true")
args = p.parse_args()

if not args.dry_run:
    import wandb

uploaded = 0
for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
    csv_path = os.path.join(run_dir, "metrics.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    if not (os.path.isfile(csv_path) and os.path.isfile(summary_path)):
        print(f"skip (unfinished): {os.path.basename(run_dir)}")
        continue

    with open(summary_path) as f:
        s = json.load(f)
    metrics = pd.read_csv(csv_path)
    name = s["run_name"]
    config = {k: s[k] for k in CONFIG_KEYS if k in s}
    print(f"{'would upload' if args.dry_run else 'upload'}: {name}  "
          f"({len(metrics)} epochs, best val {s['best_val_acc']:.2f}%)")
    if args.dry_run:
        uploaded += 1
        continue

    run = wandb.init(project=args.project, name=name, config=config,
                     tags=["uploaded"], reinit=True)
    for row in metrics.to_dict("records"):
        run.log({k: row[k] for k in ("epoch", "lr", "train_loss", "train_acc",
                                     "val_loss", "val_acc")}, step=int(row["epoch"]))
    run.summary.update({k: s[k] for k in SUMMARY_KEYS if k in s})
    run.finish()
    uploaded += 1

print(f"\n{uploaded} runs {'found' if args.dry_run else 'uploaded'}.")