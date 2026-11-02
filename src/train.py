"""
train.py: train VGG6 on CIFAR-10 with one configuration (Q1b, Q1c, Q2).

"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_cifar10
from src.evaluate import evaluate
from src.model import ACTIVATIONS, vgg6

OPTIMIZERS = ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(name, params, lr, weight_decay=0.0):
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """Template's training loop, plus counting loss/accuracy as we go."""
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for i, (data, target) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * target.size(0)
        total += target.size(0)
        correct += outputs.argmax(1).eq(target).sum().item()
    return loss_sum / total, 100.0 * correct / total


def main():
    p = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--optimizer", default="adam", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--autoaugment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./runs")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=None,
                   help="only for quick local checks; never for reported results")
    p.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
    p.add_argument("--wandb_project", default="vgg6-cifar10")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    run_name = (f"{args.activation}_{args.optimizer}_lr{args.lr:g}_bs{args.batch_size}"
                f"_ep{args.epochs}_{args.scheduler}")
    if args.weight_decay != 0:
        run_name += f"_wd{args.weight_decay:g}"
    if args.seed != 42:
        run_name += f"_s{args.seed}"
    if args.autoaugment:
        run_name += "_aa"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar10(
        args.batch_size, args.data_dir, seed=args.seed,
        num_workers=args.num_workers, autoaugment=args.autoaugment)

    model = vgg6(args.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.scheduler == "cosine" else None)

    config = vars(args)
    print(f"Run {run_name} on {device}")

    wb = None
    if args.wandb:
        import wandb  # only needed when --wandb is used
        wb = wandb.init(project=args.wandb_project, name=run_name, config=config)

    best_val_acc, best_epoch = 0.0, 0
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr_now = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                    optimizer, device, args.max_batches)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if scheduler:
                scheduler.step()

            writer.writerow([epoch, lr_now, train_loss, train_acc, val_loss, val_acc])
            f.flush()
            if wb:
                wb.log({"epoch": epoch, "lr": lr_now, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc},
                       step=epoch)
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save({"model_state": model.state_dict(), "config": config,
                            "epoch": epoch, "val_acc": val_acc}, ckpt_path)

            print(f"Epoch {epoch:3d}  lr {lr_now:.4f}  "
                  f"train loss {train_loss:.4f} acc {train_acc:.2f}%  "
                  f"val loss {val_loss:.4f} acc {val_acc:.2f}%  ({time.time() - t0:.0f}s)")

    # Test the best-validation model once.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    summary = {"run_name": run_name, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
               "test_acc": test_acc, "test_loss": test_loss, **config}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best val acc {best_val_acc:.2f}% (epoch {best_epoch})  test acc {test_acc:.2f}%")
    if wb:
        wb.summary.update({"best_val_acc": best_val_acc, "best_epoch": best_epoch,
                           "test_acc": test_acc, "test_loss": test_loss})
        wb.finish()


if __name__ == "__main__":
    main()