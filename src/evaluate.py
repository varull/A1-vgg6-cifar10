"""
evaluate.py: measure loss and accuracy; test a saved model (Q1c, Q4).

"""

import argparse

import torch
import torch.nn as nn


def evaluate(model, loader, criterion, device):
    """Return (average loss, accuracy in %) over a data loader."""
    model.eval()  # BatchNorm uses its running averages
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss_sum += criterion(outputs, target).item() * target.size(0)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return loss_sum / total, 100.0 * correct / total


if __name__ == "__main__":
    from src.data import get_cifar10
    from src.model import vgg6

    p = argparse.ArgumentParser(description="Test a saved VGG6 model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = vgg6(cfg["activation"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    _, _, test_loader = get_cifar10(256, args.data_dir, seed=cfg["seed"],
                                    num_workers=args.num_workers)
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    print("Config:", {k: cfg[k] for k in ("activation", "optimizer", "lr", "batch_size",
                                          "epochs", "seed")})
    print(f"Saved at epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.2f}%)")
    print(f"Test loss {test_loss:.4f}   Test top-1 accuracy {test_acc:.2f}%")