"""
data.py: CIFAR-10 data loading (Q1a).

"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.autoaugment import CIFAR10Policy

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class Cutout:
    """Blank out one random 16x16 square of the (already normalized) image tensor."""

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            y = torch.randint(h, (1,)).item()
            x = torch.randint(w, (1,)).item()
            y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
            x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.expand_as(img)


def get_cifar10(batch_size, data_dir="./data", val_size=5000, seed=42,
                num_workers=2, autoaugment=False):
    """Return (train_loader, val_loader, test_loader)."""
    train_ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if autoaugment:
        train_ops.append(CIFAR10Policy())
    train_ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD), Cutout(1, 16)]
    trans_t = transforms.Compose(train_ops)                                   # training
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])  # val/test

    # Load the 50k training images twice: once augmented (train), once clean (val).
    train_full = datasets.CIFAR10(data_dir, train=True, transform=trans_t, download=True)
    val_full = datasets.CIFAR10(data_dir, train=True, transform=trans, download=True)
    test_data = datasets.CIFAR10(data_dir, train=False, transform=trans, download=True)

    # Same seed -> same 45k/5k split in every run.
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_data = Subset(train_full, perm[val_size:])
    val_data = Subset(val_full, perm[:val_size])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tr, va, te = get_cifar10(128, num_workers=0)
    print("train / val / test images:", len(tr.dataset), len(va.dataset), len(te.dataset))
    x, y = next(iter(tr))
    print("one batch:", tuple(x.shape), tuple(y.shape))
    print("train/val overlap:", len(set(tr.dataset.indices) & set(va.dataset.indices)))