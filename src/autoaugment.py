"""
autoaugment.py: CIFAR-10 AutoAugment policy (optional, enabled with --autoaugment).


"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

_FILL = (128, 128, 128)

# Magnitude ranges: index 0..9 picks the strength of each operation.
_RANGES = {
    "shearX": np.linspace(0, 0.3, 10),
    "shearY": np.linspace(0, 0.3, 10),
    "translateX": np.linspace(0, 150 / 331, 10),
    "translateY": np.linspace(0, 150 / 331, 10),
    "rotate": np.linspace(0, 30, 10),
    "color": np.linspace(0.0, 0.9, 10),
    "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
    "solarize": np.linspace(256, 0, 10),
    "contrast": np.linspace(0.0, 0.9, 10),
    "sharpness": np.linspace(0.0, 0.9, 10),
    "brightness": np.linspace(0.0, 0.9, 10),
    "autocontrast": [0] * 10,
    "equalize": [0] * 10,
    "invert": [0] * 10,
}


def _sign():
    return random.choice([-1, 1])


def _rotate_with_fill(img, degrees):
    rot = img.convert("RGBA").rotate(degrees)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


# Each op takes (PIL image, magnitude) and returns a PIL image.
_OPS = {
    "shearX": lambda img, m: img.transform(
        img.size, Image.Transform.AFFINE, (1, m * _sign(), 0, 0, 1, 0),
        Image.Resampling.BICUBIC, fillcolor=_FILL),
    "shearY": lambda img, m: img.transform(
        img.size, Image.Transform.AFFINE, (1, 0, 0, m * _sign(), 1, 0),
        Image.Resampling.BICUBIC, fillcolor=_FILL),
    "translateX": lambda img, m: img.transform(
        img.size, Image.Transform.AFFINE, (1, 0, m * img.size[0] * _sign(), 0, 1, 0),
        fillcolor=_FILL),
    "translateY": lambda img, m: img.transform(
        img.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, m * img.size[1] * _sign()),
        fillcolor=_FILL),
    "rotate": lambda img, m: _rotate_with_fill(img, m * _sign()),
    "color": lambda img, m: ImageEnhance.Color(img).enhance(1 + m * _sign()),
    "posterize": lambda img, m: ImageOps.posterize(img, int(m)),
    "solarize": lambda img, m: ImageOps.solarize(img, float(m)),
    "contrast": lambda img, m: ImageEnhance.Contrast(img).enhance(1 + m * _sign()),
    "sharpness": lambda img, m: ImageEnhance.Sharpness(img).enhance(1 + m * _sign()),
    "brightness": lambda img, m: ImageEnhance.Brightness(img).enhance(1 + m * _sign()),
    "autocontrast": lambda img, m: ImageOps.autocontrast(img),
    "equalize": lambda img, m: ImageOps.equalize(img),
    "invert": lambda img, m: ImageOps.invert(img),
}


class SubPolicy:
    """Two operations, each applied with its own probability and magnitude."""

    def __init__(self, p1, op1, mag_idx1, p2, op2, mag_idx2):
        self.p1, self.op1, self.m1 = p1, _OPS[op1], _RANGES[op1][mag_idx1]
        self.p2, self.op2, self.m2 = p2, _OPS[op2], _RANGES[op2][mag_idx2]
        self.desc = f"({op1},{p1},{mag_idx1})+({op2},{p2},{mag_idx2})"

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.op1(img, self.m1)
        if random.random() < self.p2:
            img = self.op2(img, self.m2)
        return img


class CIFAR10Policy:
    """Pick one of the 25 best CIFAR-10 sub-policies at random for each image."""

    def __init__(self):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1),
        ]

    def __call__(self, img):
        return random.choice(self.policies)(img)

    def __repr__(self):
        return "CIFAR10Policy(AutoAugment, 25 sub-policies)"