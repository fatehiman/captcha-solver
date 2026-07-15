"""Dataset + augmentation for the math-CAPTCHA CRNN.

We train on the real sample images and apply randomized augmentation that
mimics the CAPTCHA generator (distortion lines, speckle, mild affine warps)
so the model generalizes to unseen images of the same style.
"""
import os
import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import IMG_H, IMG_W, encode


def load_labels(labels_csv):
    """Return list of (filename, expression)."""
    rows = []
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["filename"], row["expression"].strip()))
    return rows


def to_input(gray):
    """Resize a grayscale uint8 image to the network input and normalize.

    Returns a float32 tensor of shape (1, IMG_H, IMG_W) in [-1, 1].
    """
    resized = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)


# --- Augmentation ---------------------------------------------------------

def _rand_affine(gray, rng):
    h, w = gray.shape
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.85, 1.15)
    tx = rng.uniform(-0.08, 0.08) * w
    ty = rng.uniform(-0.12, 0.12) * h
    shear = rng.uniform(-0.25, 0.25)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 1] += shear
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(gray, M, (w, h), borderValue=255,
                          flags=cv2.INTER_LINEAR)


def _perspective(gray, rng):
    h, w = gray.shape
    d = 0.12
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-d, d) * w, rng.uniform(-d, d) * h]
                            for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h), borderValue=255)


def _elastic(gray, rng, alpha=None, sigma=6):
    """Smooth random displacement field (elastic distortion)."""
    h, w = gray.shape
    if alpha is None:
        alpha = rng.uniform(8, 18)
    dx = np.random.rand(h, w).astype(np.float32) * 2 - 1
    dy = np.random.rand(h, w).astype(np.float32) * 2 - 1
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (xx + dx).astype(np.float32)
    mapy = (yy + dy).astype(np.float32)
    return cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _random_erase(gray, rng):
    h, w = gray.shape
    out = gray.copy()
    for _ in range(rng.randint(1, 3)):
        ew, eh = rng.randint(4, 14), rng.randint(4, 14)
        x, y = rng.randint(0, w - ew), rng.randint(0, h - eh)
        out[y:y + eh, x:x + ew] = int(rng.choice([0, 255]))
    return out


def _add_lines(gray, rng):
    h, w = gray.shape
    out = gray.copy()
    for _ in range(rng.randint(0, 2)):
        p1 = (rng.randint(-5, w + 5), rng.randint(0, h))
        p2 = (rng.randint(-5, w + 5), rng.randint(0, h))
        color = int(rng.choice([0, 60, 100]))
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
    return out


def _add_speckle(gray, rng):
    h, w = gray.shape
    out = gray.copy()
    n = rng.randint(0, int(0.010 * h * w))
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = np.random.choice([0, 255], n)
    return out


def augment(gray, rng):
    """Mild, realistic augmentation applied on top of the already-noisy
    source images. Kept gentle because the training set is small; the goal
    is invariance to extra lines / speckle / small warps, not to destroy
    the signal."""
    g = _rand_affine(gray, rng)
    if rng.random() < 0.6:
        g = _perspective(g, rng)
    if rng.random() < 0.6:
        g = _elastic(g, rng)
    if rng.random() < 0.6:
        g = _add_lines(g, rng)
    if rng.random() < 0.4:
        g = _add_speckle(g, rng)
    if rng.random() < 0.3:
        g = _random_erase(g, rng)
    if rng.random() < 0.25:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    if rng.random() < 0.4:
        alpha = rng.uniform(0.8, 1.2)   # contrast
        beta = rng.uniform(-20, 20)     # brightness
        g = np.clip(g.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return g


class CaptchaDataset(Dataset):
    def __init__(self, images_dir, samples, train=True, seed=0):
        self.images_dir = images_dir
        self.samples = samples
        self.train = train
        self._cache = {}
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def _load_gray(self, fn):
        if fn not in self._cache:
            path = os.path.join(self.images_dir, fn)
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(path)
            self._cache[fn] = g
        return self._cache[fn]

    def __getitem__(self, i):
        fn, expr = self.samples[i]
        gray = self._load_gray(fn)
        if self.train:
            gray = augment(gray, self.rng)
        x = to_input(gray)
        target = torch.tensor(encode(expr), dtype=torch.long)
        return x, target, len(target), expr


def collate(batch):
    xs, targets, tlens, exprs = zip(*batch)
    xs = torch.stack(xs, 0)
    targets = torch.cat(targets, 0)
    tlens = torch.tensor(tlens, dtype=torch.long)
    return xs, targets, tlens, list(exprs)
