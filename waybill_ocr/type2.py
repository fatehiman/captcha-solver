"""Type 2: 5-digit CAPTCHAs (100x25 JPG, grey pixel font, thin noise lines).

Only a few real samples exist, so training data is made here:

1. ``extract_glyphs`` cuts every digit out of the labelled real images (the
   site's own font), giving a bank of real digit shapes.
2. ``synth`` composes new CAPTCHAs from that bank: random digits, spacing and
   offsets on a background like the real one (light grey, faint sketch lines,
   a few darker strokes), then a JPEG round trip.

The model is the same CRNN as type 1 (same input size and charset, so
``to_input`` / ``greedy_decode`` are shared) with its own weights file.
Nothing here is used by type 1.
"""
import csv
import os
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import encode
from .dataset import to_input

SRC_W, SRC_H = 100, 25
INK = 160          # pixels darker than this are digit ink (digits ~100-150, background 170-250)
N_DIGITS = 5
_TEXT_RE = re.compile(r"^\d{5}$")


def is_valid_text(text: str) -> bool:
    return bool(_TEXT_RE.match(text))


def load_labels(labels_csv):
    """Return list of (filename, text)."""
    with open(labels_csv, newline="") as f:
        return [(r["filename"], r["text"].strip()) for r in csv.DictReader(f)]


# --- glyph bank ----------------------------------------------------------

def _cut_points(ink_cols, n):
    """n-1 column cuts inside the text span, each at the lowest ink between
    evenly spaced guesses (digits can touch, so plain gaps are not enough)."""
    xs = np.nonzero(ink_cols)[0]
    left, right = int(xs[0]), int(xs[-1]) + 1
    step = (right - left) / n
    cuts = [left]
    for k in range(1, n):
        guess = left + step * k
        lo, hi = int(guess - step * 0.35), int(guess + step * 0.35) + 1
        window = ink_cols[lo:hi]
        cuts.append(lo + int(np.argmin(window)))
    cuts.append(right)
    return cuts


def extract_glyphs(images_dir, samples):
    """{digit: [alpha arrays]} cut from labelled images. Alpha = ink strength 0..1."""
    bank = {str(d): [] for d in range(10)}
    for fn, text in samples:
        g = cv2.imread(os.path.join(images_dir, fn), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        alpha = np.clip((INK + 20 - g) / 45.0, 0, 1)          # ink mask, soft only at the edges
        ink_cols = (g < INK).sum(0).astype(np.float32)
        ink_cols = np.convolve(ink_cols, [0.25, 0.5, 0.25], mode="same")
        cuts = _cut_points(ink_cols, len(text))
        for ch, x0, x1 in zip(text, cuts[:-1], cuts[1:]):
            piece = alpha[:, x0:x1]
            rows = np.nonzero((piece > 0.5).sum(1) > 0)[0]
            if piece.shape[1] < 4 or len(rows) < 8:
                continue
            bank[ch].append(piece[rows[0]:rows[-1] + 1].copy())
    return bank


# --- synthetic CAPTCHAs ---------------------------------------------------

def _background(rng):
    bg = np.full((SRC_H, SRC_W), rng.uniform(212, 228), np.float32)
    bg += np.random.normal(0, rng.uniform(2, 6), bg.shape).astype(np.float32)
    # faint sketch strokes (the drawings behind the digits)
    for _ in range(rng.randint(3, 9)):
        cx, cy = rng.uniform(0, SRC_W), rng.uniform(0, SRC_H)
        axes = (int(rng.uniform(3, 14)), int(rng.uniform(3, 12)))
        a0 = rng.uniform(0, 360)
        cv2.ellipse(bg, (int(cx), int(cy)), axes, rng.uniform(0, 180), a0, a0 + rng.uniform(60, 200),
                    rng.uniform(185, 208), 1, cv2.LINE_AA)
    return bg


def _strokes(img, rng):
    """Thin darker curves across the digits, like the real noise lines."""
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        pts = []
        x, y = rng.uniform(-5, SRC_W * 0.6), rng.uniform(0, SRC_H)
        for _ in range(rng.randint(2, 5)):
            pts.append([x, y])
            x += rng.uniform(5, 30)
            y = np.clip(y + rng.uniform(-10, 10), -3, SRC_H + 3)
        cv2.polylines(img, [np.array(pts, np.int32)], False, rng.uniform(70, 150), 1, cv2.LINE_AA)
    return img


def synth(bank, rng, text=None):
    """One synthetic CAPTCHA -> (gray uint8 100x25, text)."""
    if text is None:
        text = "".join(rng.choice("0123456789") for _ in range(N_DIGITS))
    glyphs = []
    for ch in text:
        a = rng.choice(bank[ch])
        s = rng.uniform(0.95, 1.05)
        h = int(np.clip(round(a.shape[0] * s), 14, SRC_H - 1))
        w = max(4, int(round(a.shape[1] * h / a.shape[0] * rng.uniform(0.92, 1.08))))
        glyphs.append(cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR))
    total = sum(g.shape[1] for g in glyphs)
    gap_room = SRC_W - 4 - total
    gaps = [rng.uniform(-2, max(0.0, gap_room / N_DIGITS)) for _ in range(N_DIGITS - 1)]
    used = total + sum(gaps)
    x = rng.uniform(1, max(1.5, SRC_W - used - 1))
    img = _background(rng)
    ink = rng.uniform(105, 125)
    for g, gap in zip(glyphs, gaps + [0]):
        h, w = g.shape
        y = int(np.clip(round((SRC_H - h) / 2 + rng.uniform(-2, 2)), 0, SRC_H - h))
        xi = int(round(x))
        x0, x1 = max(0, xi), min(SRC_W, xi + w)
        if x1 > x0:
            a = g[:, x0 - xi:x1 - xi]
            region = img[y:y + h, x0:x1]
            img[y:y + h, x0:x1] = region * (1 - a) + ink * a
        x += w + gap
    img = _strokes(img, rng)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 92)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), text


def light_augment(gray, rng):
    """Small changes for the real images (they are already noisy)."""
    h, w = gray.shape
    M = np.float32([[1, rng.uniform(-0.08, 0.08), rng.uniform(-3, 3)],
                    [0, 1, rng.uniform(-1.5, 1.5)]])
    g = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    g = _strokes(g.astype(np.float32), rng)
    alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    return np.clip(g * alpha + beta, 0, 255).astype(np.uint8)


class Type2Dataset(Dataset):
    """Synthetic CAPTCHAs plus the real labelled ones (lightly augmented)."""

    def __init__(self, bank, real=(), size=20000, real_repeat=0, seed=0):
        self.bank = bank
        self.real = list(real)                 # [(gray, text)]
        self.size = size
        self.real_repeat = real_repeat
        self.rng = random.Random(seed)

    def __len__(self):
        return self.size + len(self.real) * self.real_repeat

    def __getitem__(self, i):
        if i >= self.size and self.real:
            gray, text = self.real[(i - self.size) % len(self.real)]
            gray = light_augment(gray, self.rng)
        else:
            gray, text = synth(self.bank, self.rng)
        target = torch.tensor(encode(text), dtype=torch.long)
        return to_input(gray), target, len(target), text
