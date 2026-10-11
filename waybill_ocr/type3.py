"""Type 3: 6-digit CAPTCHAs (300x100 PNG, Persian digits drawn as outlines, black speckle).

The images are two-tone (black ink / light grey background) with ~500 single black
pixels of speckle. The digits are the Persian shapes (۰ ۱ ۲ ... ۹); the text is returned
as ASCII digits 0-9. Only 20 real samples exist, so training data is made here
(same idea as type 2):

1. ``extract_glyphs`` cuts every digit out of the labelled real images (6 cuts chosen
   together by dynamic programming at the columns with the least ink) and keeps each
   shape with its height position. Odd shapes (bad cuts) are dropped.
2. ``synth`` builds new CAPTCHAs from those shapes: random digits, small rotation / scale /
   position changes per digit, a light background and speckle like the real ones.

The CRNN is the same class as the other types, but with a bigger input (48x136 from a crop
of the text area) because the digits are small and detailed. Nothing here is used by
types 1 and 2.
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

SRC_W, SRC_H = 300, 100
# Text area of the 300x100 image (ink is at x 52..249, y 23..84 in the samples).
CROP_X0, CROP_X1, CROP_Y0, CROP_Y1 = 40, 262, 12, 92
IMG_H3, IMG_W3 = 48, 192          # network input (H must be a multiple of 16; W/4 = CTC steps)
N_DIGITS = 6
PITCH = 31.0                       # distance between digit centres in the real images
FIRST_CENTRE = 68.0
BG = 248
_TEXT_RE = re.compile(r"^\d{6}$")


def is_valid_text(text: str) -> bool:
    return bool(_TEXT_RE.match(text))


def load_labels(labels_csv):
    """Return list of (filename, text)."""
    with open(labels_csv, newline="") as f:
        return [(r["filename"], r["text"].strip()) for r in csv.DictReader(f)]


def _trim_border(gray):
    """Drop dark frame lines at the edges (an element screenshot can show the box border)."""
    for _ in range(3):
        h, w = gray.shape
        if h < 10 or w < 10:
            break
        dark = lambda v: (v < 160).mean() > 0.5
        t, b = dark(gray[0]), dark(gray[-1])
        l, r = dark(gray[:, 0]), dark(gray[:, -1])
        if not (t or b or l or r):
            break
        gray = gray[int(t):h - int(b), int(l):w - int(r)]
    return gray


def normalize(gray):
    """Any form of the picture -> the original 300x100 frame.

    The site shows the 300x100 image in a smaller, wider box with CSS ``object-fit: cover``
    (e.g. 326x58): scaled to the box width, top and bottom cut off. A screenshot of the element
    (capture() in Macromania) is that. Scale to width 300 (keeps the shapes) and centre it on a
    100-px-high background, so the digits are where they are in the original. Works for the
    original image (no change), screenshots at any zoom / devicePixelRatio, PNG or JPEG.
    """
    if gray.shape == (SRC_H, SRC_W):
        return gray
    gray = _trim_border(gray)
    h, w = gray.shape
    nh = max(1, int(round(h * SRC_W / w)))
    g = cv2.resize(gray, (SRC_W, nh), interpolation=cv2.INTER_AREA if w > SRC_W else cv2.INTER_LINEAR)
    if nh >= SRC_H:
        y = (nh - SRC_H) // 2
        return g[y:y + SRC_H]
    out = np.full((SRC_H, SRC_W), BG, np.uint8)
    y = (SRC_H - nh) // 2
    out[y:y + nh] = g
    return out


def to_input(gray):
    """Gray uint8 image of any size -> float tensor (1, IMG_H3, IMG_W3) in [-1, 1]."""
    gray = normalize(gray)
    crop = gray[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    resized = cv2.resize(crop, (IMG_W3, IMG_H3), interpolation=cv2.INTER_AREA)
    x = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)


# --- glyph bank ----------------------------------------------------------

def _ink(gray):
    """Ink mask without the speckle (small components are removed)."""
    b = (gray < 120).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros_like(b)
    for i in range(1, n):
        if st[i, 4] >= 18:
            keep[lab == i] = 1
    return keep


def _cuts(ink, n, wmin=14, wmax=48, wavg=30):
    """n+1 column positions cutting the text into n pieces. Chosen together (DP): few
    ink pixels at each cut, piece widths near ``wavg``. None if impossible."""
    xs = np.nonzero(ink.sum(0))[0]
    if len(xs) == 0:
        return None
    left, right = int(xs[0]), int(xs[-1]) + 1
    cols = np.convolve(ink.sum(0).astype(np.float32), [.25, .5, .25], "same")
    dp = [dict() for _ in range(n + 1)]
    dp[0][left] = (0.0, None)
    for j in range(1, n + 1):
        for x in range(left + wmin * j, right + 1):
            if j == n and x != right:
                continue
            best = None
            for w in range(wmin, wmax + 1):
                p = x - w
                if p in dp[j - 1]:
                    c = dp[j - 1][p][0] + (0 if j == n else cols[x]) + 0.05 * (w - wavg) ** 2
                    if best is None or c < best[0]:
                        best = (c, p)
            if best:
                dp[j][x] = best
    if right not in dp[n]:
        return None
    out, x = [right], right
    for j in range(n, 0, -1):
        x = dp[j][x][1]
        out.append(x)
    return out[::-1]


def extract_glyphs(images_dir, samples):
    """{digit: [(alpha, y_top)]}. alpha = ink strength 0..1 of the cut shape (cropped to
    its ink), y_top = row of its top in the source image."""
    raw = {str(d): [] for d in range(10)}
    for fn, text in samples:
        g = cv2.imdecode(np.fromfile(os.path.join(images_dir, fn), np.uint8), cv2.IMREAD_GRAYSCALE)
        if g.shape != (SRC_H, SRC_W):
            continue                           # screenshots: used as real samples, not for shapes
        ink = _ink(g)
        cuts = _cuts(ink, len(text))
        if cuts is None:
            continue
        for ch, x0, x1 in zip(text, cuts[:-1], cuts[1:]):
            piece = ink[:, x0:x1]
            rows = np.nonzero(piece.sum(1))[0]
            if len(rows) == 0:
                continue
            alpha = piece[rows[0]:rows[-1] + 1].astype(np.float32)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
            raw[ch].append((alpha, int(rows[0])))
    # drop odd shapes: ink area far from the median of the same digit
    bank = {}
    for ch, items in raw.items():
        if len(items) >= 3:
            med = float(np.median([a.sum() for a, _ in items]))
            good = [it for it in items if 0.55 * med <= it[0].sum() <= 1.7 * med]
            items = good or items
        bank[ch] = items
    return bank


# --- synthetic CAPTCHAs ---------------------------------------------------

def _rotate(alpha, deg):
    h, w = alpha.shape
    pad = int(max(h, w) * 0.3) + 2
    a = cv2.copyMakeBorder(alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), deg, 1.0)
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)


def synth(bank, rng, text=None):
    """One synthetic CAPTCHA -> (gray uint8 300x100, text)."""
    if text is None:
        text = "".join(rng.choice("0123456789") for _ in range(N_DIGITS))
    img = np.full((SRC_H, SRC_W), BG, np.float32)
    dx0 = rng.uniform(-8, 8)
    shapes = []
    for ch in text:
        alpha, y_top = rng.choice(bank[ch])
        s = rng.uniform(0.9, 1.12)
        a = cv2.resize(alpha, (max(4, int(round(alpha.shape[1] * s * rng.uniform(0.92, 1.08)))),
                               max(4, int(round(alpha.shape[0] * s)))), interpolation=cv2.INTER_LINEAR)
        shapes.append((a, y_top, alpha.shape[0]))
    # Centres: a fixed pitch, or (like on the site) by the digit widths, so narrow digits
    # such as "11" can stand close together.
    if rng.random() < 0.5:
        centres = [FIRST_CENTRE + dx0 + k * PITCH + rng.uniform(-4, 4) for k in range(len(text))]
    else:
        centres, x = [], 0.0
        for a, _, _ in shapes:
            centres.append(x + a.shape[1] / 2)
            x += a.shape[1] + rng.uniform(-2, 12)
        shift = SRC_W / 2 + rng.uniform(-15, 15) - x / 2
        centres = [c + shift for c in centres]
    for (a, y_top, h0), cx in zip(shapes, centres):
        a = _rotate(a, rng.uniform(-10, 10))
        cy = y_top + h0 / 2 + rng.uniform(-5, 5)
        h, w = a.shape
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(SRC_W, x0 + w), min(SRC_H, y0 + h)
        if xb > xa and yb > ya:
            part = a[ya - y0:yb - y0, xa - x0:xb - x0]
            img[ya:yb, xa:xb] *= (1 - np.clip(part, 0, 1))
    # speckle: single black pixels, like the real ones (~1.5-2 % of the pixels)
    n = int(SRC_W * SRC_H * rng.uniform(0.012, 0.022))
    img[np.random.randint(0, SRC_H, n), np.random.randint(0, SRC_W, n)] = 0
    return np.clip(img, 0, 255).astype(np.uint8), text


def as_screenshot(gray, rng):
    """Make a 300x100 picture look like a screenshot of the site's element: scaled up a bit,
    top and bottom cut (object-fit: cover), sometimes a 1-px border, sometimes JPEG."""
    s = rng.uniform(1.0, 1.25) * rng.choice([1, 1, 1, 2])          # zoom / devicePixelRatio
    w = int(round(SRC_W * s))
    g = cv2.resize(gray, (w, int(round(SRC_H * s))), interpolation=cv2.INTER_LINEAR)
    vh = int(round(w * rng.uniform(0.16, 0.24)))                   # box height (326x58 -> 0.178)
    y = (g.shape[0] - vh) // 2 + rng.randint(-2, 2)
    g = g[max(0, y):max(0, y) + vh].copy()
    if rng.random() < 0.5:
        g = cv2.copyMakeBorder(g, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(rng.uniform(120, 200)))
    if rng.random() < 0.3:
        _, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 95)])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return g


def light_augment(gray, rng):
    """Small changes for the real images."""
    h, w = gray.shape
    M = np.float32([[1, rng.uniform(-0.06, 0.06), rng.uniform(-4, 4)],
                    [rng.uniform(-0.03, 0.03), 1, rng.uniform(-3, 3)]])
    g = cv2.warpAffine(gray, M, (w, h), borderValue=BG)
    n = rng.randint(0, 600)
    g[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 0
    return g


class Type3Dataset(Dataset):
    """Synthetic CAPTCHAs plus the real labelled ones (lightly augmented)."""

    def __init__(self, bank, real=(), size=20000, real_repeat=0, seed=0, shot_p=0.35):
        self.bank = bank
        self.shot_p = shot_p                   # part of the 300x100 pictures turned into "screenshots"
        self.real = list(real)                 # [(gray, text)]
        self.size = size
        self.real_repeat = real_repeat
        self.rng = random.Random(seed)

    def __len__(self):
        return self.size + len(self.real) * self.real_repeat

    def __getitem__(self, i):
        if i >= self.size and self.real:
            gray, text = self.real[(i - self.size) % len(self.real)]
            if gray.shape == (SRC_H, SRC_W):
                gray = light_augment(gray, self.rng)
        else:
            gray, text = synth(self.bank, self.rng)
        if gray.shape == (SRC_H, SRC_W) and self.rng.random() < self.shot_p:
            gray = as_screenshot(gray, self.rng)
        target = torch.tensor(encode(text), dtype=torch.long)
        return to_input(gray), target, len(target), text
