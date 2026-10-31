"""Label production CAPTCHA samples for training.

The app saves every issue-time captcha the portal ACCEPTED as
real/<tenant>/ok/<epoch-ms>_<answer>.png — the file name is the true answer,
not the expression. When the current model's expression evaluates to exactly
that answer, the expression is taken as the label. Writes labels_real.csv
(filename relative to real/, expression) and prints how many were kept.

Usage: python label_real.py [--model model.pt]
"""
import os
import csv
import glob
import argparse

import torch

from ocr import load_model, ocr_files

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "model.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "labels_real.csv"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    files = sorted(glob.glob(os.path.join(ROOT, "real", "*", "ok", "*.png")))
    kept, miss = [], 0
    for path, text, val in ocr_files(model, files, device):
        answer = os.path.basename(path).split("_")[1].split(".")[0]
        if val is not None and str(val) == answer:
            kept.append((os.path.relpath(path, os.path.join(ROOT, "real")).replace("\\", "/"), text))
        else:
            miss += 1
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "expression"])
        w.writerows(kept)
    print(f"{len(files)} accepted samples: {len(kept)} labelled, {miss} skipped (model answer differs)")


if __name__ == "__main__":
    main()
