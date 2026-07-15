"""Evaluate the trained model against labels.csv and report accuracy.

Usage:
    python evaluate.py                 # accuracy over all labeled images
    python evaluate.py --errors        # also list every mismatch
"""
import os
import argparse

import cv2
import torch

from waybill_ocr.dataset import load_labels, to_input
from waybill_ocr.model import CRNN
from waybill_ocr.decode import greedy_decode

ROOT = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "images"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "labels.csv"))
    ap.add_argument("--model", default=os.path.join(ROOT, "model.pt"))
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = CRNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    samples = load_labels(args.labels)
    xs = torch.stack([
        to_input(cv2.imread(os.path.join(args.images, fn), cv2.IMREAD_GRAYSCALE))
        for fn, _ in samples
    ], 0).to(device)
    preds = greedy_decode(model(xs))

    correct = 0
    errors = []
    for (fn, gt), pred in zip(samples, preds):
        if pred == gt:
            correct += 1
        else:
            errors.append((fn, gt, pred))

    total = len(samples)
    print(f"exact-match accuracy: {correct}/{total} = {correct/total:.3f}")
    print(f"errors: {len(errors)}")
    if args.errors:
        for fn, gt, pred in errors:
            print(f"  {fn:16s} gt={gt:8s} pred={pred}")


if __name__ == "__main__":
    main()
