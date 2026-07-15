"""Run OCR on math-CAPTCHA images using the trained CRNN.

Usage:
    python ocr.py images/math-01.jpg              # one image
    python ocr.py images/                         # every image in a folder
    python ocr.py images/ --csv results.csv       # also write results to CSV
    python ocr.py images/ --solve                 # print the computed answer too
"""
import os
import csv
import glob
import argparse

import cv2
import torch

from waybill_ocr.dataset import to_input
from waybill_ocr.model import CRNN
from waybill_ocr.decode import greedy_decode, evaluate_expression, is_valid_expression

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CRNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def gather(target):
    if os.path.isdir(target):
        files = [p for p in glob.glob(os.path.join(target, "*"))
                 if p.lower().endswith(IMG_EXTS)]
        return sorted(files)
    return [target]


@torch.no_grad()
def ocr_files(model, files, device, batch=64):
    results = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        tensors, ok = [], []
        for p in chunk:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                results.append((p, "", None))
                continue
            tensors.append(to_input(g))
            ok.append(p)
        if not tensors:
            continue
        xs = torch.stack(tensors, 0).to(device)
        preds = greedy_decode(model(xs))
        for p, text in zip(ok, preds):
            results.append((p, text, evaluate_expression(text)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="image file or folder")
    ap.add_argument("--model", default=os.path.join(ROOT, "model.pt"))
    ap.add_argument("--csv", default=None, help="write results to this CSV")
    ap.add_argument("--solve", action="store_true", help="print computed answer")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}\nTrain it first: python train.py")
    model = load_model(args.model, device)

    files = gather(args.target)
    results = ocr_files(model, files, device)

    for path, text, answer in results:
        name = os.path.basename(path)
        flag = "" if is_valid_expression(text) else "  [!invalid format]"
        line = f"{name:16s} -> {text}"
        if args.solve and answer is not None:
            line += f" = {answer}"
        print(line + flag)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "expression", "answer", "valid"])
            for path, text, answer in results:
                w.writerow([os.path.basename(path), text,
                            "" if answer is None else answer,
                            int(is_valid_expression(text))])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
