"""Train the math-CAPTCHA CRNN on the labeled sample images.

Usage:
    python train.py                      # train on all data, save model.pt
    python train.py --val 0.2            # hold out 20% for honest val accuracy
    python train.py --epochs 400

The trained weights are saved to model.pt (state_dict + config).
"""
import os
import csv
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from waybill_ocr.config import NUM_CLASSES, IMG_H, IMG_W, CHARSET
from waybill_ocr.dataset import load_labels, CaptchaDataset, collate
from waybill_ocr.model import CRNN
from waybill_ocr.decode import greedy_decode

ROOT = os.path.dirname(os.path.abspath(__file__))


def sequence_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xs, _, _, exprs in loader:
            xs = xs.to(device)
            preds = greedy_decode(model(xs))
            for p, gt in zip(preds, exprs):
                correct += int(p == gt)
                total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "images"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model.pt"))
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.0,
                    help="fraction held out for validation (0 = train on all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  classes={NUM_CLASSES}  input={IMG_H}x{IMG_W}")

    samples = load_labels(args.labels)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"train={len(train_samples)}  val={len(val_samples)}")

    train_ds = CaptchaDataset(args.images, train_samples, train=True, seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=0)

    # Clean (non-augmented) loaders for accuracy reporting.
    train_eval_ld = DataLoader(
        CaptchaDataset(args.images, train_samples, train=False),
        batch_size=args.batch, collate_fn=collate)
    val_ld = None
    if val_samples:
        val_ld = DataLoader(
            CaptchaDataset(args.images, val_samples, train=False),
            batch_size=args.batch, collate_fn=collate)

    model = CRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xs, targets, tlens, _ in train_ld:
            xs, targets, tlens = xs.to(device), targets.to(device), tlens.to(device)
            logits = model(xs)                      # (T, B, C)
            T, B, _ = logits.shape
            in_lens = torch.full((B,), T, dtype=torch.long, device=device)
            loss = ctc(logits, targets, in_lens, tlens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * B
        sched.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            tr_acc = sequence_accuracy(model, train_eval_ld, device)
            msg = f"epoch {epoch:4d}  loss={running/len(train_ds):.4f}  train_acc={tr_acc:.3f}"
            if val_ld is not None:
                v = sequence_accuracy(model, val_ld, device)
                msg += f"  val_acc={v:.3f}"
                if v >= best_val:
                    best_val = v
            print(msg)

    torch.save({
        "state_dict": model.state_dict(),
        "charset": CHARSET,
        "img_h": IMG_H,
        "img_w": IMG_W,
    }, args.out)
    print(f"saved -> {args.out}")
    if val_ld is not None:
        print(f"best val_acc={best_val:.3f}")


if __name__ == "__main__":
    main()
