"""Train the type-2 model (5-digit CAPTCHAs) -> model_type2.pt.

The model is the type-1 CRNN with its own weights. Training data is mostly
synthetic, built from digit shapes cut out of the real samples
(waybill_ocr/type2.py), plus the real samples themselves.

Usage:
    python train_type2.py              # train on all samples -> model_type2.pt
    python train_type2.py --folds 4    # honest accuracy: each real image is
                                       # tested by a model that never saw it
                                       # (not even its digit shapes); no file saved
"""
import argparse
import os
import random

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from waybill_ocr.config import CHARSET, IMG_H, IMG_W
from waybill_ocr.dataset import collate, to_input
from waybill_ocr.decode import greedy_decode
from waybill_ocr.model import CRNN
from waybill_ocr.type2 import Type2Dataset, extract_glyphs, load_labels

ROOT = os.path.dirname(os.path.abspath(__file__))


def train(samples, images_dir, args, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    bank = extract_glyphs(images_dir, samples)
    missing = [d for d, g in bank.items() if not g]
    if missing:
        raise SystemExit(f"no sample of digit(s) {missing} in the training images")
    real = [(cv2.imread(os.path.join(images_dir, fn), cv2.IMREAD_GRAYSCALE), t) for fn, t in samples]
    ds = Type2Dataset(bank, real, size=args.synth, real_repeat=args.real_repeat, seed=seed)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0)
    model = CRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(ld))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xs, targets, tlens, _ in ld:
            xs, targets, tlens = xs.to(device), targets.to(device), tlens.to(device)
            logits = model(xs)
            T, B, _ = logits.shape
            loss = ctc(logits, targets, torch.full((B,), T, dtype=torch.long, device=device), tlens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item() * B
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  loss={total / len(ds):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, images_dir, samples, device):
    model.eval()
    xs = torch.stack([to_input(cv2.imread(os.path.join(images_dir, fn), cv2.IMREAD_GRAYSCALE))
                      for fn, _ in samples]).to(device)
    return greedy_decode(model(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "images_type2"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "labels_type2.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model_type2.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--synth", type=int, default=20000, help="synthetic images per epoch")
    ap.add_argument("--real-repeat", type=int, default=50, help="each real image per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_labels(args.labels)
    print(f"device={device}  samples={len(samples)}")

    if args.folds:
        order = samples[:]
        random.Random(args.seed).shuffle(order)
        folds = [order[k::args.folds] for k in range(args.folds)]
        right = digits_right = 0
        for k, test in enumerate(folds):
            train_s = [s for s in order if s not in test]
            print(f"fold {k + 1}/{args.folds}: train {len(train_s)}, test {len(test)}")
            model = train(train_s, args.images, args, device, args.seed + k)
            for (fn, gt), pred in zip(test, predict(model, args.images, test, device)):
                ok = pred == gt
                right += ok
                digits_right += sum(a == b for a, b in zip(pred, gt)) if len(pred) == len(gt) else 0
                print(f"  {fn}: {gt} -> {pred} {'OK' if ok else 'WRONG'}")
        n = len(samples)
        print(f"cross-validation: {right}/{n} codes = {right / n:.3f}, "
              f"digits {digits_right}/{n * 5} = {digits_right / (n * 5):.3f}")
        return

    model = train(samples, args.images, args, device, args.seed)
    preds = predict(model, args.images, samples, device)
    right = sum(p == gt for p, (_, gt) in zip(preds, samples))
    print(f"training images: {right}/{len(samples)}")
    # CPU tensors: the server has no GPU.
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "charset": CHARSET, "img_h": IMG_H,
                "img_w": IMG_W, "type": 2}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
