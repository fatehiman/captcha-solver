"""Accuracy of a model on the hand-labelled hard set (labels_hard.csv):
production captchas the portal REJECTED (the old model misread most of them)."""
import os, csv, sys
import torch
from ocr import load_model, ocr_files
ROOT = os.path.dirname(os.path.abspath(__file__))
model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'model.pt')
rows = list(csv.DictReader(open(os.path.join(ROOT, 'labels_hard.csv'))))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = load_model(model_path, dev)
res = ocr_files(m, [os.path.join(ROOT, 'real', r['filename']) for r in rows], dev)
ok = 0
for r, (p, text, val) in zip(rows, res):
    good = val is not None and val == eval(r['expression'])
    ok += good
    if not good and '-v' in sys.argv: print(f"  {r['expression']:>7} read {text}")
print(f"{os.path.basename(model_path)}: {ok}/{len(rows)} answers right ({100*ok/len(rows):.0f}%)")
