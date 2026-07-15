# Math-CAPTCHA OCR (auto-waybill-solve)

OCR for noisy **math CAPTCHA** images of the form `<number><operator><number>`
— e.g. `48-10`, `3+39`, `0+14`. The images (150×50 JPG) contain italic, bold,
distorted digits crossed by random distortion lines and speckle noise, which
defeats off-the-shelf OCR (Tesseract) and simple segmentation (adjacent digits
touch, so column projection can't split them).

Instead this project uses a small **CRNN + CTC** model that reads the whole
image end-to-end — no character segmentation — and is trained on the sample
images with heavy, realistic augmentation so it generalizes to future images
from the same generator.

## Approach

```
image ─► resize 32×128, normalize ─► CNN (feature columns) ─► BiLSTM ─► CTC ─► text
```

- **No segmentation.** CTC alignment handles variable-length output and
  touching characters natively.
- **Robust to noise.** The model is trained *through* the distortion lines and
  speckle rather than trying to remove them; augmentation adds extra lines,
  elastic/perspective warps, speckle and erasing so the model learns invariance.
- **Domain post-processing.** Output is validated against
  `^\d{1,2}[+-]\d{1,2}$`, and the arithmetic result can be computed with
  `--solve` (useful for auto-answering the CAPTCHA).

## Project layout

```
auto-waybill-solve/
├── images/              # 126 sample CAPTCHA images (math-01.jpg …)
├── labels.csv           # ground-truth transcriptions (filename,expression)
├── waybill_ocr/
│   ├── config.py        # charset, image size, CTC conventions
│   ├── dataset.py       # data loading + augmentation
│   ├── model.py         # compact CRNN
│   └── decode.py        # CTC greedy decode + expression validation/solve
├── train.py             # train the model -> model.pt
├── ocr.py               # run OCR on an image or a folder
├── evaluate.py          # accuracy vs labels.csv
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
# For a specific PyTorch/CUDA build, follow https://pytorch.org/get-started
```

A CUDA GPU is used automatically when available; otherwise it runs on CPU.

## Usage

### 1. Train

```bash
python train.py                 # train on all labeled images -> model.pt
python train.py --val 0.2       # hold out 20% to report honest accuracy
python train.py --epochs 1500   # more epochs (default 800)
```

### 2. OCR images

```bash
python ocr.py images/math-01.jpg          # single image
python ocr.py images/                      # whole folder
python ocr.py images/ --csv results.csv    # write results to CSV
python ocr.py images/ --solve              # also print the computed answer
```

Example output:

```
math-01.jpg      -> 10-0 = 10
math-08.jpg      -> 80-32 = 48
```

### 3. Evaluate

```bash
python evaluate.py            # exact-match accuracy over labels.csv
python evaluate.py --errors   # list every mismatch
```

## Adapting to new images

- **Same generator, new images:** just run `ocr.py` — no retraining needed.
- **Accuracy drifts / new styles appear:** add the new images to `images/`,
  append their transcriptions to `labels.csv`, and re-run `python train.py`.
  More labeled samples directly improve generalization.
- **Different character set / operators** (e.g. `*`, `/`): extend `CHARSET`
  in `waybill_ocr/config.py` and retrain.

## Notes on labels

`labels.csv` was transcribed by hand from the noisy samples; a few images are
genuinely ambiguous under the distortion. If you spot a wrong transcription,
fix it in `labels.csv` and retrain — the model is only as good as its labels.
```
