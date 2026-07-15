"""HTTP API for the math-CAPTCHA OCR model.

Receives an image and returns the recognized expression (and optional answer).
Designed to run headless on a CPU-only server behind a reverse proxy.

Endpoints
---------
GET  /health
    -> {"status": "ok", "device": "cpu"}

POST /ocr
    Body: either
      - multipart/form-data with a file field named "image", or
      - raw image bytes as the request body (Content-Type: image/*).
    Query/form params:
      - solve=1   include the computed arithmetic result
    -> {"expression": "48-10", "answer": 38, "valid": true}

Run (dev):        python api.py
Run (prod):       gunicorn -w 2 -b 127.0.0.1:25470 api:app
"""
import os
import threading

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify

from waybill_ocr.dataset import to_input
from waybill_ocr.model import CRNN
from waybill_ocr.decode import greedy_decode, evaluate_expression, is_valid_expression

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("OCR_MODEL", os.path.join(ROOT, "model.pt"))
MAX_BYTES = int(os.environ.get("OCR_MAX_BYTES", 2 * 1024 * 1024))  # 2 MB
API_KEY = os.environ.get("OCR_API_KEY")  # optional; if set, require header X-API-Key

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
# oneDNN/MKLDNN conv primitives fail to build on some virtualized CPUs
# (RuntimeError: "could not create a primitive"). Fall back to the generic
# convolution kernel — negligible cost for a model this small.
torch.backends.mkldnn.enabled = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
                m = CRNN().to(DEVICE)
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                _model = m
    return _model


def _read_image_bytes():
    """Return raw image bytes from a multipart 'image' field or the body."""
    if "image" in request.files:
        return request.files["image"].read()
    if request.data:
        return request.data
    return None


@app.get("/health")
def health():
    return jsonify(status="ok", device=DEVICE)


@app.post("/ocr")
def ocr():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401

    raw = _read_image_bytes()
    if not raw:
        return jsonify(error="no image provided (send multipart 'image' or raw body)"), 400

    arr = np.frombuffer(raw, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return jsonify(error="could not decode image"), 400

    x = to_input(gray).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        text = greedy_decode(get_model()(x))[0]

    solve = str(request.values.get("solve", "")).lower() in ("1", "true", "yes")
    resp = {"expression": text, "valid": is_valid_expression(text)}
    if solve:
        resp["answer"] = evaluate_expression(text)
    return jsonify(resp)


if __name__ == "__main__":
    get_model()  # warm up
    app.run(host="127.0.0.1", port=int(os.environ.get("OCR_PORT", 25470)))
