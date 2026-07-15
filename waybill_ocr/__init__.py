"""Waybill math-CAPTCHA OCR package.

A compact CRNN + CTC model that reads noisy math CAPTCHA images of the form
`<number><operator><number>` (e.g. "48-10", "3+39") end-to-end, without
fragile character segmentation.
"""
from .config import CHARSET, BLANK_IDX, IMG_H, IMG_W, idx_to_char, char_to_idx

__all__ = [
    "CHARSET", "BLANK_IDX", "IMG_H", "IMG_W", "idx_to_char", "char_to_idx",
]
