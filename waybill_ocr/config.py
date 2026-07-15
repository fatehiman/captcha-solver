"""Shared configuration: character set, image size, CTC conventions."""

# Characters the model can output. Index 0 is reserved for the CTC blank,
# so real characters occupy indices 1..len(CHARSET).
CHARSET = "0123456789+-"

BLANK_IDX = 0

# Fixed input geometry fed to the CRNN. Source images are 150x50; we resize
# to a taller-than-wide-friendly canvas that keeps the digits legible.
IMG_H = 32
IMG_W = 128

# Mapping helpers ----------------------------------------------------------
# Real characters are indexed 1..N (0 == blank).
_CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
_IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # + blank


def char_to_idx(c: str) -> int:
    return _CHAR_TO_IDX[c]


def idx_to_char(i: int) -> str:
    return _IDX_TO_CHAR.get(i, "")


def encode(text: str):
    """Expression string -> list of class indices (1-based, no blank)."""
    return [_CHAR_TO_IDX[c] for c in text]
