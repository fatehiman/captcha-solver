"""CTC greedy decoding and domain post-processing."""
import re

import torch

from .config import BLANK_IDX, idx_to_char

_EXPR_RE = re.compile(r"^\d{1,2}[+-]\d{1,2}$")


def greedy_decode(log_probs):
    """(T, B, C) log-probs -> list of decoded strings (CTC collapse)."""
    # (T, B)
    best = log_probs.argmax(dim=2).transpose(0, 1).cpu().numpy()  # (B, T)
    out = []
    for seq in best:
        chars = []
        prev = -1
        for idx in seq:
            idx = int(idx)
            if idx != prev and idx != BLANK_IDX:
                chars.append(idx_to_char(idx))
            prev = idx
        out.append("".join(chars))
    return out


def is_valid_expression(text: str) -> bool:
    return bool(_EXPR_RE.match(text))


def evaluate_expression(text: str):
    """Return the integer result of a valid expression, else None."""
    if not is_valid_expression(text):
        return None
    m = re.match(r"^(\d{1,2})([+-])(\d{1,2})$", text)
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return a + b if op == "+" else a - b
