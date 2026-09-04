from __future__ import annotations

import numpy as np


def block_positions(length: int, block_size: int, stride: int) -> list[int]:
    positions = list(range(0, max(length - block_size + 1, 1), stride))
    final = max(length - block_size, 0)
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def raised_cosine_window(size: int) -> np.ndarray:
    vector = np.hanning(size).astype(np.float32)
    vector = np.maximum(vector, 0.08)
    return np.outer(vector, vector)
