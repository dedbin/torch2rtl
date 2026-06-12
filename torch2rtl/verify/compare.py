from __future__ import annotations

import numpy as np


def classes_match(expected: np.ndarray, actual: np.ndarray) -> bool:
    return np.array_equal(
        np.asarray(expected, dtype=np.int64),
        np.asarray(actual, dtype=np.int64),
    )
