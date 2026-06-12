from __future__ import annotations

import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    QuantizedGraph,
    QuantizedLinearIR,
    QuantizedReluIR,
    infer_quantized,
    linear_fixed,
)


def test_fixed_point_linear_matches_manual_calculation() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=2, acc_bits=32)
    inputs = np.asarray([4, -2], dtype=np.int64)
    weight = np.asarray([[2, -4]], dtype=np.int64)
    bias = np.asarray([1], dtype=np.int64)
    out = linear_fixed(inputs, weight, bias, cfg)
    manual = ((4 * 2) + (-2 * -4) + (1 << cfg.frac_bits)) >> cfg.frac_bits
    assert out.tolist() == [manual]


def test_python_reference_inference_returns_stable_class() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=2, acc_bits=32)
    qgraph = QuantizedGraph(
        input_shape=(2,),
        cfg=cfg,
        ops=(
            QuantizedLinearIR(
                name="linear0",
                in_features=2,
                out_features=2,
                weight=np.asarray([[4, 0], [0, 4]], dtype=np.int64),
                bias=np.asarray([0, 0], dtype=np.int64),
            ),
            QuantizedReluIR(name="relu"),
            QuantizedArgmaxIR(name="argmax"),
        ),
    )
    result = infer_quantized(qgraph, np.asarray([1, 3], dtype=np.int64))
    assert result.class_id == 1
