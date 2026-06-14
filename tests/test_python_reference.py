from __future__ import annotations

import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    QuantizedConv2dIR,
    QuantizedGraph,
    QuantizedLinearIR,
    QuantizedReluIR,
    conv2d_fixed,
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


def test_fixed_point_conv2d_matches_manual_calculation() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=0, acc_bits=32)
    op = QuantizedConv2dIR(
        name="conv",
        in_channels=1,
        out_channels=1,
        input_height=3,
        input_width=3,
        output_height=2,
        output_width=2,
        kernel_height=2,
        kernel_width=2,
        stride=(1, 1),
        padding=(0, 0),
        weight=np.asarray([[[[1, 0], [0, 1]]]], dtype=np.int64),
        bias=np.asarray([1], dtype=np.int64),
    )
    inputs = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

    output = conv2d_fixed(inputs, op, cfg)

    np.testing.assert_array_equal(
        output,
        np.asarray([[[7, 9], [13, 15]]], dtype=np.int64),
    )
