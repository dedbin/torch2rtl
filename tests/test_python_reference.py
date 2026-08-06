from __future__ import annotations

import numpy as np
import pytest

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
    validate_accumulator_width,
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
    assert [activation.name for activation in result.activations] == [
        "linear0",
        "relu",
        "argmax",
    ]
    np.testing.assert_array_equal(result.activations[0].values, [1, 3])
    np.testing.assert_array_equal(result.activations[-1].values, 1)


def test_quantized_reference_derives_class_without_explicit_argmax() -> None:
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
        ),
    )

    result = infer_quantized(qgraph, np.asarray([1, 3], dtype=np.int64))

    assert result.class_id == 1
    assert [activation.name for activation in result.activations] == ["linear0"]


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


def test_linear_fixed_rejects_product_overflow() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    inputs = np.asarray([127], dtype=np.int64)
    weight = np.asarray([[127]], dtype=np.int64)
    bias = np.asarray([0], dtype=np.int64)
    with pytest.raises(OverflowError):
        linear_fixed(inputs, weight, bias, cfg)


def test_linear_fixed_rejects_sum_overflow() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    inputs = np.asarray([127, 127], dtype=np.int64)
    weight = np.asarray([[9, 9]], dtype=np.int64)
    bias = np.asarray([0], dtype=np.int64)
    with pytest.raises(OverflowError):
        linear_fixed(inputs, weight, bias, cfg)


def test_conv2d_fixed_rejects_product_overflow() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    op = QuantizedConv2dIR(
        name="conv",
        in_channels=1,
        out_channels=1,
        input_height=1,
        input_width=1,
        output_height=1,
        output_width=1,
        kernel_height=1,
        kernel_width=1,
        stride=(1, 1),
        padding=(0, 0),
        weight=np.asarray([[[[30]]]], dtype=np.int64),
        bias=np.asarray([-16], dtype=np.int64),
    )
    inputs = np.asarray([100], dtype=np.int64)
    with pytest.raises(OverflowError):
        conv2d_fixed(inputs, op, cfg)


def test_linear_fixed_accepts_accumulator_max() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    output = linear_fixed(
        np.asarray([89], dtype=np.int64),
        np.asarray([[23]], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        cfg,
    )
    np.testing.assert_array_equal(output, np.asarray([31], dtype=np.int64))


def test_linear_fixed_rejects_accumulator_max_plus_one() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    with pytest.raises(OverflowError):
        linear_fixed(
            np.asarray([64, 64], dtype=np.int64),
            np.asarray([[16, 16]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            cfg,
        )


def test_linear_fixed_accepts_accumulator_min() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    output = linear_fixed(
        np.asarray([-128], dtype=np.int64),
        np.asarray([[16]], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        cfg,
    )
    np.testing.assert_array_equal(output, np.asarray([-32], dtype=np.int64))


def test_linear_fixed_rejects_accumulator_min_minus_one() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    with pytest.raises(OverflowError):
        linear_fixed(
            np.asarray([-128, 1], dtype=np.int64),
            np.asarray([[16, -1]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            cfg,
        )


def test_linear_fixed_uses_arithmetic_shift_for_negative_accumulator() -> None:
    inputs = [-1]
    weight = [[1], [63], [64], [65]]
    bias = [0, 0, 0, 0]
    expected = [-1, -1, -1, -2]
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    output = linear_fixed(inputs, weight, bias, cfg)
    np.testing.assert_array_equal(output, expected)


def test_linear_fixed_requantizes_without_additional_rounding() -> None:
    inputs = [96]
    weight = [[1]]
    bias = [0]
    expected = [1]
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=12)
    output = linear_fixed(inputs, weight, bias, cfg)
    np.testing.assert_array_equal(output, expected)


def test_linear_fixed_saturates_output_at_signed_int8_boundaries() -> None:
    inputs = [64]
    weight = [[0], [1], [0], [-1]]
    bias = [127, 127, -128, -128]
    expected = [127, 127, -128, -128]
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=18)
    output = linear_fixed(inputs, weight, bias, cfg)
    np.testing.assert_array_equal(output, expected)


def test_accumulator_width_validation_rejects_possible_linear_wraparound() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=14)
    qgraph = QuantizedGraph(
        input_shape=(1,),
        cfg=cfg,
        ops=(
            QuantizedLinearIR(
                name="unsafe_linear",
                in_features=1,
                out_features=1,
                weight=np.asarray([[127]], dtype=np.int64),
                bias=np.asarray([0], dtype=np.int64),
            ),
        ),
    )

    with pytest.raises(
        OverflowError,
        match=r"ACC_BITS=14.*unsafe_linear.*requires at least 15",
    ):
        validate_accumulator_width(qgraph)


def test_accumulator_width_validation_accepts_exact_linear_boundary() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=15)
    qgraph = QuantizedGraph(
        input_shape=(1,),
        cfg=cfg,
        ops=(
            QuantizedLinearIR(
                name="safe_linear",
                in_features=1,
                out_features=1,
                weight=np.asarray([[127]], dtype=np.int64),
                bias=np.asarray([0], dtype=np.int64),
            ),
        ),
    )

    validate_accumulator_width(qgraph)


def test_accumulator_width_validation_checks_conv_mac_prefixes() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=15)
    op = QuantizedConv2dIR(
        name="unsafe_conv",
        in_channels=1,
        out_channels=1,
        input_height=1,
        input_width=2,
        output_height=1,
        output_width=1,
        kernel_height=1,
        kernel_width=2,
        stride=(1, 1),
        padding=(0, 0),
        weight=np.asarray([[[[127, 127]]]], dtype=np.int64),
        bias=np.asarray([0], dtype=np.int64),
    )
    qgraph = QuantizedGraph(input_shape=(1, 1, 2), cfg=cfg, ops=(op,))

    with pytest.raises(
        OverflowError,
        match=r"unsafe_conv.*MAC prefix 2.*requires at least 16",
    ):
        validate_accumulator_width(qgraph)


@pytest.mark.parametrize("value", [-129, 128])
def test_fixed_reference_rejects_values_outside_data_width(value: int) -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=16)

    with pytest.raises(OverflowError, match="Linear input.*signed 8-bit"):
        linear_fixed(
            np.asarray([value], dtype=np.int64),
            np.asarray([[1]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            cfg,
        )
