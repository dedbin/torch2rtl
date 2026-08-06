from __future__ import annotations

import numpy as np
import pytest

from torch2rtl.quant.fixed_point import FixedPointConfig, dequantize_array, quantize_array


def test_quantization_clamps_correctly() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    values = np.asarray([-10.0, -2.0, 0.0, 1.0, 10.0])
    quantized = quantize_array(values, cfg)
    assert quantized.tolist() == [-128, -128, 0, 64, 127]


def test_dequantize_round_trip_sanity() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    values = np.asarray([-0.5, 0.0, 0.25, 1.0])
    quantized = quantize_array(values, cfg)
    restored = dequantize_array(quantized, cfg)
    np.testing.assert_allclose(restored, values, atol=1.0 / cfg.scale)


def test_quantization_uses_ties_to_even() -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=2, acc_bits=32)
    values = np.asarray([0.125, 0.375, -0.125, -0.375])

    quantized = quantize_array(values, cfg)

    assert quantized.tolist() == [0, 2, 0, -2]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_quantization_rejects_non_finite_values(value: float) -> None:
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)

    with pytest.raises(ValueError, match="NaN or infinity"):
        quantize_array(np.asarray([value]), cfg)
