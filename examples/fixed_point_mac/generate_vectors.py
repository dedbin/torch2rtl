import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import linear_fixed

TAPS = 4
DATA_BITS = 8
FRAC_BITS = 6
ACC_BITS = 18
DATA_MASK = (1 << DATA_BITS) - 1
BUS_HEX_DIGITS = TAPS * DATA_BITS // 4
DATA_HEX_DIGITS = DATA_BITS // 4

CFG = FixedPointConfig(
    bits=DATA_BITS,
    frac_bits=FRAC_BITS,
    acc_bits=ACC_BITS,
)


def pack_lane(value: int, lane: int) -> int:
    return (value & DATA_MASK) << (DATA_BITS * lane)


def pack_lanes(values: np.ndarray) -> int:
    if len(values) != TAPS:
        raise ValueError(f"Expected {TAPS} lanes, got {len(values)}")

    packed = 0
    for lane, value in enumerate(values):
        packed |= pack_lane(int(value), lane)
    return packed


TEST_CASES = (
    (
        [0, 16, 32, 64],
        [0, 0, 32, 64],
        -10,
    ),
    (
        [0, 0, 0, -1],
        [0, 0, 0, 65],
        0,
    ),
    (
        [0, 0, 0, 64],
        [0, 0, 0, 1],
        127,
    ),
    (
        [0, 0, 0, 64],
        [0, 0, 0, -1],
        -128,
    ),
    (
        [-128, -128, -128, -128],
        [-128, -128, -128, -128],
        127,
    ),
    (
        [-128, -128, -128, -128],
        [127, 127, 127, 127],
        -128,
    ),
)


def main() -> None:
    for input_values, weight_values, bias_value in TEST_CASES:
        inputs = np.asarray(input_values, dtype=np.int64)
        weights = np.asarray([weight_values], dtype=np.int64)
        bias = np.asarray([bias_value], dtype=np.int64)
        expected = int(linear_fixed(inputs, weights, bias, CFG)[0])

        print(
            f"{pack_lanes(inputs):0{BUS_HEX_DIGITS}x} "
            f"{pack_lanes(weights[0]):0{BUS_HEX_DIGITS}x} "
            f"{bias_value & DATA_MASK:0{DATA_HEX_DIGITS}x} "
            f"{expected & DATA_MASK:0{DATA_HEX_DIGITS}x}"
        )


if __name__ == "__main__":
    main()
