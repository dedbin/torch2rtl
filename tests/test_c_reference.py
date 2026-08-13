from __future__ import annotations

import ctypes
import itertools
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.eda_tools import find_eda_tool
from torch2rtl.frontend.pytorch_fx import load_model_from_file, parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    QuantizedConv2dIR,
    QuantizedFlattenIR,
    QuantizedGraph,
    QuantizedLinearIR,
    QuantizedReluIR,
    conv2d_fixed,
    infer_quantized,
    linear_fixed,
    quantize_graph,
)
from torch2rtl.verify.simulator import run_simulation


REPO_ROOT = Path(__file__).resolve().parents[1]
C_SOURCE = REPO_ROOT / "c_reference" / "fixed_reference.c"

T2R_OK = 0
T2R_VALUE_OUT_OF_RANGE = 4
T2R_ACCUMULATOR_OVERFLOW = 5

_I64 = ctypes.c_int64
_I64_PTR = ctypes.POINTER(_I64)
_SIZE_PTR = ctypes.POINTER(ctypes.c_size_t)


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("bits", ctypes.c_uint),
        ("frac_bits", ctypes.c_uint),
        ("acc_bits", ctypes.c_uint),
    ]


class _CConv2dParams(ctypes.Structure):
    _fields_ = [
        ("in_channels", ctypes.c_size_t),
        ("out_channels", ctypes.c_size_t),
        ("input_height", ctypes.c_size_t),
        ("input_width", ctypes.c_size_t),
        ("kernel_height", ctypes.c_size_t),
        ("kernel_width", ctypes.c_size_t),
        ("stride_height", ctypes.c_size_t),
        ("stride_width", ctypes.c_size_t),
        ("padding_height", ctypes.c_size_t),
        ("padding_width", ctypes.c_size_t),
    ]


def _ptr(values: np.ndarray) -> _I64_PTR:
    return values.ctypes.data_as(_I64_PTR)


def _as_i64(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _config(cfg: FixedPointConfig) -> _CConfig:
    return _CConfig(cfg.bits, cfg.frac_bits, cfg.acc_bits)


def _require_ok(status: int, operation: str) -> None:
    assert status == T2R_OK, f"{operation} returned t2r_status={status}"


@dataclass(frozen=True)
class _CLibrary:
    library: ctypes.CDLL

    def linear(
        self,
        cfg: FixedPointConfig,
        inputs: object,
        weights: object,
        bias: object | None,
    ) -> np.ndarray:
        x = _as_i64(inputs).reshape(-1)
        w = _as_i64(weights)
        assert w.ndim == 2
        out_features, in_features = w.shape
        bias_values = None if bias is None else _as_i64(bias).reshape(-1)
        output = np.empty(out_features, dtype=np.int64)
        c_cfg = _config(cfg)
        status = self.library.t2r_linear(
            ctypes.byref(c_cfg),
            _ptr(x),
            x.size,
            _ptr(w),
            w.size,
            None if bias_values is None else _ptr(bias_values),
            0 if bias_values is None else bias_values.size,
            in_features,
            out_features,
            _ptr(output),
            output.size,
        )
        _require_ok(status, "t2r_linear")
        return output

    def conv2d(
        self,
        cfg: FixedPointConfig,
        inputs: object,
        op: QuantizedConv2dIR,
        *,
        bias: object | None = None,
    ) -> np.ndarray:
        data = _as_i64(inputs).reshape(-1)
        weights = _as_i64(op.weight).reshape(-1)
        bias_values = (
            _as_i64(op.bias if bias is None else bias).reshape(-1)
            if bias is not False
            else None
        )
        params = _CConv2dParams(
            op.in_channels,
            op.out_channels,
            op.input_height,
            op.input_width,
            op.kernel_height,
            op.kernel_width,
            op.stride[0],
            op.stride[1],
            op.padding[0],
            op.padding[1],
        )
        output_height = ctypes.c_size_t()
        output_width = ctypes.c_size_t()
        status = self.library.t2r_conv2d_output_shape(
            ctypes.byref(params),
            ctypes.byref(output_height),
            ctypes.byref(output_width),
        )
        _require_ok(status, "t2r_conv2d_output_shape")
        assert (output_height.value, output_width.value) == (
            op.output_height,
            op.output_width,
        )
        output = np.empty(
            op.out_channels * output_height.value * output_width.value,
            dtype=np.int64,
        )
        c_cfg = _config(cfg)
        status = self.library.t2r_conv2d(
            ctypes.byref(c_cfg),
            ctypes.byref(params),
            _ptr(data),
            data.size,
            _ptr(weights),
            weights.size,
            None if bias_values is None else _ptr(bias_values),
            0 if bias_values is None else bias_values.size,
            _ptr(output),
            output.size,
        )
        _require_ok(status, "t2r_conv2d")
        return output.reshape(
            op.out_channels,
            output_height.value,
            output_width.value,
        )

    def relu(self, cfg: FixedPointConfig, inputs: object) -> np.ndarray:
        data = _as_i64(inputs)
        output = np.empty(data.size, dtype=np.int64)
        c_cfg = _config(cfg)
        status = self.library.t2r_relu(
            ctypes.byref(c_cfg),
            _ptr(data),
            data.size,
            _ptr(output),
            output.size,
        )
        _require_ok(status, "t2r_relu")
        return output.reshape(data.shape)

    def relu_in_place(self, cfg: FixedPointConfig, inputs: object) -> np.ndarray:
        data = _as_i64(inputs).copy()
        c_cfg = _config(cfg)
        status = self.library.t2r_relu(
            ctypes.byref(c_cfg),
            _ptr(data),
            data.size,
            _ptr(data),
            data.size,
        )
        _require_ok(status, "in-place t2r_relu")
        return data

    def flatten(self, cfg: FixedPointConfig, inputs: object) -> np.ndarray:
        data = _as_i64(inputs)
        output = np.empty(data.size, dtype=np.int64)
        c_cfg = _config(cfg)
        status = self.library.t2r_flatten(
            ctypes.byref(c_cfg),
            _ptr(data),
            data.size,
            _ptr(output),
            output.size,
        )
        _require_ok(status, "t2r_flatten")
        return output

    def flatten_overlap(
        self,
        cfg: FixedPointConfig,
        storage: object,
        *,
        input_offset: int,
        output_offset: int,
        count: int,
    ) -> np.ndarray:
        data = _as_i64(storage).copy()
        c_cfg = _config(cfg)
        input_ptr = ctypes.cast(
            ctypes.byref(data.ctypes.data_as(_I64_PTR).contents, input_offset * 8),
            _I64_PTR,
        )
        output_ptr = ctypes.cast(
            ctypes.byref(data.ctypes.data_as(_I64_PTR).contents, output_offset * 8),
            _I64_PTR,
        )
        status = self.library.t2r_flatten(
            ctypes.byref(c_cfg),
            input_ptr,
            count,
            output_ptr,
            count,
        )
        _require_ok(status, "overlapping t2r_flatten")
        return data

    def argmax(self, cfg: FixedPointConfig, inputs: object) -> int:
        data = _as_i64(inputs).reshape(-1)
        c_cfg = _config(cfg)
        output = _I64()
        status = self.library.t2r_argmax(
            ctypes.byref(c_cfg),
            _ptr(data),
            data.size,
            ctypes.byref(output),
        )
        _require_ok(status, "t2r_argmax")
        return int(output.value)


@pytest.fixture(scope="session")
def c_reference(tmp_path_factory: pytest.TempPathFactory) -> _CLibrary:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler named 'cc' is required for C reference tests")
    build_dir = tmp_path_factory.mktemp("c-reference-shared")
    library_path = build_dir / "libtorch2rtl_fixed_reference.so"
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wconversion",
        "-Wsign-conversion",
        "-Werror",
        str(C_SOURCE),
        "-o",
        str(library_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    library = ctypes.CDLL(str(library_path))
    _declare_signatures(library)
    return _CLibrary(library)


def _declare_signatures(library: ctypes.CDLL) -> None:
    config_ptr = ctypes.POINTER(_CConfig)
    params_ptr = ctypes.POINTER(_CConv2dParams)
    library.t2r_linear.argtypes = [
        config_ptr,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
    ]
    library.t2r_linear.restype = ctypes.c_int
    library.t2r_conv2d_output_shape.argtypes = [params_ptr, _SIZE_PTR, _SIZE_PTR]
    library.t2r_conv2d_output_shape.restype = ctypes.c_int
    library.t2r_conv2d.argtypes = [
        config_ptr,
        params_ptr,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
    ]
    library.t2r_conv2d.restype = ctypes.c_int
    library.t2r_relu.argtypes = [
        config_ptr,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
        ctypes.c_size_t,
    ]
    library.t2r_relu.restype = ctypes.c_int
    library.t2r_flatten.argtypes = library.t2r_relu.argtypes
    library.t2r_flatten.restype = ctypes.c_int
    library.t2r_argmax.argtypes = [
        config_ptr,
        _I64_PTR,
        ctypes.c_size_t,
        _I64_PTR,
    ]
    library.t2r_argmax.restype = ctypes.c_int


def _conv_op(
    *,
    cfg: FixedPointConfig,
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> QuantizedConv2dIR:
    in_channels, input_height, input_width = inputs.shape
    out_channels, weight_channels, kernel_height, kernel_width = weights.shape
    assert weight_channels == in_channels
    output_height = (input_height + 2 * padding[0] - kernel_height) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_width) // stride[1] + 1
    assert output_height > 0 and output_width > 0
    return QuantizedConv2dIR(
        name="differential_conv",
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride=stride,
        padding=padding,
        weight=_as_i64(weights),
        bias=_as_i64(bias),
    )


def _boundary_values(cfg: FixedPointConfig) -> np.ndarray:
    values = {
        cfg.min_int,
        cfg.min_int + 1,
        -1,
        0,
        1,
        cfg.max_int - 1,
        cfg.max_int,
    }
    return np.asarray(sorted(values), dtype=np.int64)


@pytest.mark.parametrize(
    "cfg",
    [
        FixedPointConfig(3, 0, 16),
        FixedPointConfig(4, 2, 16),
        FixedPointConfig(8, 6, 32),
        FixedPointConfig(16, 0, 64),
    ],
)
def test_linear_matches_python_on_deterministic_boundary_heavy_cases(
    c_reference: _CLibrary,
    cfg: FixedPointConfig,
) -> None:
    rng = np.random.default_rng(0xC11 + cfg.bits)
    population = _boundary_values(cfg)
    for _ in range(40):
        in_features = int(rng.integers(1, 6))
        out_features = int(rng.integers(1, 5))
        inputs = rng.choice(population, size=in_features)
        weights = rng.choice(population, size=(out_features, in_features))
        bias = rng.choice(population, size=out_features)
        expected = linear_fixed(inputs, weights, bias, cfg)
        actual = c_reference.linear(cfg, inputs, weights, bias)
        np.testing.assert_array_equal(actual, expected)
        without_bias = c_reference.linear(cfg, inputs, weights, None)
        np.testing.assert_array_equal(
            without_bias,
            linear_fixed(inputs, weights, np.zeros(out_features, dtype=np.int64), cfg),
        )


@pytest.mark.parametrize(
    ("cfg", "stride", "padding"),
    [
        (FixedPointConfig(3, 1, 16), (1, 1), (0, 0)),
        (FixedPointConfig(5, 2, 24), (2, 1), (1, 0)),
        (FixedPointConfig(8, 6, 32), (1, 2), (1, 1)),
    ],
)
def test_conv2d_matches_python_on_deterministic_boundary_heavy_cases(
    c_reference: _CLibrary,
    cfg: FixedPointConfig,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    rng = np.random.default_rng(0xC022 + cfg.bits)
    population = _boundary_values(cfg)
    for _ in range(24):
        inputs = rng.choice(population, size=(2, 4, 5))
        weights = rng.choice(population, size=(2, 2, 2, 3))
        bias = rng.choice(population, size=2)
        op = _conv_op(
            cfg=cfg,
            inputs=inputs,
            weights=weights,
            bias=bias,
            stride=stride,
            padding=padding,
        )
        expected = conv2d_fixed(inputs, op, cfg)
        actual = c_reference.conv2d(cfg, inputs, op)
        np.testing.assert_array_equal(actual, expected)
        without_bias = c_reference.conv2d(cfg, inputs, op, bias=False)
        zero_bias_op = _conv_op(
            cfg=cfg,
            inputs=inputs,
            weights=weights,
            bias=np.zeros(op.out_channels, dtype=np.int64),
            stride=stride,
            padding=padding,
        )
        np.testing.assert_array_equal(
            without_bias,
            conv2d_fixed(inputs, zero_bias_op, cfg),
        )


@pytest.mark.parametrize("bits", [2, 3])
def test_exhaustive_small_bit_one_mac_linear_and_conv2d(
    c_reference: _CLibrary,
    bits: int,
) -> None:
    cfg = FixedPointConfig(bits, bits - 1, 16)
    values = range(cfg.min_int, cfg.max_int + 1)
    for input_value, weight, bias in itertools.product(values, repeat=3):
        inputs = np.asarray([input_value], dtype=np.int64)
        weights = np.asarray([[weight]], dtype=np.int64)
        biases = np.asarray([bias], dtype=np.int64)
        np.testing.assert_array_equal(
            c_reference.linear(cfg, inputs, weights, biases),
            linear_fixed(inputs, weights, biases, cfg),
        )
        conv_input = inputs.reshape(1, 1, 1)
        conv_weight = weights.reshape(1, 1, 1, 1)
        op = _conv_op(
            cfg=cfg,
            inputs=conv_input,
            weights=conv_weight,
            bias=biases,
        )
        np.testing.assert_array_equal(
            c_reference.conv2d(cfg, conv_input, op),
            conv2d_fixed(conv_input, op, cfg),
        )


def test_exhaustive_b2_two_mac_prefixes_match_python(
    c_reference: _CLibrary,
) -> None:
    cfg = FixedPointConfig(2, 1, 8)
    values = range(cfg.min_int, cfg.max_int + 1)
    for x0, x1, w0, w1, bias in itertools.product(values, repeat=5):
        inputs = np.asarray([x0, x1], dtype=np.int64)
        weights = np.asarray([[w0, w1]], dtype=np.int64)
        biases = np.asarray([bias], dtype=np.int64)
        np.testing.assert_array_equal(
            c_reference.linear(cfg, inputs, weights, biases),
            linear_fixed(inputs, weights, biases, cfg),
        )
        conv_input = inputs.reshape(1, 1, 2)
        conv_weight = weights.reshape(1, 1, 1, 2)
        op = _conv_op(
            cfg=cfg,
            inputs=conv_input,
            weights=conv_weight,
            bias=biases,
        )
        np.testing.assert_array_equal(
            c_reference.conv2d(cfg, conv_input, op),
            conv2d_fixed(conv_input, op, cfg),
        )


def test_c_and_python_reject_the_same_range_and_accumulator_failures(
    c_reference: _CLibrary,
) -> None:
    cfg = FixedPointConfig(8, 6, 12)
    c_cfg = _config(cfg)
    output = np.empty(1, dtype=np.int64)

    out_of_range_input = _as_i64([128])
    unit_weight = _as_i64([[1]])
    zero_bias = _as_i64([0])
    with pytest.raises(OverflowError, match="Linear input"):
        linear_fixed(out_of_range_input, unit_weight, zero_bias, cfg)
    status = c_reference.library.t2r_linear(
        ctypes.byref(c_cfg),
        _ptr(out_of_range_input),
        out_of_range_input.size,
        _ptr(unit_weight),
        unit_weight.size,
        _ptr(zero_bias),
        zero_bias.size,
        1,
        1,
        _ptr(output),
        output.size,
    )
    assert status == T2R_VALUE_OUT_OF_RANGE

    maximum_input = _as_i64([127])
    maximum_weight = _as_i64([[127]])
    with pytest.raises(OverflowError, match="signed 12-bit accumulator"):
        linear_fixed(maximum_input, maximum_weight, zero_bias, cfg)
    status = c_reference.library.t2r_linear(
        ctypes.byref(c_cfg),
        _ptr(maximum_input),
        maximum_input.size,
        _ptr(maximum_weight),
        maximum_weight.size,
        _ptr(zero_bias),
        zero_bias.size,
        1,
        1,
        _ptr(output),
        output.size,
    )
    assert status == T2R_ACCUMULATOR_OVERFLOW

    conv_input = _as_i64([[[100]]])
    conv_weight = _as_i64([[[[30]]]])
    conv_bias = _as_i64([-16])
    op = _conv_op(
        cfg=cfg,
        inputs=conv_input,
        weights=conv_weight,
        bias=conv_bias,
    )
    with pytest.raises(OverflowError, match="signed 12-bit accumulator"):
        conv2d_fixed(conv_input, op, cfg)
    params = _CConv2dParams(1, 1, 1, 1, 1, 1, 1, 1, 0, 0)
    status = c_reference.library.t2r_conv2d(
        ctypes.byref(c_cfg),
        ctypes.byref(params),
        _ptr(conv_input),
        conv_input.size,
        _ptr(conv_weight),
        conv_weight.size,
        _ptr(conv_bias),
        conv_bias.size,
        _ptr(output),
        output.size,
    )
    assert status == T2R_ACCUMULATOR_OVERFLOW


def test_c_and_python_reject_overflow_before_later_cancellation(
    c_reference: _CLibrary,
) -> None:
    cfg = FixedPointConfig(8, 0, 9)
    c_cfg = _config(cfg)
    output = np.empty(1, dtype=np.int64)

    cases = [
        (
            np.asarray([16], dtype=np.int64),
            np.asarray([[16]], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
        ),
        (
            np.asarray([20, 10, -10], dtype=np.int64),
            np.asarray([[10, 10, 10]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        ),
        (
            np.asarray([-20, -10, 10], dtype=np.int64),
            np.asarray([[10, 10, 10]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        ),
    ]
    for inputs, weights, bias in cases:
        with pytest.raises(OverflowError, match="signed 9-bit accumulator"):
            linear_fixed(inputs, weights, bias, cfg)
        status = c_reference.library.t2r_linear(
            ctypes.byref(c_cfg),
            _ptr(inputs),
            inputs.size,
            _ptr(weights),
            weights.size,
            _ptr(bias),
            bias.size,
            inputs.size,
            1,
            _ptr(output),
            output.size,
        )
        assert status == T2R_ACCUMULATOR_OVERFLOW

        conv_input = inputs.reshape(1, 1, inputs.size)
        conv_weight = weights.reshape(1, 1, 1, inputs.size)
        op = _conv_op(
            cfg=cfg,
            inputs=conv_input,
            weights=conv_weight,
            bias=bias,
        )
        with pytest.raises(OverflowError, match="signed 9-bit accumulator"):
            conv2d_fixed(conv_input, op, cfg)
        params = _CConv2dParams(
            1,
            1,
            1,
            inputs.size,
            1,
            inputs.size,
            1,
            1,
            0,
            0,
        )
        status = c_reference.library.t2r_conv2d(
            ctypes.byref(c_cfg),
            ctypes.byref(params),
            _ptr(conv_input),
            conv_input.size,
            _ptr(conv_weight),
            conv_weight.size,
            _ptr(bias),
            bias.size,
            _ptr(output),
            output.size,
        )
        assert status == T2R_ACCUMULATOR_OVERFLOW


def test_signed_32_bit_endpoints_match_python(
    c_reference: _CLibrary,
) -> None:
    cfg = FixedPointConfig(32, 0, 64)
    inputs = np.asarray([cfg.min_int, cfg.max_int], dtype=np.int64)
    weights = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    bias = np.asarray([0, 0], dtype=np.int64)

    np.testing.assert_array_equal(
        c_reference.linear(cfg, inputs, weights, bias),
        linear_fixed(inputs, weights, bias, cfg),
    )


def test_relu_flatten_and_argmax_match_python_and_metamorphic_properties(
    c_reference: _CLibrary,
) -> None:
    cfg = FixedPointConfig(8, 6, 32)
    rng = np.random.default_rng(0x5EED)
    values = rng.choice(_boundary_values(cfg), size=(3, 4, 5))

    relu_once = c_reference.relu(cfg, values)
    np.testing.assert_array_equal(relu_once, np.maximum(values, 0))
    np.testing.assert_array_equal(c_reference.relu(cfg, relu_once), relu_once)
    np.testing.assert_array_equal(c_reference.relu_in_place(cfg, values), relu_once)

    flattened = c_reference.flatten(cfg, values)
    np.testing.assert_array_equal(flattened, values.reshape(-1))
    np.testing.assert_array_equal(c_reference.flatten(cfg, flattened), flattened)

    logits = np.asarray([-20, 7, 7, -1, 6], dtype=np.int64)
    assert c_reference.argmax(cfg, logits) == int(np.argmax(logits)) == 1
    first_logits = np.asarray([7, 6, 7, -1], dtype=np.int64)
    assert c_reference.argmax(cfg, first_logits) == int(np.argmax(first_logits)) == 0
    translated = logits + 11
    assert translated.max() <= cfg.max_int and translated.min() >= cfg.min_int
    assert c_reference.argmax(cfg, translated) == c_reference.argmax(cfg, logits)


def test_flatten_has_memmove_semantics_for_both_overlap_directions(
    c_reference: _CLibrary,
) -> None:
    cfg = FixedPointConfig(8, 6, 32)
    original = np.arange(-4, 4, dtype=np.int64)

    forward = c_reference.flatten_overlap(
        cfg,
        original,
        input_offset=0,
        output_offset=2,
        count=6,
    )
    expected_forward = original.copy()
    expected_forward[2:8] = original[0:6]
    np.testing.assert_array_equal(forward, expected_forward)

    backward = c_reference.flatten_overlap(
        cfg,
        original,
        input_offset=2,
        output_offset=0,
        count=6,
    )
    expected_backward = original.copy()
    expected_backward[0:6] = original[2:8]
    np.testing.assert_array_equal(backward, expected_backward)


def _tiny_conv_graph() -> tuple[object, QuantizedGraph, FixedPointConfig]:
    model = load_model_from_file(REPO_ROOT / "examples" / "tiny_conv" / "model.py")
    graph = parse_model(model, input_shape=(1, 3, 3))
    cfg = FixedPointConfig(8, 6, 32)
    return graph, quantize_graph(graph, cfg), cfg


def _run_c_qgraph(
    c_reference: _CLibrary,
    qgraph: QuantizedGraph,
    inputs: object,
) -> tuple[tuple[tuple[str, np.ndarray], ...], np.ndarray, int]:
    data = _as_i64(inputs).reshape(qgraph.input_shape)
    activations: list[tuple[str, np.ndarray]] = []
    class_id: int | None = None
    for op in qgraph.ops:
        if isinstance(op, QuantizedConv2dIR):
            data = c_reference.conv2d(qgraph.cfg, data, op)
        elif isinstance(op, QuantizedReluIR):
            data = c_reference.relu(qgraph.cfg, data)
        elif isinstance(op, QuantizedFlattenIR):
            data = c_reference.flatten(qgraph.cfg, data)
        elif isinstance(op, QuantizedLinearIR):
            data = c_reference.linear(qgraph.cfg, data, op.weight, op.bias)
        elif isinstance(op, QuantizedArgmaxIR):
            class_id = c_reference.argmax(qgraph.cfg, data)
            continue
        else:
            raise AssertionError(f"unexpected quantized op {type(op).__name__}")
        activations.append((op.name, data.copy()))
    assert class_id is not None
    return tuple(activations), data, class_id


def _tiny_conv_inputs() -> np.ndarray:
    return np.asarray(
        [
            [-128, 127, 64, 0, -64, 32, 1, -1, 126],
            [127, 127, 127, 127, 127, 127, 127, 127, 127],
            [-128, -128, -128, -128, -128, -128, -128, -128, -128],
        ],
        dtype=np.int64,
    )


def test_complete_tiny_conv_c_pipeline_matches_every_python_activation(
    c_reference: _CLibrary,
) -> None:
    _, qgraph, _ = _tiny_conv_graph()
    for inputs in _tiny_conv_inputs():
        expected = infer_quantized(qgraph, inputs)
        c_activations, c_logits, c_class = _run_c_qgraph(
            c_reference,
            qgraph,
            inputs,
        )
        assert [name for name, _ in c_activations] == [
            activation.name for activation in expected.activations
        ]
        for (_, actual), activation in zip(
            c_activations,
            expected.activations,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, activation.values)
        np.testing.assert_array_equal(c_logits, expected.logits)
        assert c_class == expected.class_id


def test_same_tiny_conv_fixture_matches_c_python_and_rtl(
    c_reference: _CLibrary,
    tmp_path: Path,
) -> None:
    if find_eda_tool("iverilog") is None or find_eda_tool("vvp") is None:
        pytest.skip("Icarus Verilog is not available")
    graph, qgraph, cfg = _tiny_conv_graph()
    inputs = _tiny_conv_inputs()
    expected = [infer_quantized(qgraph, row) for row in inputs]
    for row, python_result in zip(inputs, expected, strict=True):
        _, c_logits, c_class = _run_c_qgraph(c_reference, qgraph, row)
        np.testing.assert_array_equal(c_logits, python_result.logits)
        assert c_class == python_result.class_id

    emit_systemverilog(
        graph,
        cfg,
        tmp_path,
        input_vectors=inputs,
        vector_source="c_python_rtl_shared_fixture",
    )
    simulation = run_simulation(tmp_path)
    assert simulation.ok, simulation.stdout + simulation.stderr
    assert f"PASS vectors={len(inputs)}" in simulation.stdout
