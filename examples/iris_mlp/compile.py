from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .data import TEST_SIZE, restore_iris_data
    from .model import (
        DEFAULT_CHECKPOINT_PATH,
        INPUT_SIZE,
        OUTPUT_SIZE,
        load_model_checkpoint,
    )
else:
    from data import TEST_SIZE, restore_iris_data
    from model import (
        DEFAULT_CHECKPOINT_PATH,
        INPUT_SIZE,
        OUTPUT_SIZE,
        load_model_checkpoint,
    )

from torch2rtl.backend.systemverilog.emit import BuildReport, emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array
from torch2rtl.quant.reference import infer_float_graph, infer_quantized, quantize_graph


DEFAULT_BUILD_DIR = Path(__file__).resolve().parents[2] / "build" / "iris_mlp" / "rtl_q8_4"
IRIS_VECTOR_SOURCE = "iris_test_split"


@dataclass(frozen=True)
class CompileResult:
    build_report: BuildReport
    pytorch_logits: np.ndarray
    graphir_logits: np.ndarray
    fixed_logits: np.ndarray
    pytorch_classes: np.ndarray
    graphir_classes: np.ndarray
    fixed_classes: np.ndarray
    quantized_inputs: np.ndarray
    saturation: dict[str, dict[str, int]]
    test_accuracy: float


def compile_iris_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    out_dir: Path = DEFAULT_BUILD_DIR,
) -> CompileResult:
    loaded = load_model_checkpoint(checkpoint_path)
    data = restore_iris_data(loaded.payload)
    graph = parse_model(loaded.model, input_shape=(INPUT_SIZE,))

    with torch.no_grad():
        pytorch_logits = loaded.model(data.test_inputs).numpy()
    graphir_logits = np.stack(
        [
            infer_float_graph(graph, sample.numpy()).logits
            for sample in data.test_inputs
        ]
    )
    expected_logits_shape = (TEST_SIZE, OUTPUT_SIZE)
    if (
        pytorch_logits.shape != expected_logits_shape
        or graphir_logits.shape != expected_logits_shape
    ):
        raise AssertionError(
            "PyTorch and GraphIR logits must both have shape "
            f"{expected_logits_shape}"
        )
    np.testing.assert_allclose(pytorch_logits, graphir_logits, rtol=1e-5, atol=1e-6)
    pytorch_classes = np.argmax(pytorch_logits, axis=1).astype(np.int64)
    graphir_classes = np.argmax(graphir_logits, axis=1).astype(np.int64)
    np.testing.assert_array_equal(pytorch_classes, graphir_classes)

    cfg = FixedPointConfig(bits=8, frac_bits=4, acc_bits=32)
    qgraph = quantize_graph(graph, cfg)
    quantized_inputs = np.stack(
        [quantize_array(sample.numpy(), cfg) for sample in data.test_inputs]
    )
    fixed_results = [infer_quantized(qgraph, row) for row in quantized_inputs]
    fixed_logits = np.stack([result.logits for result in fixed_results])
    fixed_classes = np.asarray([result.class_id for result in fixed_results], dtype=np.int64)

    if quantized_inputs.shape != (TEST_SIZE, INPUT_SIZE):
        raise AssertionError(
            f"quantized Iris inputs must have shape {(TEST_SIZE, INPUT_SIZE)}"
        )
    if fixed_logits.shape != expected_logits_shape:
        raise AssertionError(
            f"fixed-point logits must have shape {expected_logits_shape}"
        )
    if fixed_classes.shape != (TEST_SIZE,):
        raise AssertionError(
            f"fixed-point classes must have shape {(TEST_SIZE,)}"
        )
    _assert_signed_range("quantized inputs", quantized_inputs, cfg)
    _assert_signed_range("fixed-point logits", fixed_logits, cfg)
    np.testing.assert_array_equal(fixed_classes, pytorch_classes)

    saturation = {
        "fc1": {"min": 0, "max": 0},
        "fc2": {"min": 0, "max": 0},
    }
    for result in fixed_results:
        activations = {activation.name: activation.values for activation in result.activations}
        for name, counts in saturation.items():
            values = activations[name]
            _assert_signed_range(f"{name} activations", values, cfg)
            counts["min"] += int(np.count_nonzero(values == cfg.min_int))
            counts["max"] += int(np.count_nonzero(values == cfg.max_int))
    if any(count for counts in saturation.values() for count in counts.values()):
        raise AssertionError(f"unexpected Q8.4 activation saturation: {saturation}")

    build_report = emit_systemverilog(
        graph=graph,
        cfg=cfg,
        out_dir=out_dir,
        input_vectors=quantized_inputs,
        vector_source=IRIS_VECTOR_SOURCE,
    )
    test_accuracy = float(np.mean(fixed_classes == data.test_targets.numpy()))
    return CompileResult(
        build_report=build_report,
        pytorch_logits=pytorch_logits,
        graphir_logits=graphir_logits,
        fixed_logits=fixed_logits,
        pytorch_classes=pytorch_classes,
        graphir_classes=graphir_classes,
        fixed_classes=fixed_classes,
        quantized_inputs=quantized_inputs,
        saturation=saturation,
        test_accuracy=test_accuracy,
    )


def _assert_signed_range(
    name: str,
    values: np.ndarray,
    cfg: FixedPointConfig,
) -> None:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise AssertionError(f"{name} must have an integer dtype")
    if int(array.min()) < cfg.min_int or int(array.max()) > cfg.max_int:
        raise AssertionError(
            f"{name} must fit signed {cfg.bits}-bit range "
            f"[{cfg.min_int}, {cfg.max_int}]"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile the trained Iris MLP to Q8.4 RTL.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUILD_DIR)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run examples/iris_mlp/train.py first."
        )

    result = compile_iris_checkpoint(args.checkpoint, args.out)
    print("checkpoint load: PASS (missing=0, unexpected=0)")
    print(
        "PyTorch float vs GraphIR float: "
        f"{TEST_SIZE}/{TEST_SIZE} classes match"
    )
    print(
        "PyTorch float vs Q8.4 fixed: "
        f"{TEST_SIZE}/{TEST_SIZE} classes match"
    )
    print(f"test accuracy: {result.test_accuracy:.2%}")
    print(f"saturation: {result.saturation}")
    print(f"verification vectors: {result.quantized_inputs.shape}, source={IRIS_VECTOR_SOURCE}")
    print(f"RTL build: {result.build_report.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
