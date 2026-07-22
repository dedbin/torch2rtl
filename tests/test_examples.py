from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig


def test_grid_classifier_example_compiles(tmp_path: Path) -> None:
    module = _load_module(Path("examples/grid_classifier/model.py"))
    model = module.create_model()
    graph = parse_model(model, input_shape=(module.GRID_ROWS, module.GRID_COLS))

    assert [type(op) for op in graph.ops] == [
        FlattenIR,
        LinearIR,
        ReluIR,
        LinearIR,
        ReluIR,
        LinearIR,
        ArgmaxIR,
    ]
    assert graph.linear_ops[0].in_features == 16
    assert graph.linear_ops[-1].out_features == 4

    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    report = emit_systemverilog(graph, cfg, tmp_path, vector_count=3, seed=7)
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert "top.sv" in report.generated_files
    assert "visualization.html" in report.generated_files
    assert payload["graph"]["input_shape"] == [4, 4]
    assert payload["graph"]["ops"] == [
        "FlattenIR",
        "LinearIR",
        "ReluIR",
        "LinearIR",
        "ReluIR",
        "LinearIR",
        "ArgmaxIR",
    ]


def test_tiny_conv_example_compiles(tmp_path: Path) -> None:
    module = _load_module(Path("examples/tiny_conv/model.py"))
    model = module.create_model()
    graph = parse_model(
        model,
        input_shape=(module.INPUT_CHANNELS, module.GRID_ROWS, module.GRID_COLS),
    )

    assert [type(op) for op in graph.ops] == [
        Conv2dIR,
        ReluIR,
        FlattenIR,
        LinearIR,
        ArgmaxIR,
    ]
    conv = graph.ops[0]
    linear = graph.ops[3]
    assert isinstance(conv, Conv2dIR)
    assert isinstance(linear, LinearIR)
    assert conv.input.shape == (1, 3, 3)
    assert conv.output.shape == (1, 2, 2)
    assert conv.kernel_height == 2
    assert conv.kernel_width == 2
    assert conv.stride == (1, 1)
    assert conv.padding == (0, 0)
    assert linear.in_features == 4
    assert linear.out_features == 4

    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    report = emit_systemverilog(graph, cfg, tmp_path, vector_count=3, seed=11)
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "visualization.json").read_text(encoding="utf-8"))

    assert "conv2d_comb.sv" in report.generated_files
    assert "visualization.html" in report.generated_files
    assert payload["graph"]["input_shape"] == [1, 3, 3]
    assert payload["quant"] == {"bits": 8, "frac_bits": 6, "acc_bits": 32}
    assert payload["metrics"]["parameters"] == 25
    assert payload["metrics"]["macs"] == 32
    assert payload["reference"]["kind"] == "fixed_vs_float_ir"
    assert [block["kind"] for block in manifest["blocks"] if block["lane"] == "main"] == [
        "input",
        "conv2d",
        "relu",
        "flatten",
        "linear",
        "argmax",
        "output",
    ]


def test_image_cnn_example_compiles(tmp_path: Path) -> None:
    data_module = _load_module(Path("examples/image_cnn/data.py"))
    model_module = _load_module(Path("examples/image_cnn/model.py"))
    images, labels = data_module.generate_dataset(
        data_module.DatasetConfig(samples_per_class=2, noise=0.02, seed=5)
    )

    assert tuple(images.shape) == (8, 1, 4, 4)
    assert sorted(labels.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]

    model = model_module.create_model(checkpoint_path=tmp_path / "missing.pt")
    graph = parse_model(
        model,
        input_shape=(
            model_module.INPUT_CHANNELS,
            data_module.IMAGE_SIZE,
            data_module.IMAGE_SIZE,
        ),
    )

    assert [type(op) for op in graph.ops] == [
        Conv2dIR,
        ReluIR,
        FlattenIR,
        LinearIR,
        ArgmaxIR,
    ]

    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    report = emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=13)
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert "conv2d_comb.sv" in report.generated_files
    assert "visualization.html" in report.generated_files
    assert payload["graph"]["input_shape"] == [1, 4, 4]
    assert payload["metrics"]["parameters"] > 0
    assert payload["metrics"]["macs"] > 0


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(spec.name)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        sys.path.pop(0)
    return module
