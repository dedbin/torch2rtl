from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.ir.ops import Conv2dIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import (
    QuantizedConv2dIR,
    infer_quantized,
    quantize_graph,
)
from torch2rtl.verify.simulator import run_simulation


_BATCHNORM_SUPPORTS_BIAS = "bias" in inspect.signature(nn.BatchNorm2d).parameters


@pytest.mark.parametrize(
    "batchnorm_bias",
    [
        pytest.param(True, id="with-bn-bias"),
        pytest.param(
            False,
            id="without-bn-bias",
            marks=pytest.mark.skipif(
                not _BATCHNORM_SUPPORTS_BIAS,
                reason="installed BatchNorm2d constructor has no bias keyword",
            ),
        ),
    ],
)
def test_conv_batchnorm_relu_conv_reaches_bit_exact_icarus(
    tmp_path: Path,
    batchnorm_bias: bool,
) -> None:
    batchnorm_kwargs: dict[str, object] = {}
    if not batchnorm_bias:
        batchnorm_kwargs["bias"] = False
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2, eps=0.25, **batchnorm_kwargs),
        nn.ReLU(),
        nn.Conv2d(2, 2, kernel_size=1, bias=True),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([0.5, -0.75], dtype=torch.float32).reshape(2, 1, 1, 1)
        )
        model[1].running_mean.copy_(torch.tensor([0.5, -0.5]))
        model[1].running_var.copy_(torch.tensor([0.75, 3.75]))
        model[1].weight.copy_(torch.tensor([2.0, -1.0]))
        if model[1].bias is not None:
            model[1].bias.copy_(torch.tensor([0.25, 0.5]))
        model[3].weight.copy_(
            torch.tensor(
                [
                    [0.5, -0.25],
                    [-0.5, 0.75],
                ]
            ).reshape(2, 2, 1, 1)
        )
        model[3].bias.copy_(torch.tensor([0.125, -0.25]))

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR, Conv2dIR]
    assert graph.output.shape == (2, 2, 2)
    first_conv = graph.ops[0]
    assert isinstance(first_conv, Conv2dIR)
    np.testing.assert_array_equal(
        first_conv.weight,
        np.asarray([1.0, 0.375], dtype=np.float32).reshape(2, 1, 1, 1),
    )
    np.testing.assert_array_equal(
        first_conv.bias,
        np.asarray(
            [-0.75, 0.25] if batchnorm_bias else [-1.0, -0.25],
            dtype=np.float32,
        ),
    )
    assert graph.metadata["input_adapter"] == {"kind": "singleton_batch_n1"}
    transformations = graph.metadata["transformations"]
    assert len(transformations) == 1
    assert transformations[0]["kind"] == "conv2d_batchnorm2d_fusion"

    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    qgraph = quantize_graph(graph, cfg)
    quantized_convs = [
        op for op in qgraph.ops if isinstance(op, QuantizedConv2dIR)
    ]
    assert len(quantized_convs) == 2
    np.testing.assert_array_equal(
        quantized_convs[0].weight.reshape(-1),
        np.asarray([64, 24], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        quantized_convs[0].bias,
        np.asarray(
            [-48, 16] if batchnorm_bias else [-64, -16],
            dtype=np.int64,
        ),
    )

    input_vectors = np.asarray(
        [
            [64, 32, -64, 0],
            [-32, 127, -128, 16],
        ],
        dtype=np.int64,
    )
    emit_systemverilog(
        graph,
        cfg,
        tmp_path,
        input_vectors=input_vectors,
        vector_source="directed_conv_batchnorm_fusion",
    )

    expected = [infer_quantized(qgraph, vector) for vector in input_vectors]
    expected_logits = np.asarray([result.logits.reshape(-1) for result in expected])
    expected_classes = np.asarray([result.class_id for result in expected])
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "expected_logits.txt", dtype=np.int64),
        expected_logits,
    )
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "expected_classes.txt", dtype=np.int64),
        expected_classes,
    )

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["graph"]["input_adapter"] == {"kind": "singleton_batch_n1"}
    assert report["graph"]["transformations"] == transformations
    assert "BatchNorm" not in (tmp_path / "top.sv").read_text(encoding="utf-8")

    simulation = run_simulation(tmp_path)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=2" in simulation.stdout
