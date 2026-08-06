from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig, dequantize_array
from torch2rtl.quant.reference import infer_float_graph, infer_quantized, quantize_graph


def _graph_and_cfg() -> tuple[object, FixedPointConfig]:
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[0.5, -0.25], [0.25, 0.5], [-0.5, 0.25]])
        )
        model[0].bias.copy_(torch.tensor([0.0, 0.125, -0.125]))
        model[2].weight.copy_(torch.tensor([[0.5, -0.25, 0.25], [-0.25, 0.5, 0.5]]))
        model[2].bias.zero_()
    return parse_model(model, input_shape=(2,)), FixedPointConfig(8, 4, 32)


def test_custom_vectors_drive_every_generated_oracle_and_metadata(
    tmp_path: Path,
) -> None:
    graph, cfg = _graph_and_cfg()
    inputs = np.asarray([[16, -8], [-32, 24], [0, 7]], dtype=np.int16)

    emit_systemverilog(
        graph=graph,
        cfg=cfg,
        out_dir=tmp_path,
        input_vectors=inputs,
        vector_source="directed_custom_test",
    )

    written_inputs = np.loadtxt(tmp_path / "input_vectors.txt", dtype=np.int64)
    np.testing.assert_array_equal(written_inputs, inputs)

    qgraph = quantize_graph(graph, cfg)
    results = [infer_quantized(qgraph, row) for row in inputs]
    expected_logits = np.stack([result.logits for result in results])
    expected_classes = np.asarray([result.class_id for result in results])
    floating_results = [
        infer_float_graph(graph, dequantize_array(row, cfg)) for row in inputs
    ]
    report_errors = np.concatenate(
        [
            np.abs(
                dequantize_array(result.logits, cfg)
                - floating.logits.reshape(-1)
            )
            for result, floating in zip(results, floating_results, strict=True)
        ]
    )
    report_class_matches = sum(
        result.class_id == floating.class_id
        for result, floating in zip(results, floating_results, strict=True)
    )
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "expected_logits.txt", dtype=np.int64),
        expected_logits,
    )
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "expected_classes.txt", dtype=np.int64),
        expected_classes,
    )

    metadata = json.loads((tmp_path / "vectors.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    visualization = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    testbench = (tmp_path / "tb_top.sv").read_text(encoding="utf-8")

    assert metadata["count"] == len(inputs)
    assert metadata["source"] == "directed_custom_test"
    assert "seed" not in metadata
    assert report["vectors"] == {
        "count": len(inputs),
        "source": "directed_custom_test",
    }
    assert report["reference"]["vectors"] == len(inputs)
    assert report["reference"]["vector_source"] == "directed_custom_test"
    assert report["reference"]["class_matches"] == report_class_matches
    assert report["reference"]["class_mismatches"] == len(inputs) - report_class_matches
    assert report["reference"]["mean_abs_logit_error"] == round(
        float(report_errors.mean()), 6
    )
    assert report["reference"]["max_abs_logit_error"] == round(
        float(report_errors.max()), 6
    )
    assert visualization["vectors"]["count"] == len(inputs)
    assert visualization["vectors"]["source"] == "directed_custom_test"
    assert visualization["trace"]["input"]["values"] == inputs[0].tolist()
    assert visualization["trace"]["class_id"] == int(expected_classes[0])
    assert f"localparam int EXPECTED_VECTORS = {len(inputs)}" in testbench


def test_random_vector_flow_remains_backward_compatible(tmp_path: Path) -> None:
    graph, cfg = _graph_and_cfg()

    emit_systemverilog(graph, cfg, tmp_path, vector_count=4, seed=9)

    metadata = json.loads((tmp_path / "vectors.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    visualization = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert metadata["count"] == 4
    assert metadata["seed"] == 9
    assert metadata["source"] == "random_uniform"
    assert report["vectors"] == {"count": 4, "source": "random_uniform"}
    assert visualization["vectors"]["source"] == "random_uniform"

    default_dir = tmp_path / "default"
    emit_systemverilog(graph, cfg, default_dir)
    defaults = json.loads(
        (default_dir / "vectors.json").read_text(encoding="utf-8")
    )
    assert defaults["count"] == 16
    assert defaults["seed"] == 0
    assert defaults["source"] == "random_uniform"


@pytest.mark.parametrize(
    ("inputs", "error_type", "message"),
    [
        (np.asarray([1, 2], dtype=np.int64), ValueError, "two-dimensional"),
        (np.empty((0, 2), dtype=np.int64), ValueError, "at least one"),
        (np.zeros((2, 3), dtype=np.int64), ValueError, "input_size=2"),
        (np.zeros((2, 2), dtype=np.float32), TypeError, "integer dtype"),
        (np.asarray([[0, 128]], dtype=np.int64), ValueError, "signed 8-bit range"),
    ],
)
def test_custom_vectors_are_validated_before_generation(
    tmp_path: Path,
    inputs: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    graph, cfg = _graph_and_cfg()

    with pytest.raises(error_type, match=message):
        emit_systemverilog(
            graph=graph,
            cfg=cfg,
            out_dir=tmp_path,
            input_vectors=inputs,
            vector_source="invalid_test",
        )

    assert not (tmp_path / "input_vectors.txt").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"vector_count": 2}, "vector_count cannot be used"),
        ({"seed": 1}, "seed cannot be used"),
        ({"vector_source": None}, "vector_source must be a non-empty string"),
    ],
)
def test_custom_vectors_reject_conflicting_or_missing_provenance(
    tmp_path: Path,
    extra: dict[str, object],
    message: str,
) -> None:
    graph, cfg = _graph_and_cfg()
    arguments: dict[str, object] = {
        "graph": graph,
        "cfg": cfg,
        "out_dir": tmp_path,
        "input_vectors": np.zeros((2, 2), dtype=np.int64),
        "vector_source": "custom",
    }
    arguments.update(extra)

    with pytest.raises(ValueError, match=message):
        emit_systemverilog(**arguments)  # type: ignore[arg-type]


def test_expected_results_cannot_be_supplied_as_a_custom_oracle(tmp_path: Path) -> None:
    graph, cfg = _graph_and_cfg()

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        emit_systemverilog(
            graph=graph,
            cfg=cfg,
            out_dir=tmp_path,
            input_vectors=np.zeros((1, 2), dtype=np.int64),
            vector_source="custom",
            expected_logits=np.zeros((1, 2), dtype=np.int64),  # type: ignore[call-arg]
        )


def test_vector_source_cannot_be_supplied_without_custom_inputs(tmp_path: Path) -> None:
    graph, cfg = _graph_and_cfg()

    with pytest.raises(ValueError, match="vector_source requires input_vectors"):
        emit_systemverilog(
            graph=graph,
            cfg=cfg,
            out_dir=tmp_path,
            vector_source="orphan_source",
        )


def test_invalid_custom_vectors_do_not_destroy_an_existing_build(
    tmp_path: Path,
) -> None:
    graph, cfg = _graph_and_cfg()
    emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=3)
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    with pytest.raises(TypeError, match="integer dtype"):
        emit_systemverilog(
            graph=graph,
            cfg=cfg,
            out_dir=tmp_path,
            input_vectors=np.zeros((2, 2), dtype=np.float32),
            vector_source="invalid_replacement",
        )

    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after == before
