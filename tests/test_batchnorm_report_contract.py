from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.synth.report import ReportValidationError, read_compile_report
from torch2rtl.verify.simulator import run_simulation


def _compiled_bn_report(tmp_path: Path) -> tuple[dict[str, Any], object]:
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[[[0.25]]], [[[-0.5]]]]))
        assert model[1].running_mean is not None
        assert model[1].running_var is not None
        assert model[1].weight is not None
        assert model[1].bias is not None
        model[1].running_mean.copy_(torch.tensor([0.125, -0.25]))
        model[1].running_var.copy_(torch.tensor([0.75, 1.5]))
        model[1].weight.copy_(torch.tensor([0.5, -0.75]))
        model[1].bias.copy_(torch.tensor([-0.125, 0.25]))
    graph = parse_model(model, input_shape=(1, 2, 2))
    emit_systemverilog(
        graph,
        FixedPointConfig(bits=8, frac_bits=6, acc_bits=32),
        tmp_path,
        vector_count=1,
        seed=17,
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    return report, graph


def test_report_carries_fusion_metadata_and_deployed_parameter_count(
    tmp_path: Path,
) -> None:
    report, graph = _compiled_bn_report(tmp_path)

    assert report["graph"]["input_adapter"] == graph.metadata["input_adapter"]
    assert report["graph"]["transformations"] == graph.metadata["transformations"]
    assert report["graph"]["ops"] == ["Conv2dIR"]
    assert report["metrics"]["parameters"] == 4
    assert report["metrics"]["ops"][0]["parameters"] == 4
    assert len(report["graph"]["transformations"]) == 1
    assert report["graph"]["transformations"][0]["kind"] == (
        "conv2d_batchnorm2d_fusion"
    )
    assert read_compile_report(tmp_path / "report.json")["graph"] == report["graph"]


def test_old_report_without_optional_fusion_metadata_remains_valid(
    tmp_path: Path,
) -> None:
    graph = parse_model(
        nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU()).eval(),
        input_shape=(1, 2, 2),
    )
    emit_systemverilog(
        graph,
        FixedPointConfig(),
        tmp_path,
        vector_count=1,
        seed=3,
    )

    report = read_compile_report(tmp_path / "report.json")

    assert "input_adapter" not in report["graph"]
    assert "transformations" not in report["graph"]


_REQUIRED_TRANSFORMATION_FIELDS = (
    "kind",
    "conv_node",
    "conv_target",
    "batchnorm_node",
    "batchnorm_target",
    "fused_target",
)


def _damage_report(report: dict[str, Any], case: str) -> None:
    graph = report["graph"]
    transformations = graph["transformations"]
    record = transformations[0]
    if case == "adapter-array":
        graph["input_adapter"] = []
    elif case == "adapter-missing-kind":
        graph["input_adapter"] = {}
    elif case == "adapter-kind-bool":
        graph["input_adapter"] = {"kind": True}
    elif case == "adapter-unknown-kind":
        graph["input_adapter"] = {"kind": "unspecified_adapter"}
    elif case == "adapter-only":
        del graph["transformations"]
    elif case == "transformations-only":
        del graph["input_adapter"]
    elif case == "transformations-object":
        graph["transformations"] = {}
    elif case == "transformations-empty":
        graph["transformations"] = []
    elif case == "transformation-number":
        graph["transformations"] = [1]
    elif case.startswith("missing-"):
        del record[case.removeprefix("missing-")]
    elif case == "unknown-transformation-kind":
        record["kind"] = "batchnorm_identity"
    elif case == "node-bool":
        record["conv_node"] = True
    elif case == "target-empty":
        record["fused_target"] = ""
    elif case == "duplicate-fused-target":
        duplicate = copy.deepcopy(record)
        duplicate["conv_node"] = f"{record['conv_node']}_1"
        duplicate["batchnorm_node"] = f"{record['batchnorm_node']}_1"
        transformations.append(duplicate)
    elif case == "duplicate-conv-node":
        duplicate = copy.deepcopy(record)
        duplicate["batchnorm_node"] = f"{record['batchnorm_node']}_1"
        duplicate["fused_target"] = f"{record['fused_target']}_1"
        transformations.append(duplicate)
    elif case == "duplicate-batchnorm-node":
        duplicate = copy.deepcopy(record)
        duplicate["conv_node"] = f"{record['conv_node']}_1"
        duplicate["fused_target"] = f"{record['fused_target']}_1"
        transformations.append(duplicate)
    else:
        raise AssertionError(f"unknown damage case: {case}")


_MALFORMED_FUSION_REPORT_CASES = (
    "adapter-array",
    "adapter-missing-kind",
    "adapter-kind-bool",
    "adapter-unknown-kind",
    "adapter-only",
    "transformations-only",
    "transformations-object",
    "transformations-empty",
    "transformation-number",
    *(f"missing-{name}" for name in _REQUIRED_TRANSFORMATION_FIELDS),
    "unknown-transformation-kind",
    "node-bool",
    "target-empty",
    "duplicate-fused-target",
    "duplicate-conv-node",
    "duplicate-batchnorm-node",
)


@pytest.mark.parametrize("case", _MALFORMED_FUSION_REPORT_CASES)
def test_report_validator_rejects_malformed_fusion_metadata(
    tmp_path: Path,
    case: str,
) -> None:
    report, _graph = _compiled_bn_report(tmp_path)
    _damage_report(report, case)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ReportValidationError,
        match="input_adapter|transformation|fusion|invalid compile report",
    ):
        read_compile_report(report_path)


def test_malformed_fusion_report_fails_before_eda_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _graph = _compiled_bn_report(tmp_path)
    report["graph"]["transformations"][0]["kind"] = "unknown_transformation"
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    def unexpected_eda_lookup(_name: str) -> str | None:
        pytest.fail("simulation reached EDA discovery after invalid report metadata")

    monkeypatch.setattr(
        "torch2rtl.verify.simulator.find_eda_tool",
        unexpected_eda_lookup,
    )

    result = run_simulation(tmp_path)

    assert not result.ok
    assert "invalid compile report" in result.message
    recorded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == "simulation"
    assert recorded["status"]["simulation"]["ok"] is False
