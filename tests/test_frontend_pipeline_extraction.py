from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


PIPELINE_STAGES = (
    "_validate_input_shape",
    "_validate_model_semantics",
    "_copy_model_for_tracing",
    "symbolic_trace",
    "_fuse_conv_batchnorm_eval",
    "lower_fx_graph",
    "_validate_fusion_boundary",
    "_validate_lowered_semantics",
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()


def test_pipeline_has_one_authored_stage_order_without_registry() -> None:
    source = inspect.getsource(frontend._pipeline_component._parse_model_impl)
    positions = tuple(source.index(stage) for stage in PIPELINE_STAGES)
    assert positions == tuple(sorted(positions))
    assert "register" not in source.lower()
    assert "stages" not in source.lower()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        for node in ast.walk(tree)
    )


def test_pipeline_dependency_dag_is_one_way_and_has_no_public_entrypoint() -> None:
    component = frontend._pipeline_component
    component_path = Path(component.__file__).resolve()
    tree = ast.parse(component_path.read_text(encoding="utf-8"))
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_from == {
        "__future__",
        "torch2rtl.frontend._errors",
        "torch2rtl.frontend._fusion",
        "torch2rtl.frontend._lowering",
        "torch2rtl.frontend._model_contract",
        "torch2rtl.frontend._semantics",
        "torch2rtl.frontend._state_guard",
        "torch2rtl.ir.graph",
        "typing",
    }
    assert "torch2rtl.frontend.pytorch_fx" not in imported_from
    assert "parse_model" not in component.__dict__


@pytest.mark.parametrize("namespace_name", ["source", "consumer"])
def test_pipeline_entrypoint_source_and_consumer_bindings_are_consistent(
    namespace_name: str,
) -> None:
    source_namespace = frontend._pipeline_component.__dict__
    consumer_namespace = frontend.__dict__
    namespace = source_namespace if namespace_name == "source" else consumer_namespace
    name = "_parse_model_impl"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed {namespace_name} pipeline executed")

    namespace[name] = replacement
    try:
        with pytest.raises(
            UnsupportedOpError,
            match=(
                "component pipeline"
                if namespace_name == "source"
                else "parser implementation"
            ),
        ):
            parse_model(_model(), input_shape=(2,))
    finally:
        namespace[name] = original
    assert calls == 0
    graph = parse_model(_model(), input_shape=(2,))
    assert [type(op).__name__ for op in graph.ops] == ["LinearIR", "ReluIR"]


def test_pipeline_definition_site_stage_binding_is_checked() -> None:
    namespace = frontend._pipeline_component.__dict__
    name = "_fuse_conv_batchnorm_eval"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("changed pipeline fusion stage executed")

    namespace[name] = replacement
    try:
        with pytest.raises(UnsupportedOpError, match="component pipeline"):
            parse_model(_model(), input_shape=(2,))
    finally:
        namespace[name] = original
    assert calls == 0
    parse_model(_model(), input_shape=(2,))
