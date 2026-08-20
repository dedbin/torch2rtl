from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


SEMANTICS_FUNCTIONS = (
    "_run_concrete_probe",
    "_validate_fusion_boundary",
    "_validate_lowered_semantics",
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()


def test_semantics_component_owns_exact_probe_function_set() -> None:
    component = frontend._semantics_component
    declared = tuple(
        source for source, _consumer in frontend._SEMANTICS_FUNCTION_BINDINGS
    )
    assert declared == SEMANTICS_FUNCTIONS
    assert tuple(
        component.__dict__[name].__module__ for name in declared
    ) == ("torch2rtl.frontend._semantics",) * len(declared)


def test_semantics_component_dependency_dag_is_one_way() -> None:
    component_path = Path(frontend._semantics_component.__file__).resolve()
    tree = ast.parse(component_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported == {"math", "numpy", "torch2rtl.quant.reference"}
    assert imported_from == {
        "__future__",
        "torch2rtl.frontend._errors",
        "torch2rtl.frontend._state_guard",
        "torch2rtl.ir.graph",
    }
    assert "torch2rtl.frontend.pytorch_fx" not in imported_from
    assert "torch2rtl.frontend._pipeline" not in imported_from


def test_boundary_policies_remain_distinct_and_explicit() -> None:
    boundary_a = inspect.getsource(
        frontend._semantics_component._validate_fusion_boundary
    )
    boundary_b = inspect.getsource(
        frontend._semantics_component._validate_lowered_semantics
    )
    assert "np.allclose" in boundary_a
    assert "(1e-5, 1e-6)" in boundary_a
    assert "(1e-12, 1e-12)" in boundary_a
    assert "np.array_equal" in boundary_b
    assert "allclose" not in boundary_b


@pytest.mark.parametrize("namespace_name", ["source", "pipeline-consumer"])
def test_semantic_boundary_source_and_consumer_bindings_are_consistent(
    namespace_name: str,
) -> None:
    source_namespace = frontend._semantics_component.__dict__
    consumer_namespace = frontend._pipeline_component.__dict__
    if namespace_name == "source":
        namespace = source_namespace
        name = "_validate_fusion_boundary"
    else:
        namespace = consumer_namespace
        name = "_validate_fusion_boundary"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed {namespace_name} semantic probe executed")

    namespace[name] = replacement
    try:
        expected_component = (
            "component semantics"
            if namespace_name == "source"
            else "component pipeline"
        )
        with pytest.raises(UnsupportedOpError, match=expected_component):
            parse_model(_model(), input_shape=(2,))
    finally:
        namespace[name] = original
    assert calls == 0
    assert [
        type(op).__name__ for op in parse_model(_model(), input_shape=(2,)).ops
    ] == ["LinearIR", "ReluIR"]


def test_semantic_oracle_definition_site_binding_is_consistent() -> None:
    namespace = frontend._semantics_component.__dict__
    original = namespace["_INFER_FLOAT_GRAPH"]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("changed semantic oracle executed")

    namespace["_INFER_FLOAT_GRAPH"] = replacement
    try:
        with pytest.raises(UnsupportedOpError, match="component semantics"):
            parse_model(_model(), input_shape=(2,))
    finally:
        namespace["_INFER_FLOAT_GRAPH"] = original
    assert calls == 0
    parse_model(_model(), input_shape=(2,))
