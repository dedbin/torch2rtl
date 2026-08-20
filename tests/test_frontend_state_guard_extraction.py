from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


STATE_GUARD_FUNCTIONS = (
    "_state_framework_module_classes",
    "_copy_model_for_tracing",
    "_validate_no_custom_copy_protocol",
    "_validate_no_custom_class_state",
    "_validate_copy_safe_instance_state",
    "_is_copy_safe_state_value",
    "_snapshot_module_class_definitions",
    "_restore_module_class_definitions",
    "_raw_module_state",
    "_trusted_named_modules",
    "_trusted_named_tensors",
    "_snapshot_module_state",
    "_freeze_state_value",
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()


def test_state_guard_component_owns_exact_connected_function_set() -> None:
    component = frontend._state_guard_component
    declared = tuple(
        source for source, _consumer in frontend._STATE_GUARD_FUNCTION_BINDINGS
    )
    assert declared == STATE_GUARD_FUNCTIONS
    assert tuple(function.__module__ for function in (
        component.__dict__[name] for name in declared
    )) == ("torch2rtl.frontend._state_guard",) * len(declared)


def test_state_guard_component_has_one_way_dependencies() -> None:
    component_path = Path(frontend._state_guard_component.__file__).resolve()
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
    assert imported == {"copy", "numpy"}
    assert imported_from == {
        "__future__",
        "collections",
        "torch2rtl.frontend._errors",
        "types",
    }
    assert "torch2rtl.frontend.pytorch_fx" not in imported_from
    assert "torch2rtl.frontend" not in imported_from


@pytest.mark.parametrize("namespace_name", ["source", "pipeline_consumer"])
def test_state_snapshot_actual_source_and_consumer_bindings_are_consistent(
    namespace_name: str,
) -> None:
    source_namespace = frontend._state_guard_component.__dict__
    consumer_namespace = frontend._pipeline_component.__dict__
    namespace = source_namespace if namespace_name == "source" else consumer_namespace
    name = "_snapshot_module_state"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed {namespace_name} state snapshot executed")

    namespace[name] = replacement
    try:
        with pytest.raises(
            UnsupportedOpError,
            match=(
                "component state_guard"
                if namespace_name == "source"
                else "component pipeline"
            ),
        ):
            parse_model(_model(), input_shape=(2,))
    finally:
        namespace[name] = original
    assert calls == 0
    graph = parse_model(_model(), input_shape=(2,))
    assert [type(op).__name__ for op in graph.ops] == ["LinearIR", "ReluIR"]
