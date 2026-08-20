from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


MODEL_CONTRACT_FUNCTIONS = (
    "_contract_require_exact_string_dict",
    "_contract_raw_static_attribute",
    "_validate_model_semantics",
    "_validate_module_dispatch",
    "_validate_no_custom_introspection",
    "_validate_forward_python_state",
    "_validate_reachable_python_function",
    "_find_custom_method",
    "_next_loaded_attribute",
    "_is_immutable_python_state",
    "_contains_code_object",
    "_normalize_float_dtype",
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()


def test_model_contract_component_owns_exact_policy_function_set() -> None:
    component = frontend._model_contract_component
    declared = tuple(
        source for source, _consumer in frontend._MODEL_CONTRACT_FUNCTION_BINDINGS
    )
    assert declared == MODEL_CONTRACT_FUNCTIONS
    assert tuple(
        component.__dict__[name].__module__ for name in declared
    ) == ("torch2rtl.frontend._model_contract",) * len(declared)


def test_model_contract_dependency_dag_is_one_way() -> None:
    component_path = Path(frontend._model_contract_component.__file__).resolve()
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
    assert imported == {"builtins", "dis", "numpy"}
    assert imported_from == {
        "__future__",
        "torch2rtl.frontend._errors",
        "torch2rtl.frontend._state_guard",
        "types",
    }
    assert "torch2rtl.frontend.pytorch_fx" not in imported_from
    assert "torch2rtl.frontend._pipeline" not in imported_from


@pytest.mark.parametrize("namespace_name", ["internal_policy", "pipeline_consumer"])
def test_actual_contract_policy_bindings_are_consistent(
    namespace_name: str,
) -> None:
    if namespace_name == "internal_policy":
        namespace = frontend._model_contract_component.__dict__
        name = "_validate_reachable_python_function"
        message = "component model_contract"
    else:
        namespace = frontend._pipeline_component.__dict__
        name = "_validate_model_semantics"
        message = "component pipeline"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed {namespace_name} model policy executed")

    namespace[name] = replacement
    try:
        with pytest.raises(
            UnsupportedOpError,
            match=message,
        ):
            parse_model(_model(), input_shape=(2,))
    finally:
        namespace[name] = original
    assert calls == 0
    graph = parse_model(_model(), input_shape=(2,))
    assert [type(op).__name__ for op in graph.ops] == ["LinearIR", "ReluIR"]
