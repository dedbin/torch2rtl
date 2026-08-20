from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


FUSION_FUNCTIONS = (
    "_fusion_require_exact_string_dict",
    "_fusion_raw_static_attribute",
    "_fusion_is_fx_node",
    "_fusion_required_module_attribute",
    "_fusion_positive_module_integer",
    "_fusion_required_parameter",
    "_fusion_optional_parameter",
    "_fuse_conv_batchnorm_eval",
    "_validate_conv_batchnorm_pair",
    "_require_eval_module",
    "_validate_fused_conv",
    "_tensor_values_are_finite",
    "_fresh_fused_target",
    "_fx_node_use_count",
    "_fx_value_reference_count",
)


def _fusion_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=1, bias=True),
        nn.BatchNorm2d(1),
    ).eval()


def test_fusion_component_owns_exact_connected_function_set() -> None:
    component = frontend._fusion_component
    declared = tuple(
        source for source, _consumer in frontend._FUSION_FUNCTION_BINDINGS
    )
    assert declared == FUSION_FUNCTIONS
    assert tuple(
        component.__dict__[name].__module__ for name in declared
    ) == ("torch2rtl.frontend._fusion",) * len(declared)


def test_fusion_component_dependency_dag_is_one_way() -> None:
    component_path = Path(frontend._fusion_component.__file__).resolve()
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
    assert imported == {"math", "numpy"}
    assert imported_from == {
        "__future__",
        "torch2rtl.frontend._errors",
        "torch2rtl.frontend._state_guard",
        "types",
    }
    assert "torch2rtl.frontend.pytorch_fx" not in imported_from
    assert "torch2rtl.frontend._lowering" not in imported_from


@pytest.mark.parametrize("namespace_name", ["internal_pair", "pipeline_consumer"])
def test_actual_fusion_source_and_consumer_bindings_are_consistent(
    namespace_name: str,
) -> None:
    if namespace_name == "internal_pair":
        namespace = frontend._fusion_component.__dict__
        name = "_validate_conv_batchnorm_pair"
        message = "component fusion"
    else:
        namespace = frontend._pipeline_component.__dict__
        name = "_fuse_conv_batchnorm_eval"
        message = "component pipeline"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed {namespace_name} fusion validation executed")

    namespace[name] = replacement
    try:
        with pytest.raises(
            UnsupportedOpError,
            match=message,
        ):
            parse_model(_fusion_model(), input_shape=(1, 2, 2))
    finally:
        namespace[name] = original
    assert calls == 0
    graph = parse_model(_fusion_model(), input_shape=(1, 2, 2))
    assert [type(op).__name__ for op in graph.ops] == ["Conv2dIR"]
    assert graph.metadata["transformations"][0]["kind"] == (
        "conv2d_batchnorm2d_fusion"
    )
