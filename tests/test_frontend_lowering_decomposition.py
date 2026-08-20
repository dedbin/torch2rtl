from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest
import torch
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


LOWERING_HANDLERS = (
    "_lower_linear_module",
    "_lower_conv2d_module",
    "_lower_relu_module",
    "_lower_flatten_module",
    "_lower_argmax_node",
)


def _representative_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()


def test_module_dispatcher_is_static_and_ordered() -> None:
    source = inspect.getsource(frontend._lowering_component._parse_module_node)
    positions = tuple(source.index(name) for name in LOWERING_HANDLERS[:4])
    assert positions == tuple(sorted(positions))
    assert "dict(" not in source
    tree = ast.parse(textwrap.dedent(source))
    assert not any(isinstance(node, (ast.Dict, ast.DictComp)) for node in ast.walk(tree))
    assert "register" not in source.lower()


@pytest.mark.parametrize("handler_name", LOWERING_HANDLERS)
def test_lowering_handler_binding_mutation_never_executes(
    handler_name: str,
) -> None:
    namespace = frontend._lowering_component.__dict__
    original = namespace[handler_name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed handler executed: {handler_name}")

    namespace[handler_name] = replacement
    try:
        with pytest.raises(
            UnsupportedOpError,
            match="component lowering",
        ):
            parse_model(_representative_model(), input_shape=(2,))
    finally:
        namespace[handler_name] = original
    assert calls == 0
    graph = parse_model(_representative_model(), input_shape=(2,))
    assert [type(op).__name__ for op in graph.ops] == ["LinearIR", "ReluIR"]


def test_argmax_handler_preserves_terminal_graph_shape() -> None:
    class Model(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.argmax(value)

    graph = parse_model(Model().eval(), input_shape=(3,))
    assert [type(op).__name__ for op in graph.ops] == ["ArgmaxIR"]
    assert graph.output.shape == ()
    assert graph.output.dtype == "int64"


def test_lowering_component_has_one_way_dependencies() -> None:
    component_path = Path(frontend._lowering_component.__file__).resolve()
    tree = ast.parse(component_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules == {
        "__future__",
        "collections.abc",
        "torch2rtl.frontend._errors",
        "torch2rtl.ir.graph",
        "torch2rtl.ir.ops",
        "torch2rtl.ir.tensor",
        "typing",
    }
    assert not any(
        name in imported_modules
        for name in (
            "torch2rtl.frontend",
            "torch2rtl.frontend.pytorch_fx",
            "torch2rtl.backend",
            "torch2rtl.verify",
            "torch2rtl.visualization",
        )
    )


@pytest.mark.parametrize("lookup_name", ["internal_handler", "pipeline_entry"])
def test_actual_lowering_lookup_bindings_are_independently_checked(
    lookup_name: str,
) -> None:
    if lookup_name == "internal_handler":
        namespace = frontend._lowering_component.__dict__
        name = "_lower_linear_module"
        message = "component lowering"
    else:
        namespace = frontend._pipeline_component.__dict__
        name = "lower_fx_graph"
        message = "component pipeline"
    original = namespace[name]
    calls = 0

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"changed {lookup_name} lowering binding executed")

    namespace[name] = replacement
    try:
        with pytest.raises(
            UnsupportedOpError,
            match=message,
        ):
            parse_model(_representative_model(), input_shape=(2,))
    finally:
        namespace[name] = original
    assert calls == 0
    graph = parse_model(_representative_model(), input_shape=(2,))
    assert [type(op).__name__ for op in graph.ops] == ["LinearIR", "ReluIR"]
