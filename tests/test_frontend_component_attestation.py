from __future__ import annotations

import dis
import json
from pathlib import Path
import subprocess
import sys
from types import CodeType, FunctionType

import pytest

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expected_components() -> tuple[tuple[object, ...], ...]:
    return (
        (
            "frontend_core",
            frontend,
            frontend.__dict__,
            tuple(
                (name, name, value)
                for name, value in frontend._PROTECTED_FRONTEND_BINDINGS
            ),
            tuple(
                (name, name, state)
                for name, state in frontend._PROTECTED_FRONTEND_FUNCTION_STATES
            ),
        ),
        (
            "errors",
            frontend._errors_component,
            frontend.__dict__,
            (("UnsupportedOpError", "UnsupportedOpError", frontend.UnsupportedOpError),),
            (),
        ),
        (
            "lowering",
            frontend._lowering_component,
            frontend._lowering_component.__dict__,
            frontend._PROTECTED_LOWERING_BINDINGS,
            frontend._PROTECTED_LOWERING_FUNCTION_STATES,
        ),
        (
            "state_guard",
            frontend._state_guard_component,
            frontend._state_guard_component.__dict__,
            frontend._PROTECTED_STATE_GUARD_BINDINGS,
            frontend._PROTECTED_STATE_GUARD_FUNCTION_STATES,
        ),
        (
            "fusion",
            frontend._fusion_component,
            frontend._fusion_component.__dict__,
            frontend._PROTECTED_FUSION_BINDINGS,
            frontend._PROTECTED_FUSION_FUNCTION_STATES,
        ),
        (
            "semantics",
            frontend._semantics_component,
            frontend._semantics_component.__dict__,
            frontend._PROTECTED_SEMANTICS_BINDINGS,
            frontend._PROTECTED_SEMANTICS_FUNCTION_STATES,
        ),
        (
            "model_contract",
            frontend._model_contract_component,
            frontend._model_contract_component.__dict__,
            frontend._PROTECTED_MODEL_CONTRACT_BINDINGS,
            frontend._PROTECTED_MODEL_CONTRACT_FUNCTION_STATES,
        ),
        (
            "pipeline",
            frontend._pipeline_component,
            frontend._pipeline_component.__dict__,
            frontend._PROTECTED_PIPELINE_BINDINGS,
            frontend._PROTECTED_PIPELINE_FUNCTION_STATES,
        ),
        (
            "pipeline_facade",
            frontend._pipeline_component,
            frontend.__dict__,
            (),
            frontend._PROTECTED_PIPELINE_FACADE_FUNCTION_STATES,
        ),
    )


def test_component_manifest_is_explicit_and_complete() -> None:
    manifest = frontend._COMPONENT_ATTESTATION_MANIFEST
    expected = _expected_components()
    assert type(manifest) is tuple
    assert tuple(entry[0] for entry in manifest) == tuple(
        entry[0] for entry in expected
    )
    for actual, expected_entry in zip(manifest, expected, strict=True):
        label, module, source, consumer, bindings, functions = actual
        expected_label, expected_module, expected_consumer, expected_bindings, expected_functions = expected_entry
        assert label == expected_label
        assert module is expected_module
        assert source is module.__dict__
        assert consumer is expected_consumer
        assert bindings == expected_bindings
        assert functions == expected_functions


def test_component_manifest_is_captured_by_validator() -> None:
    validator = frontend._validate_framework_integrity
    closure = dict(
        zip(
            validator.__code__.co_freevars,
            validator.__closure__ or (),
            strict=True,
        )
    )
    checked = closure["checked_components"].cell_contents
    expected = _expected_components()
    assert len(checked) == len(expected)
    for actual, expected_entry in zip(checked, expected, strict=True):
        label, module, source, consumer = actual[:4]
        expected_label, expected_module, expected_consumer = expected_entry[:3]
        assert label == expected_label
        assert module is expected_module
        assert source is expected_module.__dict__
        assert consumer is expected_consumer


def _nested_code_objects(code: CodeType) -> tuple[CodeType, ...]:
    nested = tuple(
        item for item in code.co_consts if type(item) is CodeType
    )
    return (code,) + tuple(
        descendant
        for item in nested
        for descendant in _nested_code_objects(item)
    )


def test_component_manifest_covers_actual_mutable_global_lookups() -> None:
    components = tuple(
        entry
        for entry in frontend._COMPONENT_ATTESTATION_MANIFEST
        if entry[0]
        in {
            "lowering",
            "state_guard",
            "fusion",
            "semantics",
            "model_contract",
            "pipeline",
        }
    )
    covered_by_namespace: dict[int, dict[str, object]] = {}
    declared_functions_by_namespace: dict[int, set[FunctionType]] = {}
    for _label, _module, source, _consumer, bindings, states in components:
        namespace_id = id(source)
        covered = covered_by_namespace.setdefault(namespace_id, {})
        declared = declared_functions_by_namespace.setdefault(namespace_id, set())
        for source_name, _consumer_name, value in bindings:
            covered[source_name] = value
            assert source[source_name] is value
        for source_name, _consumer_name, state in states:
            function = state[0]
            covered[source_name] = function
            declared.add(function)
            assert source[source_name] is function

    pending = [frontend._pipeline_component._parse_model_impl]
    visited: set[FunctionType] = set()
    while pending:
        function = pending.pop()
        if function in visited:
            continue
        visited.add(function)
        namespace = function.__globals__
        namespace_id = id(namespace)
        covered = covered_by_namespace[namespace_id]
        assert function in declared_functions_by_namespace[namespace_id]
        for code in _nested_code_objects(function.__code__):
            for instruction in dis.get_instructions(code):
                if instruction.opname not in {
                    "LOAD_GLOBAL",
                    "LOAD_FROM_DICT_OR_GLOBALS",
                }:
                    continue
                name = instruction.argval
                if name not in namespace:
                    continue
                assert name in covered, (
                    f"undeclared mutable component global: "
                    f"{function.__module__}.{function.__name__} -> {name}"
                )
                dependency = namespace[name]
                assert dependency is covered[name]
                if (
                    type(dependency) is FunctionType
                    and id(dependency.__globals__) in covered_by_namespace
                ):
                    pending.append(dependency)


@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_component_manifest_binding_mutation_fails_closed(mutation: str) -> None:
    original = frontend._COMPONENT_ATTESTATION_MANIFEST
    if mutation == "replace":
        frontend._COMPONENT_ATTESTATION_MANIFEST = ()
    else:
        del frontend._COMPONENT_ATTESTATION_MANIFEST
    try:
        with pytest.raises(
            UnsupportedOpError,
            match="trust root _COMPONENT_ATTESTATION_MANIFEST",
        ):
            parse_model(object(), input_shape=(1,))
    finally:
        frontend._COMPONENT_ATTESTATION_MANIFEST = original


def test_import_orders_share_one_component_baseline() -> None:
    program = """
import importlib
import json
import sys

for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
import torch2rtl
import torch2rtl.frontend as package
import torch2rtl.frontend.pytorch_fx as frontend
entry = frontend._COMPONENT_ATTESTATION_MANIFEST[0]
print(json.dumps({
    "label": entry[0],
    "bindings": [item[0] for item in entry[4]],
    "functions": [item[0] for item in entry[5]],
    "parse_identity": torch2rtl.parse_model is package.parse_model is frontend.parse_model,
    "error_identity": torch2rtl.UnsupportedOpError is package.UnsupportedOpError is frontend.UnsupportedOpError,
}, sort_keys=True))
"""
    orders = (
        ["torch2rtl"],
        ["torch2rtl.frontend.pytorch_fx", "torch2rtl"],
        ["torch2rtl.frontend", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._errors", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._lowering", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._state_guard", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._fusion", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._semantics", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._model_contract", "torch2rtl.frontend.pytorch_fx"],
        ["torch2rtl.frontend._pipeline", "torch2rtl.frontend.pytorch_fx"],
    )
    outputs = []
    for order in orders:
        completed = subprocess.run(
            [sys.executable, "-c", program, json.dumps(order)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0])["parse_identity"] is True
    assert json.loads(outputs[0])["error_identity"] is True
