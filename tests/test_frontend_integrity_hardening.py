from __future__ import annotations

import sys
from types import FunctionType, ModuleType

import numpy as np
import pytest
import torch
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
import torch2rtl.ir.graph as graph_module
import torch2rtl.quant.reference as quant_reference
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


_HOSTILE_CODE_CALLS: list[int] = []
_HOSTILE_IR_CALLS: list[int] = []
_HOSTILE_QUANT_CALLS: list[int] = []


def _hostile_replacement_code(*_args: object, **_kwargs: object) -> object:
    _HOSTILE_CODE_CALLS.append(1)
    raise AssertionError("hostile code executed")


def _clean_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()


def _assert_clean_parse() -> None:
    graph = parse_model(_clean_model(), input_shape=(2,))
    assert [type(operation).__name__ for operation in graph.ops] == [
        "LinearIR",
        "ReluIR",
    ]


@pytest.mark.parametrize(
    ("namespace", "binding_name"),
    [
        (frontend._lowering_component.__dict__, "_parse_module_node"),
        (frontend._fusion_component.__dict__, "_fuse_conv_batchnorm_eval"),
        (
            frontend._semantics_component.__dict__,
            "_validate_lowered_semantics",
        ),
        (frontend.__dict__, "GraphIR"),
        (frontend.__dict__, "TensorIR"),
        (frontend.__dict__, "LinearIR"),
        (frontend.__dict__, "Conv2dIR"),
        (frontend.__dict__, "ReluIR"),
        (frontend.__dict__, "FlattenIR"),
        (frontend.__dict__, "ArgmaxIR"),
    ],
)
def test_parser_dependency_replace_and_delete_never_executes(
    namespace: dict[str, object],
    binding_name: str,
) -> None:
    original = namespace[binding_name]
    calls = 0

    class Hostile:
        def __call__(self, *_args: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError(f"hostile {binding_name} executed")

    for mutation in ("replace", "delete"):
        if mutation == "replace":
            namespace[binding_name] = Hostile()
        else:
            del namespace[binding_name]
        try:
            with pytest.raises(UnsupportedOpError, match="modified frontend"):
                parse_model(_clean_model(), input_shape=(2,))
        finally:
            namespace[binding_name] = original
        assert calls == 0
        _assert_clean_parse()


@pytest.mark.parametrize(
    ("namespace", "function_name", "attribute", "replacement", "message"),
    [
        (
            frontend._lowering_component.__dict__,
            "_safe_name",
            "__defaults__",
            ("tampered",),
            "component lowering",
        ),
        (
            frontend._semantics_component.__dict__,
            "_run_concrete_probe",
            "__kwdefaults__",
            {"copy_model": False},
            "component semantics",
        ),
        (
            frontend._lowering_component.__dict__,
            "_flatten_shape",
            "__annotations__",
            {"tampered": object()},
            "component lowering",
        ),
        (
            frontend._lowering_component.__dict__,
            "_validate_input_shape",
            "__dict__",
            {"tampered": object()},
            "component lowering",
        ),
    ],
)
def test_function_state_mutation_is_rejected_and_restores_cleanly(
    namespace: dict[str, object],
    function_name: str,
    attribute: str,
    replacement: object,
    message: str,
) -> None:
    function = namespace[function_name]
    original = getattr(function, attribute)
    setattr(function, attribute, replacement)
    try:
        with pytest.raises(
            UnsupportedOpError,
            match=message,
        ):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        setattr(function, attribute, original)
    _assert_clean_parse()


def test_function_code_mutation_never_executes_and_restores_cleanly() -> None:
    function = frontend._lowering_component._parse_module_node
    original_code = function.__code__
    _HOSTILE_CODE_CALLS.clear()

    function.__code__ = _hostile_replacement_code.__code__
    try:
        with pytest.raises(
            UnsupportedOpError,
            match="component lowering",
        ):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        function.__code__ = original_code
    assert _HOSTILE_CODE_CALLS == []
    _assert_clean_parse()


def test_validator_declared_closure_mutation_is_rejected_before_call() -> None:
    validator = frontend._validate_framework_integrity
    freevars = validator.__code__.co_freevars
    index = freevars.index("implementation")
    cell = validator.__closure__[index]
    original = cell.cell_contents
    calls = 0

    def hostile(_nn: object) -> None:
        nonlocal calls
        calls += 1

    cell.cell_contents = hostile
    try:
        with pytest.raises(
            UnsupportedOpError,
            match="modified frontend integrity validator",
        ):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        cell.cell_contents = original
    assert calls == 0
    _assert_clean_parse()


def test_public_parser_dependencies_are_not_stored_in_writable_closures() -> None:
    assert parse_model.__closure__ is None
    assert parse_model.__code__.co_freevars == ()
    capsules = [
        value
        for value in parse_model.__code__.co_consts
        if isinstance(value, tuple)
        and type(value) is not tuple
        and len(value) == 20
        and tuple.__getitem__(value, 19) is parse_model
    ]
    assert len(capsules) == 1
    assert vars(type(capsules[0])).get("__slots__") == ()
    assert tuple.__getitem__(capsules[0], 19) is parse_model
    _assert_clean_parse()


def test_public_parser_capsule_ignores_mutated_subclass_lookup() -> None:
    capsule = next(
        value
        for value in parse_model.__code__.co_consts
        if isinstance(value, tuple)
        and type(value) is not tuple
        and len(value) == 20
        and tuple.__getitem__(value, 19) is parse_model
    )
    capsule_type = type(capsule)
    calls = 0

    def hostile_getitem(self: object, index: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"capsule lookup executed for {index!r}")

    capsule_type.__getitem__ = hostile_getitem
    try:
        _assert_clean_parse()
    finally:
        del capsule_type.__getitem__
    assert calls == 0


def _hostile_graph_init_factory() -> object:
    first = object()
    second = object()
    third = object()

    def hostile(*_args: object, **_kwargs: object) -> None:
        _HOSTILE_IR_CALLS.append(len((first, second, third)))
        raise AssertionError("hostile GraphIR constructor executed")

    return hostile


def test_in_place_graphir_constructor_mutation_never_executes() -> None:
    constructor = frontend.GraphIR.__init__
    original_code = constructor.__code__
    hostile_code = _hostile_graph_init_factory().__code__
    assert len(hostile_code.co_freevars) == len(original_code.co_freevars)
    _HOSTILE_IR_CALLS.clear()
    graph_module.__dict__["_HOSTILE_IR_CALLS"] = _HOSTILE_IR_CALLS
    constructor.__code__ = hostile_code
    try:
        with pytest.raises(UnsupportedOpError, match="compiler class"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        constructor.__code__ = original_code
        del graph_module.__dict__["_HOSTILE_IR_CALLS"]
    assert _HOSTILE_IR_CALLS == []
    _assert_clean_parse()


def _hostile_quant_init_factory() -> object:
    closed = object()

    def hostile(*_args: object, **_kwargs: object) -> None:
        _HOSTILE_QUANT_CALLS.append(id(closed))
        raise AssertionError("hostile quant result constructor executed")

    return hostile


@pytest.mark.parametrize("class_name", ["FloatReferenceResult", "ActivationResult"])
def test_in_place_quant_result_constructor_mutation_never_executes(
    class_name: str,
) -> None:
    constructor = getattr(quant_reference, class_name).__init__
    original_code = constructor.__code__
    hostile_code = _hostile_quant_init_factory().__code__
    assert len(hostile_code.co_freevars) == len(original_code.co_freevars)
    _HOSTILE_QUANT_CALLS.clear()
    quant_reference.__dict__["_HOSTILE_QUANT_CALLS"] = _HOSTILE_QUANT_CALLS
    constructor.__code__ = hostile_code
    try:
        with pytest.raises(UnsupportedOpError, match="compiler class"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        constructor.__code__ = original_code
        del quant_reference.__dict__["_HOSTILE_QUANT_CALLS"]
    assert _HOSTILE_QUANT_CALLS == []
    _assert_clean_parse()


@pytest.mark.parametrize("name", ["array", "asarray"])
def test_numpy_oracle_dependency_mutation_never_executes(name: str) -> None:
    original = getattr(np, name)
    calls = 0

    def hostile(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"hostile np.{name} executed")

    setattr(np, name, hostile)
    try:
        with pytest.raises(UnsupportedOpError, match="modified"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        setattr(np, name, original)
    assert calls == 0
    _assert_clean_parse()


def test_sys_modules_torch_alias_mutation_never_accesses_hostile_module() -> None:
    original = sys.modules["torch"]
    accesses = 0

    class HostileTorch(ModuleType):
        def __getattr__(self, name: str) -> object:
            nonlocal accesses
            accesses += 1
            raise AssertionError(f"hostile torch.{name} accessed")

    sys.modules["torch"] = HostileTorch("torch")
    try:
        with pytest.raises(UnsupportedOpError, match="sys.modules torch"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        sys.modules["torch"] = original
    assert accesses == 0
    _assert_clean_parse()


def test_no_grad_exit_code_mutation_is_rejected_before_callback() -> None:
    original = torch.no_grad.__exit__.__code__
    torch.no_grad.__exit__.__code__ = original.replace(
        co_firstlineno=original.co_firstlineno + 1
    )
    try:
        with pytest.raises(UnsupportedOpError, match="PyTorch framework class no_grad"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        torch.no_grad.__exit__.__code__ = original
    _assert_clean_parse()


@pytest.mark.parametrize(
    "class_name",
    ["UnsupportedOpError", "_PostCallbackIntegrityError"],
)
def test_error_class_new_mutation_never_executes(class_name: str) -> None:
    cls = getattr(frontend, class_name)
    calls = 0

    def hostile_new(target: type[object], *_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return BaseException.__new__(target)

    setattr(cls, "__new__", hostile_new)
    try:
        with pytest.raises(UnsupportedOpError, match="compiler class"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        delattr(cls, "__new__")
    assert calls == 0
    _assert_clean_parse()


def test_combined_error_class_and_parser_function_tamper_never_executes() -> None:
    error_calls = 0
    function_calls = 0
    namespace = frontend._lowering_component.__dict__
    original_function = namespace["_safe_name"]

    def hostile_new(target: type[object], *_args: object, **_kwargs: object) -> object:
        nonlocal error_calls
        error_calls += 1
        return RuntimeError.__new__(target)

    def hostile_function(*_args: object, **_kwargs: object) -> object:
        nonlocal function_calls
        function_calls += 1
        raise AssertionError("hostile parser function executed")

    setattr(UnsupportedOpError, "__new__", hostile_new)
    namespace["_safe_name"] = hostile_function
    try:
        with pytest.raises(UnsupportedOpError, match="compiler class"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        namespace["_safe_name"] = original_function
        delattr(UnsupportedOpError, "__new__")
    assert error_calls == 0
    assert function_calls == 0
    _assert_clean_parse()


def test_combined_error_class_and_runtime_root_tamper_never_executes() -> None:
    original_torch_module = sys.modules["torch"]
    error_calls = 0
    module_accesses = 0

    def hostile_new(target: type[object], *_args: object, **_kwargs: object) -> object:
        nonlocal error_calls
        error_calls += 1
        return RuntimeError.__new__(target)

    class HostileTorch(ModuleType):
        def __getattr__(self, name: str) -> object:
            nonlocal module_accesses
            module_accesses += 1
            raise AssertionError(f"hostile torch.{name} accessed")

    setattr(UnsupportedOpError, "__new__", hostile_new)
    sys.modules["torch"] = HostileTorch("torch")
    try:
        with pytest.raises(UnsupportedOpError, match="compiler class"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        sys.modules["torch"] = original_torch_module
        delattr(UnsupportedOpError, "__new__")
    assert error_calls == 0
    assert module_accesses == 0
    _assert_clean_parse()


@pytest.mark.parametrize("target", ["source", "consumer"])
def test_semantic_oracle_source_and_consumer_alias_are_sealed(target: str) -> None:
    calls = 0

    def hostile(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("hostile semantic oracle executed")

    if target == "source":
        owner = quant_reference
        name = "infer_float_graph"
    else:
        owner = frontend
        name = "_TRUSTED_INFER_FLOAT_GRAPH"
    original = owner.__dict__[name]
    owner.__dict__[name] = hostile
    try:
        with pytest.raises(UnsupportedOpError, match="modified"):
            parse_model(_clean_model(), input_shape=(2,))
    finally:
        owner.__dict__[name] = original
    assert calls == 0
    _assert_clean_parse()


def test_post_callback_failed_attestation_precedes_tampered_restore_helper() -> None:
    namespace = frontend._state_guard_component.__dict__
    original = namespace["_restore_module_class_definitions"]
    hostile_calls = 0

    def hostile_restore(*_args: object, **_kwargs: object) -> object:
        nonlocal hostile_calls
        hostile_calls += 1
        raise AssertionError("tampered restore helper executed")

    def user_callback() -> object:
        namespace["_restore_module_class_definitions"] = hostile_restore
        return object()

    try:
        with pytest.raises(frontend._PostCallbackIntegrityError):
            frontend._checked_untrusted_call(
                user_callback,
                (),
                {},
                frontend._TRUSTED_TORCH_NN_MODULE,
                frontend._TRUSTED_CALLBACK_SECURITY_CONTEXT,
            )
    finally:
        namespace["_restore_module_class_definitions"] = original
    assert hostile_calls == 0
    _assert_clean_parse()


def test_post_callback_attestation_runs_for_base_exception() -> None:
    class UserAbort(BaseException):
        pass

    abort = UserAbort("stop")

    def user_callback() -> object:
        raise abort

    with pytest.raises(UserAbort) as captured:
        frontend._checked_untrusted_call(
            user_callback,
            (),
            {},
            frontend._TRUSTED_TORCH_NN_MODULE,
            frontend._TRUSTED_CALLBACK_SECURITY_CONTEXT,
        )
    assert captured.value is abort
    _assert_clean_parse()


def test_grad_mode_adapter_restores_with_preloaded_runtime_binding() -> None:
    original = torch._C._set_grad_enabled
    calls = 0

    def replacement(_enabled: bool) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("changed grad-mode binding executed")

    def callback() -> object:
        assert torch.is_grad_enabled() is False
        torch._C._set_grad_enabled = replacement
        return object()

    try:
        with pytest.raises(frontend._PostCallbackIntegrityError):
            frontend._checked_untrusted_call(
                callback,
                (),
                {},
                frontend._TRUSTED_TORCH_NN_MODULE,
                frontend._TRUSTED_CALLBACK_SECURITY_CONTEXT,
                disable_grad=True,
            )
    finally:
        torch._C._set_grad_enabled = original
    assert calls == 0
    assert torch.is_grad_enabled() is True
    _assert_clean_parse()


def test_static_manifest_covers_transitive_frontend_function_dependencies() -> None:
    declared_functions = {
        state[0]
        for _name, state in frontend._PROTECTED_FRONTEND_FUNCTION_STATES
    }
    for component in frontend._COMPONENT_ATTESTATION_MANIFEST:
        declared_functions.update(
            state[0] for _source, _consumer, state in component[5]
        )
    pending = [frontend._parse_model_impl]
    visited: set[FunctionType] = set()
    while pending:
        function = pending.pop()
        if function in visited:
            continue
        visited.add(function)
        for name in function.__code__.co_names:
            dependency = function.__globals__.get(name)
            if (
                type(dependency) is FunctionType
                and dependency.__module__.startswith("torch2rtl.frontend")
            ):
                assert dependency in declared_functions, (
                    f"undeclared compiler function dependency: "
                    f"{function.__module__}.{function.__name__} -> {name}"
                )
                pending.append(dependency)
    assert visited <= declared_functions
