from __future__ import annotations

import builtins
import copy
import dis
import importlib.util
import inspect
import math
import sys
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Sequence

import numpy as np
import torch as _torch
import torch2rtl.quant.reference as _quant_reference

from torch2rtl.frontend import _errors as _errors_component
from torch2rtl.frontend import _fusion as _fusion_component
from torch2rtl.frontend import _lowering as _lowering_component
from torch2rtl.frontend import _model_contract as _model_contract_component
from torch2rtl.frontend import _pipeline as _pipeline_component
from torch2rtl.frontend import _semantics as _semantics_component
from torch2rtl.frontend import _state_guard as _state_guard_component
from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.ir.tensor import TensorIR


_parse_model_impl = _pipeline_component._parse_model_impl

# Bootstrap must not resolve these names through mutable builtins or a mutable
# frontend alias before the builtins integrity check has completed.
_TRUSTED_TYPE = type
_TRUSTED_DICT_TYPE = dict
_TRUSTED_STR_TYPE = str
_TRUSTED_CODE_TYPE = CodeType
_TRUSTED_FUNCTION_TYPE = FunctionType
_TRUSTED_MODULE_TYPE = ModuleType
_TRUSTED_MODULE_GETATTRIBUTE = ModuleType.__getattribute__
_TRUSTED_TYPE_GETATTRIBUTE = type.__getattribute__
_TRUSTED_FRONTEND_GLOBALS = globals()
_TRUSTED_FRONTEND_MODULE = sys.modules[__name__]
_TRUSTED_EMPTY_CELL = object()
_TRUSTED_QUANT_REFERENCE_MODULE = _quant_reference
_TRUSTED_QUANT_REFERENCE_GLOBALS = ModuleType.__getattribute__(
    _quant_reference,
    "__dict__",
)
_TRUSTED_INFER_FLOAT_GRAPH = _quant_reference.infer_float_graph
_TRUSTED_IS_GRAD_ENABLED = _torch.is_grad_enabled
_TRUSTED_SET_GRAD_ENABLED = _torch._C._set_grad_enabled
_TRUSTED_TORCH_C_MODULE = _torch._C
_TRUSTED_ERRORS_MODULE = _errors_component
_TRUSTED_ERRORS_GLOBALS = ModuleType.__getattribute__(
    _errors_component,
    "__dict__",
)
_TRUSTED_FUSION_MODULE = _fusion_component
_TRUSTED_FUSION_GLOBALS = ModuleType.__getattribute__(
    _fusion_component,
    "__dict__",
)
_TRUSTED_LOWERING_MODULE = _lowering_component
_TRUSTED_LOWERING_GLOBALS = ModuleType.__getattribute__(
    _lowering_component,
    "__dict__",
)
_TRUSTED_MODEL_CONTRACT_MODULE = _model_contract_component
_TRUSTED_MODEL_CONTRACT_GLOBALS = ModuleType.__getattribute__(
    _model_contract_component,
    "__dict__",
)
_TRUSTED_PIPELINE_MODULE = _pipeline_component
_TRUSTED_PIPELINE_GLOBALS = ModuleType.__getattribute__(
    _pipeline_component,
    "__dict__",
)
_TRUSTED_SEMANTICS_MODULE = _semantics_component
_TRUSTED_SEMANTICS_GLOBALS = ModuleType.__getattribute__(
    _semantics_component,
    "__dict__",
)
_TRUSTED_STATE_GUARD_MODULE = _state_guard_component
_TRUSTED_STATE_GUARD_GLOBALS = ModuleType.__getattribute__(
    _state_guard_component,
    "__dict__",
)


def _capture_function_integrity_state(function: FunctionType) -> tuple[object, ...]:
    def mapping_state(value: object) -> tuple[object, tuple[tuple[object, object], ...]]:
        if type(value) is not dict:
            return (value, ())
        return (value, tuple(value.items()))

    closure: list[tuple[object, object]] = []
    for cell in function.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:
            contents = _TRUSTED_EMPTY_CELL
        closure.append((cell, contents))
    return (
        function,
        function.__code__,
        function.__defaults__,
        mapping_state(function.__kwdefaults__),
        mapping_state(function.__annotations__),
        mapping_state(function.__dict__),
        tuple(closure),
    )


def _function_integrity_state_matches(
    function: object,
    state: tuple[object, ...],
    trusted_type: type[object],
    trusted_function_type: type[FunctionType],
    trusted_dict_type: type[dict[object, object]],
    trusted_str_type: type[str],
    trusted_len: object,
    trusted_value_error: type[ValueError],
    empty_cell: object,
) -> bool:
    if trusted_type(function) is not trusted_function_type:
        return False
    (
        expected_function,
        expected_code,
        expected_defaults,
        expected_kwdefaults,
        expected_annotations,
        expected_function_dict,
        expected_closure,
    ) = state
    if (
        function is not expected_function
        or function.__code__ is not expected_code
        or function.__defaults__ is not expected_defaults
    ):
        return False
    for current, expected in (
        (function.__kwdefaults__, expected_kwdefaults),
        (function.__annotations__, expected_annotations),
        (function.__dict__, expected_function_dict),
    ):
        expected_mapping, expected_items = expected
        if current is not expected_mapping:
            return False
        if expected_mapping is None:
            continue
        if trusted_type(current) is not trusted_dict_type:
            return False
        if trusted_len(current) != trusted_len(expected_items):
            return False
        for key in current:
            if trusted_type(key) is not trusted_str_type:
                return False
        for key, value in expected_items:
            if current.get(key, empty_cell) is not value:
                return False
    current_closure = function.__closure__ or ()
    if trusted_len(current_closure) != trusted_len(expected_closure):
        return False
    index = 0
    for current_cell in current_closure:
        expected = expected_closure[index]
        index += 1
        expected_cell, expected_contents = expected
        if current_cell is not expected_cell:
            return False
        try:
            current_contents = current_cell.cell_contents
        except trusted_value_error:
            current_contents = empty_cell
        if current_contents is not expected_contents:
            return False
    return True


def _require_exact_string_dict(value: object, label: str) -> dict[str, object]:
    if _TRUSTED_TYPE(value) is not _TRUSTED_DICT_TYPE:
        raise UnsupportedOpError(f"Unsupported modified {label} mapping")
    for key in value:
        if _TRUSTED_TYPE(key) is not _TRUSTED_STR_TYPE:
            raise UnsupportedOpError(f"Unsupported modified {label} keys")
    return value


def _definition_fingerprint(value: object, seen: set[int] | None = None) -> object:
    active = set() if seen is None else seen
    value_type = _TRUSTED_TYPE(value)
    if (
        value is None
        or value_type is bool
        or value_type is int
        or value_type is float
        or value_type is complex
        or value_type is str
        or value_type is bytes
    ):
        return (value_type, value)
    identity = id(value)
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        if value_type is _TRUSTED_FUNCTION_TYPE:
            function_globals = _require_exact_string_dict(
                value.__globals__,
                "function globals",
            )
            closure = tuple(
                _definition_fingerprint(cell.cell_contents, active)
                for cell in (value.__closure__ or ())
            )
            referenced_globals = tuple(
                (name, _definition_fingerprint(function_globals[name], active))
                for name in value.__code__.co_names
                if name in function_globals
            )
            return (
                "function",
                identity,
                id(value.__code__),
                _definition_fingerprint(value.__defaults__, active),
                _definition_fingerprint(value.__kwdefaults__, active),
                _definition_fingerprint(value.__annotations__, active),
                _definition_fingerprint(value.__dict__, active),
                closure,
                referenced_globals,
            )
        if value_type is staticmethod or value_type is classmethod:
            return (value_type, _definition_fingerprint(value.__func__, active))
        if value_type is property:
            return (
                "property",
                _definition_fingerprint(value.fget, active),
                _definition_fingerprint(value.fset, active),
                _definition_fingerprint(value.fdel, active),
            )
        if value_type is dict:
            return (
                "dict",
                frozenset(
                    (
                        _definition_fingerprint(key, active),
                        _definition_fingerprint(item, active),
                    )
                    for key, item in value.items()
                ),
            )
        if value_type is tuple or value_type is list:
            return (
                value_type,
                tuple(_definition_fingerprint(item, active) for item in value),
            )
        if value_type is set or value_type is frozenset:
            return (
                value_type,
                frozenset(_definition_fingerprint(item, active) for item in value),
            )
        return ("identity", identity)
    finally:
        active.remove(identity)


def _shallow_state_fingerprint(
    value: object,
    seen: set[int] | None = None,
) -> object:
    active = set() if seen is None else seen
    value_type = _TRUSTED_TYPE(value)
    if (
        value is None
        or value_type is bool
        or value_type is int
        or value_type is float
        or value_type is complex
        or value_type is str
        or value_type is bytes
    ):
        return (value_type, value)
    if value_type is _TRUSTED_FUNCTION_TYPE:
        return ("function", id(value), id(value.__code__))
    if value_type is _TRUSTED_CODE_TYPE:
        return ("code", id(value))
    identity = id(value)
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        if value_type is dict:
            return (
                "dict",
                frozenset(
                    (
                        _shallow_state_fingerprint(key, active),
                        _shallow_state_fingerprint(item, active),
                    )
                    for key, item in value.items()
                ),
            )
        if value_type is tuple or value_type is list:
            return (
                value_type,
                tuple(_shallow_state_fingerprint(item, active) for item in value),
            )
        if value_type is set or value_type is frozenset:
            return (
                value_type,
                frozenset(
                    _shallow_state_fingerprint(item, active) for item in value
                ),
            )
        return ("identity", id(value_type), identity)
    finally:
        active.remove(identity)


def _function_local_fingerprint(function: FunctionType) -> object:
    closure: list[object] = []
    for cell in function.__closure__ or ():
        try:
            contents = _shallow_state_fingerprint(cell.cell_contents)
        except ValueError:
            contents = ("empty",)
        closure.append((id(cell), contents))
    return (
        id(function.__code__),
        _shallow_state_fingerprint(function.__defaults__),
        _shallow_state_fingerprint(function.__kwdefaults__),
        _shallow_state_fingerprint(function.__annotations__),
        _shallow_state_fingerprint(function.__dict__),
        tuple(closure),
    )


def _descriptor_local_fingerprint(value: object) -> object:
    value_type = _TRUSTED_TYPE(value)
    if value_type is _TRUSTED_FUNCTION_TYPE:
        return ("function", _function_local_fingerprint(value))
    if value_type is staticmethod or value_type is classmethod:
        return (
            value_type,
            _function_local_fingerprint(value.__func__),
        )
    if value_type is property:
        return (
            "property",
            tuple(
                _function_local_fingerprint(function)
                if type(function) is FunctionType
                else None
                for function in (value.fget, value.fset, value.fdel)
            ),
        )
    return ("identity", id(value_type), id(value))


def _class_local_fingerprint(cls: type[object]) -> object:
    namespace = type.__getattribute__(cls, "__dict__")
    return tuple(
        (name, id(value), _descriptor_local_fingerprint(value))
        for name, value in sorted(namespace.items(), key=lambda item: item[0])
    )


def _function_referenced_globals_fingerprint(function: FunctionType) -> object:
    function_globals = _require_exact_string_dict(
        function.__globals__,
        "function globals",
    )
    return (
        id(function_globals),
        tuple(
            (
                name,
                id(function_globals[name]),
                _function_local_fingerprint(function_globals[name])
                if _TRUSTED_TYPE(function_globals[name]) is _TRUSTED_FUNCTION_TYPE
                else _shallow_state_fingerprint(function_globals[name]),
            )
            for name in function.__code__.co_names
            if name in function_globals
        ),
    )


_TRUSTED_FRAMEWORK_TYPES = (
    _torch.nn.Module,
    _torch.nn.Sequential,
    _torch.nn.Identity,
    _torch.nn.Linear,
    _torch.nn.Conv2d,
    _torch.nn.BatchNorm2d,
    _torch.nn.ReLU,
    _torch.nn.Flatten,
)
_TRUSTED_TORCH_MODULE = _torch
_TRUSTED_TORCH_NN_MODULE = _torch.nn
_TRUSTED_PARAMETER_TYPE = _torch.nn.Parameter
_TRUSTED_BUILTINS_MODULE = builtins
_TRUSTED_BUILTINS = builtins.__dict__
_TRUSTED_LEN = builtins.len
_TRUSTED_BUILTIN_BINDINGS = tuple(
    (name, builtins.__dict__[name])
    for name in (
        "AssertionError",
        "AttributeError",
        "Exception",
        "KeyError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "__import__",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "classmethod",
        "complex",
        "dict",
        "enumerate",
        "float",
        "frozenset",
        "getattr",
        "hasattr",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "max",
        "min",
        "next",
        "object",
        "property",
        "range",
        "repr",
        "reversed",
        "set",
        "sorted",
        "str",
        "staticmethod",
        "sum",
        "tuple",
        "type",
        "vars",
        "zip",
    )
)
_TRUSTED_COPY_MODULE = copy
_TRUSTED_DEEPCOPY = copy.deepcopy
_TRUSTED_DEEPCOPY_CODE = copy.deepcopy.__code__
_TRUSTED_DEEPCOPY_LOCAL_STATE = _function_local_fingerprint(copy.deepcopy)
_TRUSTED_DEEPCOPY_DISPATCH = copy._deepcopy_dispatch
_TRUSTED_DEEPCOPY_DISPATCH_ITEMS = tuple(copy._deepcopy_dispatch.items())
_TRUSTED_DEEPCOPY_DISPATCH_FUNCTIONS = tuple(
    (value, _function_local_fingerprint(value))
    for _, value in _TRUSTED_DEEPCOPY_DISPATCH_ITEMS
    if type(value) is FunctionType
)
_TRUSTED_COPY_DISPATCH_TABLE = copy.dispatch_table
_TRUSTED_COPY_DISPATCH_TABLE_ITEMS = tuple(copy.dispatch_table.items())
_TRUSTED_COPY_DISPATCH_TABLE_FUNCTIONS = tuple(
    (value, _function_local_fingerprint(value))
    for _, value in _TRUSTED_COPY_DISPATCH_TABLE_ITEMS
    if type(value) is FunctionType
)
_TRUSTED_COPY_DEEPCOPY_GLOBALS = tuple(
    (
        name,
        value,
        _function_local_fingerprint(value)
        if type(value) is FunctionType
        else None,
    )
    for name, value in (
        ("_deepcopy_atomic", copy._deepcopy_atomic),
        ("_reconstruct", copy._reconstruct),
        ("_keep_alive", copy._keep_alive),
        ("Error", copy.Error),
    )
)
_TRUSTED_DIS_MODULE = dis
_TRUSTED_DIS_BYTECODE = dis.Bytecode
_TRUSTED_DIS_BYTECODE_METHODS = tuple(
    (
        name,
        value,
        _function_local_fingerprint(value),
        _function_referenced_globals_fingerprint(value),
    )
    for name, value in (
        ("__init__", dis.Bytecode.__init__),
        ("__iter__", dis.Bytecode.__iter__),
    )
)
_TRUSTED_DIS_CLASS_STATES = tuple(
    (cls, _class_local_fingerprint(cls))
    for cls in (
        dis.Bytecode,
        dis.Instruction,
        dis._Instruction,
        dis.Positions,
        dis._ExceptionTableEntry,
    )
)
_TRUSTED_DIS_NAMEDTUPLE_GLOBALS = tuple(
    (
        constructor,
        constructor.__globals__,
        constructor.__globals__["_tuple_new"],
    )
    for constructor in (
        type.__getattribute__(dis.Positions, "__dict__")["__new__"].__func__,
        type.__getattribute__(dis._Instruction, "__dict__")["__new__"].__func__,
        type.__getattribute__(dis._ExceptionTableEntry, "__dict__")[
            "__new__"
        ].__func__,
    )
)
_TRUSTED_DIS_CALLABLE_BINDINGS = tuple(
    (
        name,
        value,
        _function_local_fingerprint(value)
        if type(value) is FunctionType
        else None,
    )
    for name, value in (
        ("_get_code_object", dis._get_code_object),
        ("findlinestarts", dis.findlinestarts),
        ("_parse_exception_table", dis._parse_exception_table),
        ("_get_instructions_bytes", dis._get_instructions_bytes),
        ("_get_code_array", dis._get_code_array),
        ("_try_compile", dis._try_compile),
        ("_parse_varint", dis._parse_varint),
        ("findlabels", dis.findlabels),
        ("_unpack_opargs", dis._unpack_opargs),
        ("_deoptop", dis._deoptop),
        ("_get_const_value", dis._get_const_value),
        ("_get_const_info", dis._get_const_info),
        ("_get_name_info", dis._get_name_info),
        ("_is_backward_jump", dis._is_backward_jump),
        ("_ExceptionTableEntry", dis._ExceptionTableEntry),
        ("Positions", dis.Positions),
        ("Instruction", dis.Instruction),
    )
)
_TRUSTED_DIS_RUNTIME_STATE = tuple(
    (name, value, _shallow_state_fingerprint(value))
    for name, value in (
        ("__name__", dis.__name__),
        ("BINARY_OP", dis.BINARY_OP),
        ("CACHE", dis.CACHE),
        ("CALL_INTRINSIC_1", dis.CALL_INTRINSIC_1),
        ("CALL_INTRINSIC_2", dis.CALL_INTRINSIC_2),
        ("EXTENDED_ARG", dis.EXTENDED_ARG),
        ("FORMAT_VALUE", dis.FORMAT_VALUE),
        ("FORMAT_VALUE_CONVERTERS", dis.FORMAT_VALUE_CONVERTERS),
        ("Instruction", dis.Instruction),
        ("_Instruction", dis._Instruction),
        ("LOAD_ATTR", dis.LOAD_ATTR),
        ("LOAD_GLOBAL", dis.LOAD_GLOBAL),
        ("LOAD_SUPER_ATTR", dis.LOAD_SUPER_ATTR),
        ("MAKE_FUNCTION", dis.MAKE_FUNCTION),
        ("MAKE_FUNCTION_FLAGS", dis.MAKE_FUNCTION_FLAGS),
        ("Positions", dis.Positions),
        ("UNKNOWN", dis.UNKNOWN),
        ("_ExceptionTableEntry", dis._ExceptionTableEntry),
        ("_INT_OVERFLOW", dis._INT_OVERFLOW),
        ("_all_opmap", dis._all_opmap),
        ("_all_opname", dis._all_opname),
        ("_cache_format", dis._cache_format),
        ("_inline_cache_entries", dis._inline_cache_entries),
        ("_intrinsic_1_descs", dis._intrinsic_1_descs),
        ("_intrinsic_2_descs", dis._intrinsic_2_descs),
        ("_nb_ops", dis._nb_ops),
        ("cmp_op", dis.cmp_op),
        ("deoptmap", dis.deoptmap),
        ("hasarg", dis.hasarg),
        ("hascompare", dis.hascompare),
        ("hasconst", dis.hasconst),
        ("hasfree", dis.hasfree),
        ("hasjabs", dis.hasjabs),
        ("hasjrel", dis.hasjrel),
        ("haslocal", dis.haslocal),
        ("hasname", dis.hasname),
        ("opname", dis.opname),
        ("sys", dis.sys),
    )
)
_TRUSTED_DIS_HASJABS = dis.hasjabs
_TRUSTED_DIS_HASJABS_CONTENTS = dis.hasjabs.copy()
_TRUSTED_DIS_HASJREL = dis.hasjrel
_TRUSTED_DIS_HASJREL_CONTENTS = dis.hasjrel.copy()
_TRUSTED_INSPECT_MODULE = inspect
_TRUSTED_INSPECT_GETATTR_STATIC = inspect.getattr_static
_TRUSTED_INSPECT_UNWRAP = inspect.unwrap
_TRUSTED_MATH_MODULE = math
_TRUSTED_MATH_ISFINITE = math.isfinite
_TRUSTED_MATH_PROD = math.prod
_TRUSTED_NUMPY_MODULE = np
_TRUSTED_SYS_MODULE = sys
_TRUSTED_SYS_GETRECURSIONLIMIT = sys.getrecursionlimit
_TRUSTED_SYS_BYTEORDER = sys.byteorder
_TRUSTED_SYS_MODULES = sys.modules
_TRUSTED_BOOTSTRAP_GLOBAL_BINDINGS = (
    ("CodeType", _TRUSTED_CODE_TYPE, "Unsupported modified CodeType binding"),
    (
        "FunctionType",
        _TRUSTED_FUNCTION_TYPE,
        "Unsupported modified FunctionType binding",
    ),
    ("ModuleType", _TRUSTED_MODULE_TYPE, "Unsupported modified ModuleType binding"),
    ("builtins", _TRUSTED_BUILTINS_MODULE, "Unsupported modified builtins module binding"),
    ("copy", _TRUSTED_COPY_MODULE, "Unsupported modified copy module binding"),
    ("dis", _TRUSTED_DIS_MODULE, "Unsupported modified dis module binding"),
    ("inspect", _TRUSTED_INSPECT_MODULE, "Unsupported modified inspect module binding"),
    ("math", _TRUSTED_MATH_MODULE, "Unsupported modified math module binding"),
    ("np", _TRUSTED_NUMPY_MODULE, "Unsupported modified NumPy module binding"),
    ("sys", _TRUSTED_SYS_MODULE, "Unsupported modified sys module binding"),
)
_TRUSTED_TENSOR_COPY_BINDINGS = tuple(
    (
        owner,
        _raw_name,
        value,
        _function_local_fingerprint(value),
        _function_referenced_globals_fingerprint(value),
    )
    for owner, _raw_name, value in (
        (_torch.Tensor, "__deepcopy__", _torch.Tensor.__deepcopy__),
        (
            _torch.nn.Parameter,
            "__deepcopy__",
            _torch.nn.Parameter.__deepcopy__,
        ),
    )
)
_TRUSTED_NUMPY_BINDINGS = (
    (np, "all", np.all),
    (np, "allclose", np.allclose),
    (np, "any", np.any),
    (np, "argmax", np.argmax),
    (np, "array", np.array),
    (np, "array_equal", np.array_equal),
    (np, "asarray", np.asarray),
    (np, "ascontiguousarray", np.ascontiguousarray),
    (np, "dtype", np.dtype),
    (np, "float32", np.float32),
    (np, "float64", np.float64),
    (np, "int64", np.int64),
    (np, "isfinite", np.isfinite),
    (np, "linspace", np.linspace),
    (np, "ndarray", np.ndarray),
    (np, "zeros", np.zeros),
)
_TRUSTED_FRAMEWORK_CLASSES = frozenset(
    cls for module_type in _TRUSTED_FRAMEWORK_TYPES for cls in module_type.__mro__
)
_TRUSTED_FX_CLASSES = (
    _torch.fx.Tracer,
    _torch.fx.Graph,
    _torch.fx.Node,
    _torch.fx.GraphModule,
)
_TRUSTED_FRAMEWORK_DEFINITIONS = tuple(
    (
        cls,
        {
            name: (value, _definition_fingerprint(value))
            for name, value in vars(cls).items()
        },
    )
    for cls in (
        *_TRUSTED_FRAMEWORK_CLASSES,
        *_TRUSTED_FX_CLASSES,
        _torch.no_grad,
    )
)
_TRUSTED_FUNCTION_BINDINGS = (
    *_TRUSTED_NUMPY_BINDINGS,
    (_torch, "nn", _torch.nn),
    (_torch, "_C", _torch._C),
    (_torch, "fx", _torch.fx),
    (_torch, "backends", _torch.backends),
    (_torch.backends, "cudnn", _torch.backends.cudnn),
    (_torch.nn, "functional", _torch.nn.functional),
    (_torch.nn, "modules", _torch.nn.modules),
    (_torch.nn, "utils", _torch.nn.utils),
    (_torch.nn, "BatchNorm2d", _torch.nn.BatchNorm2d),
    (_torch.nn.modules, "module", _torch.nn.modules.module),
    (_torch.nn.modules, "batchnorm", _torch.nn.modules.batchnorm),
    (
        _torch.nn.modules.batchnorm,
        "BatchNorm2d",
        _torch.nn.modules.batchnorm.BatchNorm2d,
    ),
    (
        _torch.nn.modules.batchnorm,
        "_BatchNorm",
        _torch.nn.modules.batchnorm._BatchNorm,
    ),
    (
        _torch.nn.modules.batchnorm,
        "_NormBase",
        _torch.nn.modules.batchnorm._NormBase,
    ),
    (_torch.nn.utils, "fusion", _torch.nn.utils.fusion),
    (
        _torch.nn.utils.fusion,
        "fuse_conv_bn_eval",
        _torch.nn.utils.fusion.fuse_conv_bn_eval,
    ),
    (
        _torch.nn.utils.fusion,
        "fuse_conv_bn_weights",
        _torch.nn.utils.fusion.fuse_conv_bn_weights,
    ),
    (_torch.nn.functional, "linear", _torch.nn.functional.linear),
    (_torch.nn.functional, "conv2d", _torch.nn.functional.conv2d),
    (_torch.nn.functional, "batch_norm", _torch.nn.functional.batch_norm),
    (_torch.nn.functional, "relu", _torch.nn.functional.relu),
    (_torch, "batch_norm", _torch.batch_norm),
    (_torch, "zeros_like", _torch.zeros_like),
    (_torch, "ones_like", _torch.ones_like),
    (_torch, "rsqrt", _torch.rsqrt),
    (_torch, "relu", _torch.relu),
    (_torch, "relu_", _torch.relu_),
    (_torch, "argmax", _torch.argmax),
    (_torch, "from_numpy", _torch.from_numpy),
    (_torch, "is_grad_enabled", _torch.is_grad_enabled),
    (_torch, "no_grad", _torch.no_grad),
    (_torch, "get_default_dtype", _torch.get_default_dtype),
    (_torch, "Tensor", _torch.Tensor),
    (_torch, "device", _torch.device),
    (_torch, "float32", _torch.float32),
    (_torch, "float64", _torch.float64),
    (_torch, "int64", _torch.int64),
    (_torch, "preserve_format", _torch.preserve_format),
    (_torch, "strided", _torch.strided),
    (_torch._C, "_set_grad_enabled", _torch._C._set_grad_enabled),
    (_torch.fx, "symbolic_trace", _torch.fx.symbolic_trace),
    (_torch.Tensor, "argmax", _torch.Tensor.argmax),
    (_torch.Tensor, "dim", _torch.Tensor.dim),
    (_torch.Tensor, "flatten", _torch.Tensor.flatten),
    (_torch.Tensor, "reshape", _torch.Tensor.reshape),
    (_torch.Tensor, "unsqueeze", _torch.Tensor.unsqueeze),
    (_torch.Tensor, "__add__", _torch.Tensor.__add__),
    (_torch.Tensor, "__mul__", _torch.Tensor.__mul__),
    (_torch.Tensor, "__sub__", _torch.Tensor.__sub__),
    (_torch.Tensor, "detach", _torch.Tensor.detach),
    (_torch.Tensor, "numpy", _torch.Tensor.numpy),
    (_torch.Tensor, "cpu", _torch.Tensor.cpu),
    (_torch.Tensor, "contiguous", _torch.Tensor.contiguous),
    (_torch.Tensor, "clone", _torch.Tensor.clone),
    (_torch.Tensor, "resolve_conj", _torch.Tensor.resolve_conj),
    (_torch.Tensor, "resolve_neg", _torch.Tensor.resolve_neg),
    (_torch.Tensor, "is_contiguous", _torch.Tensor.is_contiguous),
    (_torch.Tensor, "is_conj", _torch.Tensor.is_conj),
    (_torch.Tensor, "is_neg", _torch.Tensor.is_neg),
    (_torch.Tensor, "stride", _torch.Tensor.stride),
    (_torch.Tensor, "storage_offset", _torch.Tensor.storage_offset),
    (_torch.Tensor, "to", _torch.Tensor.to),
    (_torch.Tensor, "device", _torch.Tensor.device),
    (_torch.Tensor, "data", _torch.Tensor.data),
    (_torch.Tensor, "dtype", _torch.Tensor.dtype),
    (_torch.Tensor, "layout", _torch.Tensor.layout),
    (_torch.Tensor, "shape", _torch.Tensor.shape),
    (_torch.Tensor, "grad", _torch.Tensor.grad),
    (_torch.Tensor, "requires_grad", _torch.Tensor.requires_grad),
)
_TRUSTED_RUNTIME_MODULES = tuple(
    (module, type(module))
    for module in (
        builtins,
        copy,
        dis,
        inspect,
        math,
        np,
        sys,
        _torch,
        _torch.backends,
        _torch.fx,
        _torch.nn,
        _torch.nn.functional,
        _torch.nn.modules,
        _torch.nn.modules.batchnorm,
        _torch.nn.modules.module,
        _torch.nn.utils,
        _torch.nn.utils.fusion,
    )
)
_TRUSTED_FUNCTION_FINGERPRINTS = tuple(
    (owner, name, function, _definition_fingerprint(function))
    for owner, name, function in _TRUSTED_FUNCTION_BINDINGS
)
del _torch


UnsupportedOpError = _errors_component.UnsupportedOpError


class _PostCallbackIntegrityError(BaseException):
    """Internal non-Exception escape used when callback re-attestation fails."""


_TRUSTED_UNSUPPORTED_OP_ERROR = UnsupportedOpError


def load_model_from_file(path: Path) -> object:
    module = _load_python_module(path)
    if hasattr(module, "create_model"):
        model = module.create_model()
    elif hasattr(module, "model"):
        model = module.model
    else:
        raise ValueError(
            f"{path} must define create_model() or a global variable named model"
        )
    if hasattr(model, "eval"):
        model.eval()
    return model


def _checked_untrusted_call(
    callback: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    nn: object,
    security_context: tuple[object, ...],
    *,
    disable_grad: bool = False,
) -> object:
    (
        validator,
        validator_state,
        state_matcher,
        state_matcher_code,
        trusted_type,
        trusted_function_type,
        trusted_dict_type,
        trusted_str_type,
        trusted_len,
        trusted_value_error,
        empty_cell,
        trusted_error,
        trusted_base_exception,
        trusted_integrity_marker,
        trusted_is_grad_enabled,
        trusted_set_grad_enabled,
    ) = security_context
    integrity_marker = trusted_integrity_marker()
    callback_error: BaseException | None = None
    result: object | None = None
    previous_grad_mode = False
    if disable_grad:
        previous_grad_mode = trusted_is_grad_enabled()
        trusted_set_grad_enabled(False)
    try:
        try:
            result = callback(*args, **kwargs)
        except trusted_base_exception as exc:
            callback_error = exc
    finally:
        if disable_grad:
            # This C-level primitive was copied to the active frame before the
            # callback. Restoring grad mode completes the controlled adapter;
            # the consistency check below is the next compiler action.
            trusted_set_grad_enabled(previous_grad_mode)

    # This is deliberately the first trusted action after callback return or
    # raise. The active frame still executes its original code object, and all
    # values below were copied to fast locals before entering user code.
    if state_matcher.__code__ is not state_matcher_code or not state_matcher(
        validator,
        validator_state,
        trusted_type,
        trusted_function_type,
        trusted_dict_type,
        trusted_str_type,
        trusted_len,
        trusted_value_error,
        empty_cell,
    ):
        raise integrity_marker
    try:
        validator(nn)
    except trusted_base_exception:
        raise integrity_marker from None
    if callback_error is not None:
        raise callback_error
    return result


def _raw_static_attribute(owner: object, name: str) -> object:
    namespace = object.__getattribute__(owner, "__dict__")
    if type(namespace) is dict:
        _require_exact_string_dict(namespace, "runtime namespace")
        if name in namespace:
            return namespace[name]
        raise AttributeError(name)
    for cls in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(cls, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def _raw_python_module_namespace(module: ModuleType) -> dict[str, object]:
    namespace = _TRUSTED_MODULE_GETATTRIBUTE(module, "__dict__")
    return _require_exact_string_dict(namespace, "runtime module namespace")


def _framework_module_classes(nn: object) -> frozenset[type[object]]:
    namespace = _raw_python_module_namespace(nn)
    current_types = (
        namespace.get("Module"),
        namespace.get("Sequential"),
        namespace.get("Identity"),
        namespace.get("Linear"),
        namespace.get("Conv2d"),
        namespace.get("BatchNorm2d"),
        namespace.get("ReLU"),
        namespace.get("Flatten"),
    )
    if any(
        current is not trusted
        for current, trusted in zip(
            current_types,
            _TRUSTED_FRAMEWORK_TYPES,
            strict=True,
        )
    ):
        raise UnsupportedOpError("Unsupported modified PyTorch framework types")
    return _TRUSTED_FRAMEWORK_CLASSES


# These helpers are the transitive Python call graph of the integrity check.
# Pin their module bindings before invoking any of them.
_TRUSTED_INTEGRITY_HELPER_BINDINGS = (
    ("_class_local_fingerprint", _class_local_fingerprint),
    ("_definition_fingerprint", _definition_fingerprint),
    ("_descriptor_local_fingerprint", _descriptor_local_fingerprint),
    ("_framework_module_classes", _framework_module_classes),
    ("_function_local_fingerprint", _function_local_fingerprint),
    (
        "_function_referenced_globals_fingerprint",
        _function_referenced_globals_fingerprint,
    ),
    ("_raw_python_module_namespace", _raw_python_module_namespace),
    ("_raw_static_attribute", _raw_static_attribute),
    ("_require_exact_string_dict", _require_exact_string_dict),
    ("_shallow_state_fingerprint", _shallow_state_fingerprint),
)


def _validate_framework_integrity_impl(nn: object) -> None:
    for key in _TRUSTED_FRONTEND_GLOBALS:
        if _TRUSTED_TYPE(key) is not _TRUSTED_STR_TYPE:
            raise _TRUSTED_UNSUPPORTED_OP_ERROR(
                "Unsupported modified frontend runtime keys"
            )
    if (
        _TRUSTED_FRONTEND_GLOBALS.get("UnsupportedOpError")
        is not _TRUSTED_UNSUPPORTED_OP_ERROR
    ):
        raise _TRUSTED_UNSUPPORTED_OP_ERROR(
            "Unsupported modified frontend error binding"
        )
    if (
        _TRUSTED_FRONTEND_GLOBALS.get("_validate_framework_integrity")
        is not _TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY
    ):
        raise _TRUSTED_UNSUPPORTED_OP_ERROR(
            "Unsupported modified frontend integrity validator"
        )
    for name, trusted, error in _TRUSTED_BOOTSTRAP_GLOBAL_BINDINGS:
        if _TRUSTED_FRONTEND_GLOBALS.get(name) is not trusted:
            raise UnsupportedOpError(error)
    for name, function in _TRUSTED_INTEGRITY_HELPER_BINDINGS:
        if _TRUSTED_FRONTEND_GLOBALS.get(name) is not function:
            raise UnsupportedOpError(
                f"Unsupported modified frontend integrity helper {name}"
            )
    for module, module_type in _TRUSTED_RUNTIME_MODULES:
        if _TRUSTED_TYPE(module) is not module_type:
            raise UnsupportedOpError("Unsupported modified runtime module type")

    builtins_state = _raw_python_module_namespace(builtins)
    if builtins_state is not _TRUSTED_BUILTINS:
        raise UnsupportedOpError("Unsupported modified builtins state mapping")
    for name, value in _TRUSTED_BUILTIN_BINDINGS:
        if builtins_state.get(name) is not value:
            raise UnsupportedOpError(f"Unsupported modified builtins binding {name}")

    if (
        _raw_python_module_namespace(sys).get("getrecursionlimit")
        is not _TRUSTED_SYS_GETRECURSIONLIMIT
        or _raw_python_module_namespace(sys).get("byteorder")
        is not _TRUSTED_SYS_BYTEORDER
    ):
        raise UnsupportedOpError("Unsupported modified sys runtime bindings")
    current_sys_modules = _raw_python_module_namespace(sys).get("modules")
    if (
        _TRUSTED_TYPE(current_sys_modules) is not _TRUSTED_DICT_TYPE
        or current_sys_modules is not _TRUSTED_SYS_MODULES
        or current_sys_modules.get("torch") is not _TRUSTED_TORCH_MODULE
    ):
        raise UnsupportedOpError("Unsupported modified sys.modules torch binding")

    inspect_state = _raw_python_module_namespace(inspect)
    if inspect_state.get("getattr_static") is not _TRUSTED_INSPECT_GETATTR_STATIC:
        raise UnsupportedOpError("Unsupported modified inspect.getattr_static binding")
    if inspect_state.get("unwrap") is not _TRUSTED_INSPECT_UNWRAP:
        raise UnsupportedOpError("Unsupported modified inspect.unwrap binding")

    copy_state = _raw_python_module_namespace(copy)
    copy_deepcopy = copy_state.get("deepcopy")
    if copy_deepcopy is not _TRUSTED_DEEPCOPY:
        raise UnsupportedOpError("Unsupported modified copy.deepcopy function")
    if (
        copy_deepcopy.__code__ is not _TRUSTED_DEEPCOPY_CODE
        or _function_local_fingerprint(copy_deepcopy)
        != _TRUSTED_DEEPCOPY_LOCAL_STATE
    ):
        raise UnsupportedOpError("Unsupported modified copy.deepcopy function")
    deepcopy_dispatch = copy_state.get("_deepcopy_dispatch")
    copy_dispatch_table = copy_state.get("dispatch_table")
    if (
        deepcopy_dispatch is not _TRUSTED_DEEPCOPY_DISPATCH
        or copy_dispatch_table is not _TRUSTED_COPY_DISPATCH_TABLE
        or _TRUSTED_LEN(deepcopy_dispatch)
        != _TRUSTED_LEN(_TRUSTED_DEEPCOPY_DISPATCH_ITEMS)
        or _TRUSTED_LEN(copy_dispatch_table)
        != _TRUSTED_LEN(_TRUSTED_COPY_DISPATCH_TABLE_ITEMS)
    ):
        raise UnsupportedOpError("Unsupported modified model copy dispatch")
    for key, value in deepcopy_dispatch.items():
        if not any(
            key is trusted_key and value is trusted_value
            for trusted_key, trusted_value in _TRUSTED_DEEPCOPY_DISPATCH_ITEMS
        ):
            raise UnsupportedOpError("Unsupported modified model copy dispatch")
    for key, value in copy_dispatch_table.items():
        if not any(
            key is trusted_key and value is trusted_value
            for trusted_key, trusted_value in _TRUSTED_COPY_DISPATCH_TABLE_ITEMS
        ):
            raise UnsupportedOpError("Unsupported modified model copy dispatch")
    for function, state in _TRUSTED_DEEPCOPY_DISPATCH_FUNCTIONS:
        if _function_local_fingerprint(function) != state:
            raise UnsupportedOpError("Unsupported modified model copy dispatch")
    for function, state in _TRUSTED_COPY_DISPATCH_TABLE_FUNCTIONS:
        if _function_local_fingerprint(function) != state:
            raise UnsupportedOpError("Unsupported modified model copy dispatch")
    for name, value, state in _TRUSTED_COPY_DEEPCOPY_GLOBALS:
        current = copy_state.get(name)
        if current is not value:
            raise UnsupportedOpError("Unsupported modified model copy helper")
        if state is not None and _function_local_fingerprint(current) != state:
            raise UnsupportedOpError("Unsupported modified model copy helper")
    for owner, name, value, state, globals_state in _TRUSTED_TENSOR_COPY_BINDINGS:
        try:
            current = _raw_static_attribute(owner, name)
        except AttributeError as exc:
            raise UnsupportedOpError(
                "Unsupported modified tensor copy binding"
            ) from exc
        if (
            current is not value
            or _function_local_fingerprint(current) != state
            or _function_referenced_globals_fingerprint(current) != globals_state
        ):
            raise UnsupportedOpError("Unsupported modified tensor copy binding")

    math_state = _raw_python_module_namespace(math)
    if (
        math_state.get("isfinite") is not _TRUSTED_MATH_ISFINITE
        or math_state.get("prod") is not _TRUSTED_MATH_PROD
    ):
        raise UnsupportedOpError("Unsupported modified math runtime bindings")

    dis_state = _raw_python_module_namespace(dis)
    current_hasjabs = dis_state.get("hasjabs")
    current_hasjrel = dis_state.get("hasjrel")
    if dis_state.get("Bytecode") is not _TRUSTED_DIS_BYTECODE:
        raise UnsupportedOpError("Unsupported modified dis.Bytecode binding")
    for name, value, state in _TRUSTED_DIS_RUNTIME_STATE:
        current = dis_state.get(name)
        if current is not value or _shallow_state_fingerprint(current) != state:
            raise UnsupportedOpError(f"Unsupported modified dis state {name}")
    for cls, state in _TRUSTED_DIS_CLASS_STATES:
        if _class_local_fingerprint(cls) != state:
            raise UnsupportedOpError("Unsupported modified dis runtime classes")
    for constructor, globals_state, tuple_new in _TRUSTED_DIS_NAMEDTUPLE_GLOBALS:
        if constructor.__globals__ is not globals_state:
            raise UnsupportedOpError(
                "Unsupported modified dis named-tuple constructor"
            )
        _require_exact_string_dict(globals_state, "dis named-tuple globals")
        if globals_state.get("_tuple_new") is not tuple_new:
            raise UnsupportedOpError(
                "Unsupported modified dis named-tuple constructor"
            )
    bytecode_namespace = type.__getattribute__(_TRUSTED_DIS_BYTECODE, "__dict__")
    for name, value, state, globals_state in _TRUSTED_DIS_BYTECODE_METHODS:
        current = bytecode_namespace.get(name)
        if (
            current is not value
            or _function_local_fingerprint(current) != state
            or _function_referenced_globals_fingerprint(current) != globals_state
        ):
            raise UnsupportedOpError("Unsupported modified dis.Bytecode methods")
    for name, value, state in _TRUSTED_DIS_CALLABLE_BINDINGS:
        current = dis_state.get(name)
        if current is not value:
            raise UnsupportedOpError(f"Unsupported modified dis binding {name}")
        if state is not None and _function_local_fingerprint(current) != state:
            raise UnsupportedOpError(f"Unsupported modified dis binding {name}")
    if (
        current_hasjabs is not _TRUSTED_DIS_HASJABS
        or current_hasjrel is not _TRUSTED_DIS_HASJREL
        or _TRUSTED_LEN(current_hasjabs) != _TRUSTED_LEN(
            _TRUSTED_DIS_HASJABS_CONTENTS
        )
        or _TRUSTED_LEN(current_hasjrel) != _TRUSTED_LEN(
            _TRUSTED_DIS_HASJREL_CONTENTS
        )
    ):
        raise UnsupportedOpError("Unsupported modified dis jump tables")
    index = 0
    for value in current_hasjabs:
        if (
            type(value) is not int
            or value != _TRUSTED_DIS_HASJABS_CONTENTS[index]
        ):
            raise UnsupportedOpError("Unsupported modified dis jump tables")
        index += 1
    index = 0
    for value in current_hasjrel:
        if (
            type(value) is not int
            or value != _TRUSTED_DIS_HASJREL_CONTENTS[index]
        ):
            raise UnsupportedOpError("Unsupported modified dis jump tables")
        index += 1

    for owner, name, function, fingerprint in _TRUSTED_FUNCTION_FINGERPRINTS:
        try:
            current = _raw_static_attribute(owner, name)
        except AttributeError as exc:
            raise UnsupportedOpError(
                f"Unsupported modified framework/runtime binding {name}"
            ) from exc
        if current is not function or _definition_fingerprint(current) != fingerprint:
            raise UnsupportedOpError(
                f"Unsupported modified framework/runtime binding {name}"
            )
    _framework_module_classes(nn)
    if nn.__dict__.get("Parameter") is not _TRUSTED_PARAMETER_TYPE:
        raise UnsupportedOpError("Unsupported modified PyTorch framework Parameter")
    for cls, before in _TRUSTED_FRAMEWORK_DEFINITIONS:
        current = dict(vars(cls))
        if current.keys() != before.keys() or any(
            current.get(name) is not value
            or _definition_fingerprint(current.get(name)) != fingerprint
            for name, (value, fingerprint) in before.items()
        ):
            raise UnsupportedOpError(
                f"Unsupported modified PyTorch framework class {cls.__name__}"
            )


def _make_integrity_validator(
    implementation: object,
    parse_implementation: object,
    frontend_globals: dict[str, object],
    trusted_type: type[object],
    trusted_str_type: type[str],
    trusted_error: type[UnsupportedOpError],
    trusted_global_bindings: tuple[tuple[str, object], ...],
    protected_frontend_bindings: tuple[tuple[str, object], ...],
    protected_function_states: tuple[tuple[str, tuple[object, ...]], ...],
    protected_class_states: tuple[tuple[str, type[object], object], ...],
    component_manifest: tuple[tuple[object, ...], ...],
    bootstrap_classes: tuple[type[BaseException], ...],
    quant_reference_module: ModuleType,
    quant_reference_globals: dict[str, object],
    protected_quant_bindings: tuple[tuple[str, object], ...],
    protected_quant_function_states: tuple[
        tuple[str, tuple[object, ...]], ...
    ],
    protected_quant_class_states: tuple[tuple[str, type[object], object], ...],
    state_matcher: object,
    state_matcher_code: CodeType,
    trusted_function_type: type[FunctionType],
    trusted_dict_type: type[dict[object, object]],
    trusted_len: object,
    trusted_value_error: type[ValueError],
    empty_cell: object,
    trusted_module_type: type[ModuleType],
    trusted_module_getattribute: object,
    trusted_type_getattribute: object,
) -> object:
    # This wrapper is the bootstrap trust boundary. Its references live in
    # closure cells, so replacing their module-global mirrors cannot affect
    # the code that detects the replacement.
    protected_class_integrity_error = trusted_error(
        "Unsupported modified frontend compiler class"
    )
    runtime_keys_error = trusted_error("Unsupported modified frontend runtime keys")
    checked_global_bindings = tuple(
        (
            name,
            value,
            trusted_error(f"Unsupported modified frontend trust root {name}"),
        )
        for name, value in trusted_global_bindings
    )
    error_binding_error = trusted_error("Unsupported modified frontend error binding")
    parser_implementation_error = trusted_error(
        "Unsupported modified frontend parser implementation"
    )
    integrity_implementation_error = trusted_error(
        "Unsupported modified frontend integrity implementation"
    )
    integrity_validator_error = trusted_error(
        "Unsupported modified frontend integrity validator"
    )
    validator_root_error = trusted_error(
        "Unsupported modified frontend validator trust root"
    )
    state_matcher_error = trusted_error("Unsupported modified frontend state matcher")
    checked_frontend_bindings = tuple(
        (
            name,
            value,
            trusted_error(f"Unsupported modified frontend parser dependency {name}"),
        )
        for name, value in protected_frontend_bindings
    )
    checked_function_states = tuple(
        (
            name,
            state,
            trusted_error(f"Unsupported modified frontend parser function {name}"),
        )
        for name, state in protected_function_states
    )
    checked_components = tuple(
        (
            label,
            module,
            source_namespace,
            consumer_namespace,
            bindings,
            function_states,
            trusted_error(f"Unsupported modified frontend component {label}"),
        )
        for (
            label,
            module,
            source_namespace,
            consumer_namespace,
            bindings,
            function_states,
        ) in component_manifest
    )
    semantic_module_error = trusted_error(
        "Unsupported modified semantic reference module"
    )
    semantic_keys_error = trusted_error("Unsupported modified semantic reference keys")
    checked_quant_bindings = tuple(
        (
            name,
            value,
            trusted_error(f"Unsupported modified semantic reference dependency {name}"),
        )
        for name, value in protected_quant_bindings
    )
    checked_quant_function_states = tuple(
        (
            name,
            state,
            trusted_error(f"Unsupported modified semantic reference function {name}"),
        )
        for name, state in protected_quant_function_states
    )
    bootstrap_class_states = tuple(
        (
            cls,
            tuple(trusted_type_getattribute(cls, "__dict__").items()),
        )
        for cls in bootstrap_classes
    )

    def validate(nn: object) -> None:
        # Error construction is part of the trust boundary.  Seal the two
        # compiler-owned exception classes before any rejection path can
        # instantiate a potentially modified class.
        for cls, expected_items in bootstrap_class_states:
            namespace = trusted_type_getattribute(cls, "__dict__")
            if trusted_len(namespace) != trusted_len(expected_items):
                raise protected_class_integrity_error
            for name, expected in expected_items:
                if namespace.get(name, empty_cell) is not expected:
                    raise protected_class_integrity_error
        for key in frontend_globals:
            if trusted_type(key) is not trusted_str_type:
                raise runtime_keys_error
        for name, value, failure in checked_global_bindings:
            if frontend_globals.get(name) is not value:
                raise failure
        if frontend_globals.get("UnsupportedOpError") is not trusted_error:
            raise error_binding_error
        if (
            frontend_globals.get("_parse_model_impl")
            is not parse_implementation
        ):
            raise parser_implementation_error
        if (
            frontend_globals.get("_validate_framework_integrity_impl")
            is not implementation
        ):
            raise integrity_implementation_error
        if frontend_globals.get("_validate_framework_integrity") is not validate:
            raise integrity_validator_error
        if (
            frontend_globals.get("_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY")
            is not validate
        ):
            raise validator_root_error
        if state_matcher.__code__ is not state_matcher_code:
            raise state_matcher_error
        for name, value, failure in checked_frontend_bindings:
            if frontend_globals.get(name, empty_cell) is not value:
                raise failure
        for name, state, failure in checked_function_states:
            function = frontend_globals.get(name, empty_cell)
            if not state_matcher(
                function,
                state,
                trusted_type,
                trusted_function_type,
                trusted_dict_type,
                trusted_str_type,
                trusted_len,
                trusted_value_error,
                empty_cell,
            ):
                raise failure
        for (
            _label,
            module,
            source_namespace,
            consumer_namespace,
            bindings,
            function_states,
            failure,
        ) in checked_components:
            if (
                trusted_type(module) is not trusted_module_type
                or trusted_module_getattribute(module, "__dict__")
                is not source_namespace
            ):
                raise failure
            for key in source_namespace:
                if trusted_type(key) is not trusted_str_type:
                    raise failure
            for key in consumer_namespace:
                if trusted_type(key) is not trusted_str_type:
                    raise failure
            for source_name, consumer_name, value in bindings:
                if (
                    source_namespace.get(source_name, empty_cell) is not value
                    or consumer_namespace.get(consumer_name, empty_cell) is not value
                ):
                    raise failure
            for source_name, consumer_name, state in function_states:
                source_function = source_namespace.get(source_name, empty_cell)
                consumer_function = consumer_namespace.get(
                    consumer_name,
                    empty_cell,
                )
                if source_function is not consumer_function or not state_matcher(
                    source_function,
                    state,
                    trusted_type,
                    trusted_function_type,
                    trusted_dict_type,
                    trusted_str_type,
                    trusted_len,
                    trusted_value_error,
                    empty_cell,
                ):
                    raise failure
        # Establish the Python/PyTorch/NumPy bootstrap before the richer class
        # fingerprints below.  Those fingerprints intentionally inspect class
        # namespaces and must never resolve a replaced builtin first.
        implementation(nn)
        for name, cls, state in protected_class_states:
            if _class_local_fingerprint(cls) != state:
                raise protected_class_integrity_error
        if (
            trusted_type(quant_reference_module) is not trusted_module_type
            or trusted_module_getattribute(quant_reference_module, "__dict__")
            is not quant_reference_globals
        ):
            raise semantic_module_error
        for key in quant_reference_globals:
            if trusted_type(key) is not trusted_str_type:
                raise semantic_keys_error
        for name, value, failure in checked_quant_bindings:
            if quant_reference_globals.get(name, empty_cell) is not value:
                raise failure
        for name, state, failure in checked_quant_function_states:
            function = quant_reference_globals.get(name, empty_cell)
            if not state_matcher(
                function,
                state,
                trusted_type,
                trusted_function_type,
                trusted_dict_type,
                trusted_str_type,
                trusted_len,
                trusted_value_error,
                empty_cell,
            ):
                raise failure
        for name, cls, state in protected_quant_class_states:
            if _class_local_fingerprint(cls) != state:
                raise protected_class_integrity_error

    return validate


def _make_parse_model(
    implementation: object,
    validator: object,
    torch: object,
    nn: object,
    frontend_globals: dict[str, object],
    trusted_type: type[object],
    trusted_str_type: type[str],
    validator_state: tuple[object, ...],
    state_matcher: object,
    state_matcher_code: CodeType,
    trusted_function_type: type[FunctionType],
    trusted_dict_type: type[dict[object, object]],
    trusted_len: object,
    trusted_value_error: type[ValueError],
    empty_cell: object,
    security_context: tuple[object, ...],
    checked_call: object,
    trusted_integrity_marker: type[_PostCallbackIntegrityError],
    integrity_errors: tuple[UnsupportedOpError, ...],
) -> object:
    def parse_model(
        model: object,
        input_shape: Sequence[int],
        input_dtype: object | None = None,
    ) -> GraphIR:
        trusted_capsule = "__torch2rtl_parse_model_trusted_capsule_v1__"
        trusted_getitem = "__torch2rtl_parse_model_trusted_getitem_v1__"
        implementation = trusted_getitem(trusted_capsule, 0)
        validator = trusted_getitem(trusted_capsule, 1)
        torch = trusted_getitem(trusted_capsule, 2)
        nn = trusted_getitem(trusted_capsule, 3)
        frontend_globals = trusted_getitem(trusted_capsule, 4)
        trusted_type = trusted_getitem(trusted_capsule, 5)
        trusted_str_type = trusted_getitem(trusted_capsule, 6)
        validator_state = trusted_getitem(trusted_capsule, 7)
        state_matcher = trusted_getitem(trusted_capsule, 8)
        state_matcher_code = trusted_getitem(trusted_capsule, 9)
        trusted_function_type = trusted_getitem(trusted_capsule, 10)
        trusted_dict_type = trusted_getitem(trusted_capsule, 11)
        trusted_len = trusted_getitem(trusted_capsule, 12)
        trusted_value_error = trusted_getitem(trusted_capsule, 13)
        empty_cell = trusted_getitem(trusted_capsule, 14)
        security_context = trusted_getitem(trusted_capsule, 15)
        checked_call = trusted_getitem(trusted_capsule, 16)
        trusted_integrity_marker = trusted_getitem(trusted_capsule, 17)
        safe_errors = trusted_getitem(trusted_capsule, 18)
        public_function = trusted_getitem(trusted_capsule, 19)
        for key in frontend_globals:
            if trusted_type(key) is not trusted_str_type:
                raise safe_errors[2]
        if frontend_globals.get("parse_model") is not public_function:
            raise safe_errors[3]
        if frontend_globals.get("_parse_model_impl") is not implementation:
            raise safe_errors[4]
        if frontend_globals.get("_validate_framework_integrity") is not validator:
            raise safe_errors[5]
        if state_matcher.__code__ is not state_matcher_code or not state_matcher(
            validator,
            validator_state,
            trusted_type,
            trusted_function_type,
            trusted_dict_type,
            trusted_str_type,
            trusted_len,
            trusted_value_error,
            empty_cell,
        ):
            raise safe_errors[5]
        validator(nn)
        if (
            frontend_globals.get("_TRUSTED_CALLBACK_SECURITY_CONTEXT")
            is not security_context
        ):
            raise safe_errors[6]
        try:
            return implementation(
                model,
                input_shape,
                input_dtype,
                torch,
                nn,
                _security_context=security_context,
                _checked_call=checked_call,
            )
        except trusted_integrity_marker:
            raise safe_errors[7] from None

    # The outer wrapper is the explicit trust anchor in the documented threat
    # model.  Embedding its dependency capsule in the immutable code constants
    # avoids writable closure/default/global roots while preserving a normal
    # Python function for signature, pickle, and public identity contracts.
    class TrustedCapsule(tuple):
        __slots__ = ()

    capsule = TrustedCapsule(
        (
            implementation,
            validator,
            torch,
            nn,
            frontend_globals,
            trusted_type,
            trusted_str_type,
            validator_state,
            state_matcher,
            state_matcher_code,
            trusted_function_type,
            trusted_dict_type,
            trusted_len,
            trusted_value_error,
            empty_cell,
            security_context,
            checked_call,
            trusted_integrity_marker,
            integrity_errors,
            parse_model,
        )
    )
    sentinel = "__torch2rtl_parse_model_trusted_capsule_v1__"
    getitem_sentinel = "__torch2rtl_parse_model_trusted_getitem_v1__"
    constants = parse_model.__code__.co_consts
    if (
        sum(value == sentinel for value in constants) != 1
        or sum(value == getitem_sentinel for value in constants) != 1
    ):
        raise RuntimeError("Could not construct frontend parser trust capsule")
    parse_model.__code__ = parse_model.__code__.replace(
        co_consts=tuple(
            capsule
            if value == sentinel
            else tuple.__getitem__
            if value == getitem_sentinel
            else value
            for value in constants
        )
    )
    return parse_model


def _load_python_module(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


# This is an authored manifest, not a discovered registry.  A new compiler
# helper must be added deliberately so the integrity tests fail closed when
# the parser call graph grows.
_PROTECTED_FRONTEND_FUNCTION_NAMES = (
    "_capture_function_integrity_state",
    "_function_integrity_state_matches",
    "_require_exact_string_dict",
    "_definition_fingerprint",
    "_shallow_state_fingerprint",
    "_function_local_fingerprint",
    "_descriptor_local_fingerprint",
    "_class_local_fingerprint",
    "_function_referenced_globals_fingerprint",
    "_checked_untrusted_call",
    "_raw_static_attribute",
    "_raw_python_module_namespace",
    "_framework_module_classes",
    "_validate_framework_integrity_impl",
    "_make_integrity_validator",
    "_make_parse_model",
)
_PROTECTED_FRONTEND_FUNCTION_STATES = tuple(
    (name, _capture_function_integrity_state(_TRUSTED_FRONTEND_GLOBALS[name]))
    for name in _PROTECTED_FRONTEND_FUNCTION_NAMES
)
_PROTECTED_FRONTEND_BINDINGS = (
    ("CodeType", CodeType),
    ("FunctionType", FunctionType),
    ("ModuleType", ModuleType),
    ("GraphIR", GraphIR),
    ("TensorIR", TensorIR),
    ("ArgmaxIR", ArgmaxIR),
    ("Conv2dIR", Conv2dIR),
    ("FlattenIR", FlattenIR),
    ("LinearIR", LinearIR),
    ("ReluIR", ReluIR),
    ("UnsupportedOpError", UnsupportedOpError),
    ("_errors_component", _errors_component),
    ("_fusion_component", _fusion_component),
    ("_lowering_component", _lowering_component),
    ("_model_contract_component", _model_contract_component),
    ("_pipeline_component", _pipeline_component),
    ("_semantics_component", _semantics_component),
    ("_state_guard_component", _state_guard_component),
    ("_quant_reference", _quant_reference),
)

_LOWERING_FUNCTION_BINDINGS = (
    ("_require_exact_string_dict", "_require_exact_string_dict"),
    ("_raw_static_attribute", "_raw_static_attribute"),
    ("lower_fx_graph", "lower_fx_graph"),
    ("_parse_module_node", "_parse_module_node"),
    ("_lower_linear_module", "_lower_linear_module"),
    ("_lower_conv2d_module", "_lower_conv2d_module"),
    ("_lower_relu_module", "_lower_relu_module"),
    ("_lower_flatten_module", "_lower_flatten_module"),
    ("_flatten_shape", "_flatten_shape"),
    ("_validate_input_shape", "_validate_input_shape"),
    ("_validate_tensor_element_count", "_validate_tensor_element_count"),
    ("_parse_output_node", "_parse_output_node"),
    ("_require_sequential_input", "_require_sequential_input"),
    ("_validate_argmax_arguments", "_validate_argmax_arguments"),
    ("_lower_argmax_node", "_lower_argmax_node"),
    ("_positional_or_keyword", "_positional_or_keyword"),
    ("_is_fx_node", "_is_fx_node"),
    ("_conv_output_dim", "_conv_output_dim"),
    ("_int_pair", "_int_pair"),
    ("_positive_module_integer", "_positive_module_integer"),
    ("_module_integer", "_module_integer"),
    ("_required_parameter", "_required_parameter"),
    ("_required_module_attribute", "_required_module_attribute"),
    ("_optional_parameter", "_optional_parameter"),
    ("_validate_conv_pair", "_validate_conv_pair"),
    ("_safe_name", "_safe_name"),
)
_PROTECTED_LOWERING_FUNCTION_STATES = tuple(
    (
        source_name,
        consumer_name,
        _capture_function_integrity_state(
            _TRUSTED_LOWERING_GLOBALS[source_name]
        ),
    )
    for source_name, consumer_name in _LOWERING_FUNCTION_BINDINGS
)
_PROTECTED_LOWERING_BINDINGS = (
    (
        "_MAX_TENSOR_ELEMENTS",
        "_MAX_TENSOR_ELEMENTS",
        _lowering_component._MAX_TENSOR_ELEMENTS,
    ),
    ("_MAX_TENSOR_RANK", "_MAX_TENSOR_RANK", _lowering_component._MAX_TENSOR_RANK),
    ("_SV_INT_MAX", "_SV_INT_MAX", _lowering_component._SV_INT_MAX),
    ("math", "math", math),
    (
        "SequenceABC",
        "SequenceABC",
        _lowering_component.SequenceABC,
    ),
    ("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),
    ("GraphIR", "GraphIR", GraphIR),
    ("TensorIR", "TensorIR", TensorIR),
    ("ArgmaxIR", "ArgmaxIR", ArgmaxIR),
    ("Conv2dIR", "Conv2dIR", Conv2dIR),
    ("FlattenIR", "FlattenIR", FlattenIR),
    ("LinearIR", "LinearIR", LinearIR),
    ("ReluIR", "ReluIR", ReluIR),
)
_STATE_GUARD_FUNCTION_BINDINGS = (
    ("_state_framework_module_classes", "_state_framework_module_classes"),
    ("_copy_model_for_tracing", "_copy_model_for_tracing"),
    ("_validate_no_custom_copy_protocol", "_validate_no_custom_copy_protocol"),
    ("_validate_no_custom_class_state", "_validate_no_custom_class_state"),
    ("_validate_copy_safe_instance_state", "_validate_copy_safe_instance_state"),
    ("_is_copy_safe_state_value", "_is_copy_safe_state_value"),
    ("_snapshot_module_class_definitions", "_snapshot_module_class_definitions"),
    ("_restore_module_class_definitions", "_restore_module_class_definitions"),
    ("_raw_module_state", "_raw_module_state"),
    ("_trusted_named_modules", "_trusted_named_modules"),
    ("_trusted_named_tensors", "_trusted_named_tensors"),
    ("_snapshot_module_state", "_snapshot_module_state"),
    ("_freeze_state_value", "_freeze_state_value"),
)
_PROTECTED_STATE_GUARD_FUNCTION_STATES = tuple(
    (
        source_name,
        consumer_name,
        _capture_function_integrity_state(
            _TRUSTED_STATE_GUARD_GLOBALS[source_name]
        ),
    )
    for source_name, consumer_name in _STATE_GUARD_FUNCTION_BINDINGS
)
_PROTECTED_STATE_GUARD_BINDINGS = (
    ("copy", "copy", copy),
    (
        "OrderedDict",
        "OrderedDict",
        _state_guard_component.OrderedDict,
    ),
    ("FunctionType", "FunctionType", FunctionType),
    ("np", "np", np),
    ("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),
)
_FUSION_FUNCTION_BINDINGS = (
    ("_fusion_require_exact_string_dict", "_fusion_require_exact_string_dict"),
    ("_fusion_raw_static_attribute", "_fusion_raw_static_attribute"),
    ("_fusion_is_fx_node", "_fusion_is_fx_node"),
    ("_fusion_required_module_attribute", "_fusion_required_module_attribute"),
    ("_fusion_positive_module_integer", "_fusion_positive_module_integer"),
    ("_fusion_required_parameter", "_fusion_required_parameter"),
    ("_fusion_optional_parameter", "_fusion_optional_parameter"),
    ("_fuse_conv_batchnorm_eval", "_fuse_conv_batchnorm_eval"),
    ("_validate_conv_batchnorm_pair", "_validate_conv_batchnorm_pair"),
    ("_require_eval_module", "_require_eval_module"),
    ("_validate_fused_conv", "_validate_fused_conv"),
    ("_tensor_values_are_finite", "_tensor_values_are_finite"),
    ("_fresh_fused_target", "_fresh_fused_target"),
    ("_fx_node_use_count", "_fx_node_use_count"),
    ("_fx_value_reference_count", "_fx_value_reference_count"),
)
_PROTECTED_FUSION_FUNCTION_STATES = tuple(
    (
        source_name,
        consumer_name,
        _capture_function_integrity_state(
            _TRUSTED_FUSION_GLOBALS[source_name]
        ),
    )
    for source_name, consumer_name in _FUSION_FUNCTION_BINDINGS
)
_PROTECTED_FUSION_BINDINGS = (
    ("math", "math", math),
    ("np", "np", np),
    ("FunctionType", "FunctionType", FunctionType),
    ("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),
    (
        "_raw_module_state",
        "_raw_module_state",
        _state_guard_component._raw_module_state,
    ),
    (
        "_trusted_named_modules",
        "_trusted_named_modules",
        _state_guard_component._trusted_named_modules,
    ),
)
_SEMANTICS_FUNCTION_BINDINGS = (
    ("_run_concrete_probe", "_run_concrete_probe"),
    ("_validate_fusion_boundary", "_validate_fusion_boundary"),
    ("_validate_lowered_semantics", "_validate_lowered_semantics"),
)
_PROTECTED_SEMANTICS_FUNCTION_STATES = tuple(
    (
        source_name,
        consumer_name,
        _capture_function_integrity_state(
            _TRUSTED_SEMANTICS_GLOBALS[source_name]
        ),
    )
    for source_name, consumer_name in _SEMANTICS_FUNCTION_BINDINGS
)
_PROTECTED_SEMANTICS_BINDINGS = (
    ("math", "math", math),
    ("np", "np", np),
    ("_quant_reference", "_quant_reference", _quant_reference),
    (
        "_INFER_FLOAT_GRAPH",
        "_INFER_FLOAT_GRAPH",
        _TRUSTED_INFER_FLOAT_GRAPH,
    ),
    ("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),
    ("GraphIR", "GraphIR", GraphIR),
    (
        "_copy_model_for_tracing",
        "_copy_model_for_tracing",
        _state_guard_component._copy_model_for_tracing,
    ),
    (
        "_restore_module_class_definitions",
        "_restore_module_class_definitions",
        _state_guard_component._restore_module_class_definitions,
    ),
    (
        "_snapshot_module_class_definitions",
        "_snapshot_module_class_definitions",
        _state_guard_component._snapshot_module_class_definitions,
    ),
    (
        "_snapshot_module_state",
        "_snapshot_module_state",
        _state_guard_component._snapshot_module_state,
    ),
)
_MODEL_CONTRACT_FUNCTION_BINDINGS = (
    ("_contract_require_exact_string_dict", "_contract_require_exact_string_dict"),
    ("_contract_raw_static_attribute", "_contract_raw_static_attribute"),
    ("_validate_model_semantics", "_validate_model_semantics"),
    ("_validate_module_dispatch", "_validate_module_dispatch"),
    ("_validate_no_custom_introspection", "_validate_no_custom_introspection"),
    ("_validate_forward_python_state", "_validate_forward_python_state"),
    (
        "_validate_reachable_python_function",
        "_validate_reachable_python_function",
    ),
    ("_find_custom_method", "_find_custom_method"),
    ("_next_loaded_attribute", "_next_loaded_attribute"),
    ("_is_immutable_python_state", "_is_immutable_python_state"),
    ("_contains_code_object", "_contains_code_object"),
    ("_normalize_float_dtype", "_normalize_float_dtype"),
)
_PROTECTED_MODEL_CONTRACT_FUNCTION_STATES = tuple(
    (
        source_name,
        consumer_name,
        _capture_function_integrity_state(
            _TRUSTED_MODEL_CONTRACT_GLOBALS[source_name]
        ),
    )
    for source_name, consumer_name in _MODEL_CONTRACT_FUNCTION_BINDINGS
)
_PROTECTED_MODEL_CONTRACT_BINDINGS = (
    ("builtins", "builtins", builtins),
    ("dis", "dis", dis),
    ("np", "np", np),
    ("CodeType", "CodeType", CodeType),
    ("FunctionType", "FunctionType", FunctionType),
    ("_BUILTINS", "_BUILTINS", _TRUSTED_BUILTINS),
    ("_LEN", "_LEN", _TRUSTED_LEN),
    ("_DIS_BYTECODE", "_DIS_BYTECODE", _TRUSTED_DIS_BYTECODE),
    ("_DIS_HASJABS", "_DIS_HASJABS", _TRUSTED_DIS_HASJABS),
    ("_DIS_HASJREL", "_DIS_HASJREL", _TRUSTED_DIS_HASJREL),
    ("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),
    (
        "_raw_module_state",
        "_raw_module_state",
        _state_guard_component._raw_module_state,
    ),
    (
        "_state_framework_module_classes",
        "_state_framework_module_classes",
        _state_guard_component._state_framework_module_classes,
    ),
    (
        "_trusted_named_modules",
        "_trusted_named_modules",
        _state_guard_component._trusted_named_modules,
    ),
    (
        "_trusted_named_tensors",
        "_trusted_named_tensors",
        _state_guard_component._trusted_named_tensors,
    ),
    (
        "_validate_copy_safe_instance_state",
        "_validate_copy_safe_instance_state",
        _state_guard_component._validate_copy_safe_instance_state,
    ),
    (
        "_validate_no_custom_class_state",
        "_validate_no_custom_class_state",
        _state_guard_component._validate_no_custom_class_state,
    ),
    (
        "_validate_no_custom_copy_protocol",
        "_validate_no_custom_copy_protocol",
        _state_guard_component._validate_no_custom_copy_protocol,
    ),
)
_PIPELINE_FUNCTION_BINDINGS = (("_parse_model_impl", "_parse_model_impl"),)
_PROTECTED_PIPELINE_FUNCTION_STATES = tuple(
    (
        source_name,
        consumer_name,
        _capture_function_integrity_state(
            _TRUSTED_PIPELINE_GLOBALS[source_name]
        ),
    )
    for source_name, consumer_name in _PIPELINE_FUNCTION_BINDINGS
)
_PROTECTED_PIPELINE_BINDINGS = (
    ("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),
    ("GraphIR", "GraphIR", GraphIR),
    (
        "_validate_input_shape",
        "_validate_input_shape",
        _lowering_component._validate_input_shape,
    ),
    (
        "_validate_model_semantics",
        "_validate_model_semantics",
        _model_contract_component._validate_model_semantics,
    ),
    (
        "_copy_model_for_tracing",
        "_copy_model_for_tracing",
        _state_guard_component._copy_model_for_tracing,
    ),
    (
        "_snapshot_module_state",
        "_snapshot_module_state",
        _state_guard_component._snapshot_module_state,
    ),
    (
        "_snapshot_module_class_definitions",
        "_snapshot_module_class_definitions",
        _state_guard_component._snapshot_module_class_definitions,
    ),
    (
        "_restore_module_class_definitions",
        "_restore_module_class_definitions",
        _state_guard_component._restore_module_class_definitions,
    ),
    (
        "_fuse_conv_batchnorm_eval",
        "_fuse_conv_batchnorm_eval",
        _fusion_component._fuse_conv_batchnorm_eval,
    ),
    (
        "_trusted_named_modules",
        "_trusted_named_modules",
        _state_guard_component._trusted_named_modules,
    ),
    ("lower_fx_graph", "lower_fx_graph", _lowering_component.lower_fx_graph),
    (
        "_validate_fusion_boundary",
        "_validate_fusion_boundary",
        _semantics_component._validate_fusion_boundary,
    ),
    (
        "_validate_lowered_semantics",
        "_validate_lowered_semantics",
        _semantics_component._validate_lowered_semantics,
    ),
)
_PROTECTED_PIPELINE_FACADE_FUNCTION_STATES = (
    (
        "_parse_model_impl",
        "_parse_model_impl",
        _capture_function_integrity_state(
            _TRUSTED_PIPELINE_GLOBALS["_parse_model_impl"]
        ),
    ),
)
_PROTECTED_IR_CLASS_STATES = tuple(
    (cls.__name__, cls, _class_local_fingerprint(cls))
    for cls in (
        GraphIR,
        TensorIR,
        ArgmaxIR,
        Conv2dIR,
        FlattenIR,
        LinearIR,
        ReluIR,
        UnsupportedOpError,
        _PostCallbackIntegrityError,
    )
)

# Authored component descriptors. Definition-site dependencies are checked in
# their real namespaces; only actual facade consumers receive facade entries.
_COMPONENT_ATTESTATION_MANIFEST = (
    (
        "frontend_core",
        _TRUSTED_FRONTEND_MODULE,
        _TRUSTED_FRONTEND_GLOBALS,
        _TRUSTED_FRONTEND_GLOBALS,
        tuple(
            (name, name, value)
            for name, value in _PROTECTED_FRONTEND_BINDINGS
        ),
        tuple(
            (name, name, state)
            for name, state in _PROTECTED_FRONTEND_FUNCTION_STATES
        ),
    ),
    (
        "errors",
        _TRUSTED_ERRORS_MODULE,
        _TRUSTED_ERRORS_GLOBALS,
        _TRUSTED_FRONTEND_GLOBALS,
        (("UnsupportedOpError", "UnsupportedOpError", UnsupportedOpError),),
        (),
    ),
    (
        "lowering",
        _TRUSTED_LOWERING_MODULE,
        _TRUSTED_LOWERING_GLOBALS,
        _TRUSTED_LOWERING_GLOBALS,
        _PROTECTED_LOWERING_BINDINGS,
        _PROTECTED_LOWERING_FUNCTION_STATES,
    ),
    (
        "state_guard",
        _TRUSTED_STATE_GUARD_MODULE,
        _TRUSTED_STATE_GUARD_GLOBALS,
        _TRUSTED_STATE_GUARD_GLOBALS,
        _PROTECTED_STATE_GUARD_BINDINGS,
        _PROTECTED_STATE_GUARD_FUNCTION_STATES,
    ),
    (
        "fusion",
        _TRUSTED_FUSION_MODULE,
        _TRUSTED_FUSION_GLOBALS,
        _TRUSTED_FUSION_GLOBALS,
        _PROTECTED_FUSION_BINDINGS,
        _PROTECTED_FUSION_FUNCTION_STATES,
    ),
    (
        "semantics",
        _TRUSTED_SEMANTICS_MODULE,
        _TRUSTED_SEMANTICS_GLOBALS,
        _TRUSTED_SEMANTICS_GLOBALS,
        _PROTECTED_SEMANTICS_BINDINGS,
        _PROTECTED_SEMANTICS_FUNCTION_STATES,
    ),
    (
        "model_contract",
        _TRUSTED_MODEL_CONTRACT_MODULE,
        _TRUSTED_MODEL_CONTRACT_GLOBALS,
        _TRUSTED_MODEL_CONTRACT_GLOBALS,
        _PROTECTED_MODEL_CONTRACT_BINDINGS,
        _PROTECTED_MODEL_CONTRACT_FUNCTION_STATES,
    ),
    (
        "pipeline",
        _TRUSTED_PIPELINE_MODULE,
        _TRUSTED_PIPELINE_GLOBALS,
        _TRUSTED_PIPELINE_GLOBALS,
        _PROTECTED_PIPELINE_BINDINGS,
        _PROTECTED_PIPELINE_FUNCTION_STATES,
    ),
    (
        "pipeline_facade",
        _TRUSTED_PIPELINE_MODULE,
        _TRUSTED_PIPELINE_GLOBALS,
        _TRUSTED_FRONTEND_GLOBALS,
        (),
        _PROTECTED_PIPELINE_FACADE_FUNCTION_STATES,
    ),
)

_PROTECTED_QUANT_BINDINGS = (
    ("infer_float_graph", _TRUSTED_INFER_FLOAT_GRAPH),
    ("_conv2d_float", _quant_reference._conv2d_float),
    ("_graph_float_dtype", _quant_reference._graph_float_dtype),
    ("_FUNCTIONAL_LINEAR", _quant_reference._FUNCTIONAL_LINEAR),
    ("_FUNCTIONAL_CONV2D", _quant_reference._FUNCTIONAL_CONV2D),
    ("_FUNCTIONAL_RELU", _quant_reference._FUNCTIONAL_RELU),
    ("GraphIR", _quant_reference.GraphIR),
    ("LinearIR", _quant_reference.LinearIR),
    ("Conv2dIR", _quant_reference.Conv2dIR),
    ("ReluIR", _quant_reference.ReluIR),
    ("FlattenIR", _quant_reference.FlattenIR),
    ("ArgmaxIR", _quant_reference.ArgmaxIR),
    ("FloatReferenceResult", _quant_reference.FloatReferenceResult),
    ("ActivationResult", _quant_reference.ActivationResult),
    ("np", _quant_reference.np),
)
_PROTECTED_QUANT_FUNCTION_STATES = tuple(
    (name, _capture_function_integrity_state(_TRUSTED_QUANT_REFERENCE_GLOBALS[name]))
    for name in ("infer_float_graph", "_conv2d_float", "_graph_float_dtype")
)
_PROTECTED_QUANT_CLASS_STATES = tuple(
    (cls.__name__, cls, _class_local_fingerprint(cls))
    for cls in (
        _quant_reference.FloatReferenceResult,
        _quant_reference.ActivationResult,
    )
)

_EXPLICIT_TRUSTED_GLOBAL_NAMES = (
    "_COMPONENT_ATTESTATION_MANIFEST",
    "_TRUSTED_BOOTSTRAP_GLOBAL_BINDINGS",
    "_TRUSTED_BUILTINS",
    "_TRUSTED_BUILTINS_MODULE",
    "_TRUSTED_BUILTIN_BINDINGS",
    "_TRUSTED_CODE_TYPE",
    "_TRUSTED_COPY_DEEPCOPY_GLOBALS",
    "_TRUSTED_COPY_DISPATCH_TABLE",
    "_TRUSTED_COPY_DISPATCH_TABLE_FUNCTIONS",
    "_TRUSTED_COPY_DISPATCH_TABLE_ITEMS",
    "_TRUSTED_COPY_MODULE",
    "_TRUSTED_DEEPCOPY",
    "_TRUSTED_DEEPCOPY_CODE",
    "_TRUSTED_DEEPCOPY_DISPATCH",
    "_TRUSTED_DEEPCOPY_DISPATCH_FUNCTIONS",
    "_TRUSTED_DEEPCOPY_DISPATCH_ITEMS",
    "_TRUSTED_DEEPCOPY_LOCAL_STATE",
    "_TRUSTED_DICT_TYPE",
    "_TRUSTED_DIS_BYTECODE",
    "_TRUSTED_DIS_BYTECODE_METHODS",
    "_TRUSTED_DIS_CALLABLE_BINDINGS",
    "_TRUSTED_DIS_CLASS_STATES",
    "_TRUSTED_DIS_HASJABS",
    "_TRUSTED_DIS_HASJABS_CONTENTS",
    "_TRUSTED_DIS_HASJREL",
    "_TRUSTED_DIS_HASJREL_CONTENTS",
    "_TRUSTED_DIS_MODULE",
    "_TRUSTED_DIS_NAMEDTUPLE_GLOBALS",
    "_TRUSTED_DIS_RUNTIME_STATE",
    "_TRUSTED_EMPTY_CELL",
    "_TRUSTED_ERRORS_GLOBALS",
    "_TRUSTED_ERRORS_MODULE",
    "_TRUSTED_FRAMEWORK_CLASSES",
    "_TRUSTED_FRAMEWORK_DEFINITIONS",
    "_TRUSTED_FRAMEWORK_TYPES",
    "_TRUSTED_FRONTEND_GLOBALS",
    "_TRUSTED_FRONTEND_MODULE",
    "_TRUSTED_FUSION_GLOBALS",
    "_TRUSTED_FUSION_MODULE",
    "_TRUSTED_FUNCTION_BINDINGS",
    "_TRUSTED_FUNCTION_FINGERPRINTS",
    "_TRUSTED_FUNCTION_TYPE",
    "_TRUSTED_FX_CLASSES",
    "_TRUSTED_INFER_FLOAT_GRAPH",
    "_TRUSTED_INSPECT_GETATTR_STATIC",
    "_TRUSTED_INSPECT_MODULE",
    "_TRUSTED_INSPECT_UNWRAP",
    "_TRUSTED_INTEGRITY_HELPER_BINDINGS",
    "_TRUSTED_IS_GRAD_ENABLED",
    "_TRUSTED_LEN",
    "_TRUSTED_LOWERING_GLOBALS",
    "_TRUSTED_LOWERING_MODULE",
    "_TRUSTED_MATH_ISFINITE",
    "_TRUSTED_MATH_MODULE",
    "_TRUSTED_MATH_PROD",
    "_TRUSTED_MODULE_GETATTRIBUTE",
    "_TRUSTED_MODULE_TYPE",
    "_TRUSTED_MODEL_CONTRACT_GLOBALS",
    "_TRUSTED_MODEL_CONTRACT_MODULE",
    "_TRUSTED_NUMPY_BINDINGS",
    "_TRUSTED_NUMPY_MODULE",
    "_TRUSTED_PARAMETER_TYPE",
    "_TRUSTED_PIPELINE_GLOBALS",
    "_TRUSTED_PIPELINE_MODULE",
    "_TRUSTED_QUANT_REFERENCE_GLOBALS",
    "_TRUSTED_QUANT_REFERENCE_MODULE",
    "_TRUSTED_RUNTIME_MODULES",
    "_TRUSTED_SEMANTICS_GLOBALS",
    "_TRUSTED_SEMANTICS_MODULE",
    "_TRUSTED_SET_GRAD_ENABLED",
    "_TRUSTED_STATE_GUARD_GLOBALS",
    "_TRUSTED_STATE_GUARD_MODULE",
    "_TRUSTED_STR_TYPE",
    "_TRUSTED_SYS_BYTEORDER",
    "_TRUSTED_SYS_GETRECURSIONLIMIT",
    "_TRUSTED_SYS_MODULE",
    "_TRUSTED_SYS_MODULES",
    "_TRUSTED_TENSOR_COPY_BINDINGS",
    "_TRUSTED_TORCH_MODULE",
    "_TRUSTED_TORCH_C_MODULE",
    "_TRUSTED_TORCH_NN_MODULE",
    "_TRUSTED_TYPE",
    "_TRUSTED_TYPE_GETATTRIBUTE",
    "_TRUSTED_UNSUPPORTED_OP_ERROR",
)
_EXPLICIT_TRUSTED_GLOBAL_BINDINGS = tuple(
    (name, _TRUSTED_FRONTEND_GLOBALS[name])
    for name in _EXPLICIT_TRUSTED_GLOBAL_NAMES
)

_validate_framework_integrity = _make_integrity_validator(
    _validate_framework_integrity_impl,
    _parse_model_impl,
    _TRUSTED_FRONTEND_GLOBALS,
    _TRUSTED_TYPE,
    _TRUSTED_STR_TYPE,
    _TRUSTED_UNSUPPORTED_OP_ERROR,
    _EXPLICIT_TRUSTED_GLOBAL_BINDINGS,
    _PROTECTED_FRONTEND_BINDINGS,
    _PROTECTED_FRONTEND_FUNCTION_STATES,
    _PROTECTED_IR_CLASS_STATES,
    _COMPONENT_ATTESTATION_MANIFEST,
    (UnsupportedOpError, _PostCallbackIntegrityError),
    _TRUSTED_QUANT_REFERENCE_MODULE,
    _TRUSTED_QUANT_REFERENCE_GLOBALS,
    _PROTECTED_QUANT_BINDINGS,
    _PROTECTED_QUANT_FUNCTION_STATES,
    _PROTECTED_QUANT_CLASS_STATES,
    _function_integrity_state_matches,
    _function_integrity_state_matches.__code__,
    _TRUSTED_FUNCTION_TYPE,
    _TRUSTED_DICT_TYPE,
    _TRUSTED_LEN,
    ValueError,
    _TRUSTED_EMPTY_CELL,
    _TRUSTED_MODULE_TYPE,
    _TRUSTED_MODULE_GETATTRIBUTE,
    _TRUSTED_TYPE_GETATTRIBUTE,
)
_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY = _validate_framework_integrity

_VALIDATOR_INTEGRITY_STATE = _capture_function_integrity_state(
    _validate_framework_integrity
)
_TRUSTED_CALLBACK_SECURITY_CONTEXT = (
    _validate_framework_integrity,
    _VALIDATOR_INTEGRITY_STATE,
    _function_integrity_state_matches,
    _function_integrity_state_matches.__code__,
    _TRUSTED_TYPE,
    _TRUSTED_FUNCTION_TYPE,
    _TRUSTED_DICT_TYPE,
    _TRUSTED_STR_TYPE,
    _TRUSTED_LEN,
    ValueError,
    _TRUSTED_EMPTY_CELL,
    _TRUSTED_UNSUPPORTED_OP_ERROR,
    BaseException,
    _PostCallbackIntegrityError,
    _TRUSTED_IS_GRAD_ENABLED,
    _TRUSTED_SET_GRAD_ENABLED,
)

_PUBLIC_PARSE_ERRORS = (
    _TRUSTED_UNSUPPORTED_OP_ERROR(
        "Unsupported modified frontend parser closure roots"
    ),
    _TRUSTED_UNSUPPORTED_OP_ERROR("Unsupported modified frontend parser closure"),
    _TRUSTED_UNSUPPORTED_OP_ERROR("Unsupported modified frontend runtime keys"),
    _TRUSTED_UNSUPPORTED_OP_ERROR("Unsupported modified frontend parser binding"),
    _TRUSTED_UNSUPPORTED_OP_ERROR(
        "Unsupported modified frontend parser implementation"
    ),
    _TRUSTED_UNSUPPORTED_OP_ERROR(
        "Unsupported modified frontend integrity validator"
    ),
    _TRUSTED_UNSUPPORTED_OP_ERROR(
        "Unsupported modified frontend trust root "
        "_TRUSTED_CALLBACK_SECURITY_CONTEXT"
    ),
    _TRUSTED_UNSUPPORTED_OP_ERROR(
        "Unsupported modified frontend integrity after untrusted callback"
    ),
)

parse_model = _make_parse_model(
    _parse_model_impl,
    _validate_framework_integrity,
    _TRUSTED_TORCH_MODULE,
    _TRUSTED_TORCH_NN_MODULE,
    _TRUSTED_FRONTEND_GLOBALS,
    _TRUSTED_TYPE,
    _TRUSTED_STR_TYPE,
    _VALIDATOR_INTEGRITY_STATE,
    _function_integrity_state_matches,
    _function_integrity_state_matches.__code__,
    _TRUSTED_FUNCTION_TYPE,
    _TRUSTED_DICT_TYPE,
    _TRUSTED_LEN,
    ValueError,
    _TRUSTED_EMPTY_CELL,
    _TRUSTED_CALLBACK_SECURITY_CONTEXT,
    _checked_untrusted_call,
    _PostCallbackIntegrityError,
    _PUBLIC_PARSE_ERRORS,
)
parse_model.__qualname__ = "parse_model"
