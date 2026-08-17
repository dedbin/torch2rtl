from __future__ import annotations

import builtins
import copy
import dis
import importlib.util
import inspect
import math
import sys
from collections import OrderedDict
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Sequence

import numpy as np
import torch as _torch

from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.ir.tensor import TensorIR


_MAX_TENSOR_ELEMENTS = 1_000_000
_MAX_TENSOR_RANK = 64
_SV_INT_MAX = (1 << 31) - 1
# Bootstrap must not resolve these names through mutable builtins or a mutable
# frontend alias before the builtins integrity check has completed.
_TRUSTED_TYPE = type
_TRUSTED_DICT_TYPE = dict
_TRUSTED_STR_TYPE = str
_TRUSTED_CODE_TYPE = CodeType
_TRUSTED_FUNCTION_TYPE = FunctionType
_TRUSTED_MODULE_TYPE = ModuleType
_TRUSTED_MODULE_GETATTRIBUTE = ModuleType.__getattribute__
_TRUSTED_FRONTEND_GLOBALS = globals()


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
    (np, "array_equal", np.array_equal),
    (np, "ascontiguousarray", np.ascontiguousarray),
    (np, "dtype", np.dtype),
    (np, "float32", np.float32),
    (np, "float64", np.float64),
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
    )
)
_TRUSTED_FUNCTION_BINDINGS = (
    *_TRUSTED_NUMPY_BINDINGS,
    (_torch, "nn", _torch.nn),
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
    (_torch, "no_grad", _torch.no_grad),
    (_torch, "get_default_dtype", _torch.get_default_dtype),
    (_torch, "Tensor", _torch.Tensor),
    (_torch, "device", _torch.device),
    (_torch, "float32", _torch.float32),
    (_torch, "float64", _torch.float64),
    (_torch, "int64", _torch.int64),
    (_torch, "preserve_format", _torch.preserve_format),
    (_torch, "strided", _torch.strided),
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


class UnsupportedOpError(RuntimeError):
    """Raised when the FX graph contains an operation outside the MVP subset."""


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


def _parse_model_impl(
    model: object,
    input_shape: Sequence[int],
    input_dtype: object | None,
    torch: object,
    nn: object,
) -> GraphIR:
    normalized_input_shape = _validate_input_shape(input_shape)
    float_dtype = _validate_model_semantics(model, nn, torch, input_dtype)
    trace_model = _copy_model_for_tracing(model, nn, torch)
    instance_state = _snapshot_module_state(trace_model, nn, torch)
    class_state = _snapshot_module_class_definitions(trace_model, nn)
    trace_error: Exception | None = None
    traced: object | None = None
    try:
        traced = torch.fx.symbolic_trace(trace_model)
    except Exception as exc:
        trace_error = exc
    class_state_changed = _restore_module_class_definitions(class_state)
    instance_state_changed = instance_state != _snapshot_module_state(
        trace_model,
        nn,
        torch,
    )
    if class_state_changed or instance_state_changed:
        raise UnsupportedOpError(
            "Unsupported Python state mutation during FX symbolic trace"
        )
    if trace_error is not None:
        raise UnsupportedOpError(
            f"Unsupported FX symbolic trace semantics: {trace_error}"
        ) from trace_error
    assert traced is not None
    traced, transformations = _fuse_conv_batchnorm_eval(traced, nn, torch)
    has_batchnorm_fusion = bool(transformations)
    modules = dict(_trusted_named_modules(traced, nn))
    input_tensor = TensorIR(
        name="input",
        shape=normalized_input_shape,
        dtype=float_dtype,
    )
    current_tensor = input_tensor
    current_node: object | None = None
    ops: list[object] = []
    saw_argmax = False
    saw_output = False
    source_current_shape = (
        (1, *normalized_input_shape) if has_batchnorm_fusion else None
    )

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if current_node is not None:
                raise UnsupportedOpError(
                    "Unsupported FX graph: exactly one tensor input is required"
                )
            current_node = node
            continue
        if node.op == "output":
            output_node = _parse_output_node(node)
            if output_node is not current_node:
                raise UnsupportedOpError(
                    "Unsupported FX graph: output must be the final node of a "
                    "single sequential chain"
                )
            saw_output = True
            continue
        if saw_output:
            raise UnsupportedOpError("Unsupported FX graph: node found after output")
        if current_node is None:
            raise UnsupportedOpError(
                "Unsupported FX graph: operation appears before the input placeholder"
            )
        if saw_argmax:
            raise UnsupportedOpError(
                "Unsupported FX graph: argmax must be the final operation"
            )
        if node.op == "call_module":
            _require_sequential_input(node, current_node)
            module = modules[str(node.target)]
            if source_current_shape is not None and type(module) is nn.Flatten:
                start_dim = _module_integer(
                    _required_module_attribute(module, "Flatten", "start_dim"),
                    "Flatten",
                    "start_dim",
                )
                end_dim = _module_integer(
                    _required_module_attribute(module, "Flatten", "end_dim"),
                    "Flatten",
                    "end_dim",
                )
                try:
                    source_flatten_shape = _flatten_shape(
                        source_current_shape,
                        start_dim,
                        end_dim,
                    )
                    fused_flatten_shape = _flatten_shape(
                        current_tensor.shape,
                        start_dim,
                        end_dim,
                    )
                except UnsupportedOpError as exc:
                    raise UnsupportedOpError(
                        "Unsupported Flatten with singleton batch adapter: "
                        "dimensions must be valid for both source NCHW and "
                        "fused CHW shapes"
                    ) from exc
                if source_flatten_shape != (1, *fused_flatten_shape):
                    raise UnsupportedOpError(
                        "Unsupported Flatten with singleton batch adapter: source "
                        f"shape {source_flatten_shape} does not preserve the fused "
                        f"batchless shape {(1, *fused_flatten_shape)}"
                    )
            current_tensor, new_op = _parse_module_node(
                node_name=_safe_name(str(node.name)),
                module=module,
                current_tensor=current_tensor,
                nn=nn,
            )
            if source_current_shape is not None:
                source_current_shape = (1, *current_tensor.shape)
            ops.append(new_op)
            current_node = node
            continue
        if node.op == "call_function" and node.target is torch.argmax:
            _require_sequential_input(node, current_node, allow_parameters=True)
            _validate_argmax_arguments(node, is_method=False)
            argmax_input = current_tensor
            current_tensor = TensorIR(name=_safe_name(str(node.name)), shape=(), dtype="int64")
            ops.append(
                ArgmaxIR(
                    name=_safe_name(str(node.name)),
                    input=argmax_input,
                    output=current_tensor,
                )
            )
            saw_argmax = True
            current_node = node
            continue
        if node.op == "call_method" and str(node.target) == "argmax":
            _require_sequential_input(node, current_node, allow_parameters=True)
            _validate_argmax_arguments(node, is_method=True)
            argmax_input = current_tensor
            current_tensor = TensorIR(name=_safe_name(str(node.name)), shape=(), dtype="int64")
            ops.append(
                ArgmaxIR(
                    name=_safe_name(str(node.name)),
                    input=argmax_input,
                    output=current_tensor,
                )
            )
            saw_argmax = True
            current_node = node
            continue
        raise UnsupportedOpError(
            f"Unsupported FX node/op/module: op={node.op}, target={node.target}"
        )

    if current_node is None:
        raise UnsupportedOpError("Unsupported FX graph: missing input placeholder")
    if not saw_output:
        raise UnsupportedOpError("Unsupported FX graph: missing output node")

    metadata: dict[str, object] = {"source": type(model).__name__}
    if has_batchnorm_fusion:
        metadata["input_adapter"] = {"kind": "singleton_batch_n1"}
        metadata["transformations"] = transformations
    graph = GraphIR(
        input=input_tensor,
        output=current_tensor,
        ops=tuple(ops),
        metadata=metadata,
    )
    if has_batchnorm_fusion:
        _validate_fusion_boundary(
            source_model=model,
            fused_model=traced,
            input_shape=normalized_input_shape,
            float_dtype=float_dtype,
            nn=nn,
            torch=torch,
        )
    _validate_lowered_semantics(
        model=traced if has_batchnorm_fusion else model,
        graph=graph,
        input_shape=normalized_input_shape,
        float_dtype=float_dtype,
        nn=nn,
        torch=torch,
        copy_model=not has_batchnorm_fusion,
    )
    return graph


def _fuse_conv_batchnorm_eval(
    traced: object,
    nn: object,
    torch: object,
) -> tuple[object, list[dict[str, str]]]:
    modules = dict(_trusted_named_modules(traced, nn))
    candidates: list[tuple[object, object, object, object]] = []
    for node in tuple(traced.graph.nodes):
        if node.op != "call_module":
            continue
        module = modules.get(str(node.target))
        if type(module) is not nn.BatchNorm2d:
            continue
        args = tuple(node.args)
        kwargs = dict(node.kwargs)
        if len(args) != 1 or kwargs or not _is_fx_node(args[0]):
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d call at node {node.name}: expected "
                "one direct Conv2d tensor input"
            )
        conv_node = args[0]
        conv = (
            modules.get(str(conv_node.target))
            if conv_node.op == "call_module"
            else None
        )
        if type(conv) is not nn.Conv2d:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d at node {node.name}: only an exact "
                "Conv2d -> BatchNorm2d pattern can be fused"
            )
        if _fx_node_use_count(traced.graph, conv_node) != 1:
            raise UnsupportedOpError(
                f"Unsupported Conv2d fan-out at node {conv_node.name}: "
                "fusion requires exactly one FX edge use"
            )
        if _fx_node_use_count(traced.graph, node) != 1:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d fan-out at node {node.name}: "
                "fusion requires exactly one FX edge use"
            )
        candidates.append((conv_node, conv, node, module))

    if not candidates:
        return traced, []

    _require_eval_module(traced, "root model")
    transformations: list[dict[str, str]] = []
    for index, (conv_node, conv, batchnorm_node, batchnorm) in enumerate(candidates):
        _validate_conv_batchnorm_pair(conv, batchnorm, nn, torch)
        try:
            fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, batchnorm)
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise UnsupportedOpError(
                "Unsupported Conv2d -> BatchNorm2d fusion at nodes "
                f"{conv_node.name} -> {batchnorm_node.name}: {exc}"
            ) from exc
        _validate_fused_conv(fused, conv, nn)
        fused_target = _fresh_fused_target(traced, index)
        traced.add_module(fused_target, fused)
        original_conv_target = str(conv_node.target)
        original_batchnorm_target = str(batchnorm_node.target)
        conv_node.target = fused_target
        batchnorm_node.replace_all_uses_with(conv_node)
        traced.graph.erase_node(batchnorm_node)
        transformations.append(
            {
                "kind": "conv2d_batchnorm2d_fusion",
                "conv_node": str(conv_node.name),
                "conv_target": original_conv_target,
                "batchnorm_node": str(batchnorm_node.name),
                "batchnorm_target": original_batchnorm_target,
                "fused_target": fused_target,
            }
        )

    traced.graph.lint()
    traced.recompile()
    traced.delete_all_unused_submodules()
    traced.graph.lint()
    traced.recompile()
    remaining_modules = dict(_trusted_named_modules(traced, nn))
    for node in traced.graph.nodes:
        if node.op == "call_module" and type(
            remaining_modules.get(str(node.target))
        ) is nn.BatchNorm2d:
            raise UnsupportedOpError(
                f"Unsupported unfused BatchNorm2d at FX node {node.name}"
            )
    return traced, transformations


def _validate_conv_batchnorm_pair(
    conv: object,
    batchnorm: object,
    nn: object,
    torch: object,
) -> None:
    if type(conv) is not nn.Conv2d or type(batchnorm) is not nn.BatchNorm2d:
        raise UnsupportedOpError(
            "Unsupported fusion module types: exact Conv2d and BatchNorm2d "
            "instances are required"
        )
    conv_state = _raw_module_state(conv)
    if {"weight", "bias"}.intersection(conv_state):
        raise UnsupportedOpError(
            "Unsupported Conv2d instance schema: parameter slots must not be "
            "shadowed by direct attributes"
        )
    _require_eval_module(conv, "Conv2d")
    _require_eval_module(batchnorm, "BatchNorm2d")

    out_channels = _positive_module_integer(
        _required_module_attribute(conv, "Conv2d", "out_channels"),
        "Conv2d",
        "out_channels",
    )
    num_features = _positive_module_integer(
        _required_module_attribute(batchnorm, "BatchNorm2d", "num_features"),
        "BatchNorm2d",
        "num_features",
    )
    if num_features != out_channels:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d num_features: expected Conv2d "
            f"out_channels={out_channels}, got {num_features}"
        )

    state = _raw_module_state(batchnorm)
    parameters = state.get("_parameters")
    buffers = state.get("_buffers")
    children = state.get("_modules")
    registry_names = {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    shadowed_names = registry_names.intersection(state)
    if shadowed_names:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d instance schema: parameter and buffer "
            "slots must not be shadowed by direct attributes"
        )
    if type(children) is not dict or children:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d module schema: child modules are not allowed"
        )
    if type(parameters) is not dict or set(parameters) != {"weight", "bias"}:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d parameter schema: expected exactly "
            "weight and bias slots"
        )
    expected_buffers = {"running_mean", "running_var", "num_batches_tracked"}
    if type(buffers) is not dict or set(buffers) != expected_buffers:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d buffer schema: expected exactly "
            "running_mean, running_var, and num_batches_tracked slots"
        )
    non_persistent = state.get("_non_persistent_buffers_set")
    if type(non_persistent) is not set or non_persistent:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d buffer schema: running statistics must "
            "be persistent"
        )

    affine = _required_module_attribute(batchnorm, "BatchNorm2d", "affine")
    if type(affine) is not bool:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d affine: expected a built-in boolean"
        )
    weight = parameters["weight"]
    bias = parameters["bias"]
    if affine:
        weight = _required_parameter(weight, "BatchNorm2d", "weight", nn)
        try:
            constructor = _raw_static_attribute(nn.BatchNorm2d, "__init__")
        except AttributeError as exc:
            raise UnsupportedOpError(
                "Unsupported BatchNorm2d constructor metadata"
            ) from exc
        if type(constructor) is not FunctionType:
            raise UnsupportedOpError(
                "Unsupported BatchNorm2d constructor metadata"
            )
        constructor_code = constructor.__code__
        kwonly_start = constructor_code.co_argcount
        kwonly_stop = kwonly_start + constructor_code.co_kwonlyargcount
        kwdefaults = constructor.__kwdefaults__
        supports_optional_bias = (
            "bias" in constructor_code.co_varnames[kwonly_start:kwonly_stop]
            and type(kwdefaults) is dict
            and kwdefaults.get("bias") is True
        )
        if supports_optional_bias:
            bias = _optional_parameter(bias, "BatchNorm2d", "bias", nn)
        else:
            bias = _required_parameter(bias, "BatchNorm2d", "bias", nn)
    elif weight is not None or bias is not None:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d parameter schema: affine=False requires "
            "empty weight and bias slots"
        )

    track_running_stats = _required_module_attribute(
        batchnorm,
        "BatchNorm2d",
        "track_running_stats",
    )
    if type(track_running_stats) is not bool or not track_running_stats:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d track_running_stats: running statistics "
            "are required for eval fusion"
        )
    running_mean = buffers["running_mean"]
    running_var = buffers["running_var"]
    counter = buffers["num_batches_tracked"]
    for name, tensor in (
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if type(tensor) is not torch.Tensor:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name}: expected a Tensor"
            )
        if tuple(tensor.shape) != (num_features,):
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name} shape: expected "
                f"({num_features},), got {tuple(tensor.shape)}"
            )
        if tensor.requires_grad:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name}: buffers must not require gradients"
            )
    if (
        type(counter) is not torch.Tensor
        or counter.dtype is not torch.int64
        or tuple(counter.shape) != ()
        or counter.requires_grad
    ):
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d num_batches_tracked: expected a "
            "non-gradient scalar int64 Tensor"
        )
    counter_value = int(counter.detach().cpu().numpy().item())
    if counter_value < 0:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d num_batches_tracked: expected a "
            "non-negative value"
        )

    expected_shape = (num_features,)
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None and tuple(tensor.shape) != expected_shape:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name} shape: expected "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )

    conv_weight = _required_parameter(
        _required_module_attribute(conv, "Conv2d", "weight"),
        "Conv2d",
        "weight",
        nn,
    )
    conv_bias = _optional_parameter(
        _required_module_attribute(conv, "Conv2d", "bias"),
        "Conv2d",
        "bias",
        nn,
    )
    floating_tensors = [conv_weight, running_mean, running_var]
    floating_tensors.extend(
        tensor for tensor in (conv_bias, weight, bias) if tensor is not None
    )
    dtype = conv_weight.dtype
    if dtype not in (torch.float32, torch.float64) or any(
        tensor.dtype is not dtype for tensor in floating_tensors
    ):
        raise UnsupportedOpError(
            "Unsupported Conv2d/BatchNorm2d dtype: all fusion parameters and "
            "statistics must use one float32 or float64 dtype"
        )
    for name, tensor in (
        ("Conv2d weight", conv_weight),
        ("Conv2d bias", conv_bias),
        ("BatchNorm2d weight", weight),
        ("BatchNorm2d bias", bias),
        ("BatchNorm2d running_mean", running_mean),
        ("BatchNorm2d running_var", running_var),
    ):
        if tensor is not None and not _tensor_values_are_finite(tensor):
            raise UnsupportedOpError(
                f"Unsupported {name}: fusion inputs must contain only finite values"
            )

    eps = _required_module_attribute(batchnorm, "BatchNorm2d", "eps")
    if type(eps) is not float or not math.isfinite(eps) or eps < 0.0:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d eps: expected a finite non-negative "
            "built-in float"
        )
    running_var_values = running_var.detach().cpu().numpy()
    if np.any(running_var_values < 0):
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d running_var: values must be non-negative"
        )
    variance_with_eps = (running_var + eps).detach().cpu().numpy()
    if not np.all(np.isfinite(variance_with_eps)) or np.any(variance_with_eps <= 0):
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d running_var + eps: every channel must "
            "have a finite positive denominator"
        )


def _require_eval_module(module: object, label: str) -> None:
    training = _required_module_attribute(module, label, "training")
    if type(training) is not bool or training:
        raise UnsupportedOpError(
            f"Unsupported {label} training state: Conv2d/BatchNorm2d fusion "
            "requires eval mode"
        )


def _validate_fused_conv(
    fused: object,
    source_conv: object,
    nn: object,
) -> None:
    if type(fused) is not nn.Conv2d:
        raise UnsupportedOpError(
            "Unsupported official fusion result: expected an exact Conv2d"
        )
    weight = _required_parameter(
        _required_module_attribute(fused, "fused Conv2d", "weight"),
        "fused Conv2d",
        "weight",
        nn,
    )
    bias = _required_parameter(
        _required_module_attribute(fused, "fused Conv2d", "bias"),
        "fused Conv2d",
        "bias",
        nn,
    )
    source_weight = _required_parameter(
        _required_module_attribute(source_conv, "Conv2d", "weight"),
        "Conv2d",
        "weight",
        nn,
    )
    if weight.dtype is not source_weight.dtype or bias.dtype is not source_weight.dtype:
        raise UnsupportedOpError(
            "Unsupported official fusion result: fused parameter dtype changed"
        )
    if not _tensor_values_are_finite(weight) or not _tensor_values_are_finite(bias):
        raise UnsupportedOpError(
            "Unsupported official fusion result: fused Conv2d parameters "
            "must contain only finite values"
        )
    _require_eval_module(fused, "fused Conv2d")


def _tensor_values_are_finite(tensor: object) -> bool:
    values = tensor.detach().cpu().numpy()
    return bool(np.all(np.isfinite(values)))


def _fresh_fused_target(traced: object, index: int) -> str:
    base = f"_torch2rtl_fused_conv_bn_{index}"
    candidate = base
    suffix = 0
    state = _raw_module_state(traced)
    modules = state.get("_modules")
    if type(modules) is not dict:
        raise UnsupportedOpError("Unsupported FX GraphModule module registry")
    while candidate in state or candidate in modules or hasattr(traced, candidate):
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _fx_node_use_count(graph: object, target: object) -> int:
    count = 0
    for node in graph.nodes:
        count += _fx_value_reference_count(node.args, target)
        count += _fx_value_reference_count(node.kwargs, target)
    return count


def _fx_value_reference_count(value: object, target: object) -> int:
    if value is target:
        return 1
    if isinstance(value, (tuple, list)):
        count = 0
        for item in value:
            count += _fx_value_reference_count(item, target)
        return count
    if isinstance(value, dict):
        count = 0
        for key, item in value.items():
            count += _fx_value_reference_count(key, target)
            count += _fx_value_reference_count(item, target)
        return count
    if type(value) is slice:
        count = 0
        for item in (value.start, value.stop, value.step):
            count += _fx_value_reference_count(item, target)
        return count
    return 0


def _parse_module_node(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
    nn: object,
) -> tuple[TensorIR, object]:
    if isinstance(module, nn.Linear):
        module_in_features = _positive_module_integer(
            _required_module_attribute(module, "Linear", "in_features"),
            "Linear",
            "in_features",
        )
        module_out_features = _positive_module_integer(
            _required_module_attribute(module, "Linear", "out_features"),
            "Linear",
            "out_features",
        )
        weight = _required_parameter(
            _required_module_attribute(module, "Linear", "weight"),
            "Linear",
            "weight",
            nn,
        )
        bias = _optional_parameter(
            _required_module_attribute(module, "Linear", "bias"),
            "Linear",
            "bias",
            nn,
        )
        if len(current_tensor.shape) != 1:
            raise UnsupportedOpError(
                "Unsupported Linear input shape: only a 1-D feature vector is "
                f"supported, got {current_tensor.shape}; flatten explicitly first"
            )
        in_features = int(current_tensor.shape[0])
        if in_features != module_in_features:
            raise UnsupportedOpError(
                "Unsupported Linear input shape: "
                f"expected ({module.in_features},), got {current_tensor.shape}"
            )
        expected_weight_shape = (module_out_features, module_in_features)
        if tuple(weight.shape) != expected_weight_shape:
            raise UnsupportedOpError(
                "Unsupported Linear weight shape: "
                f"expected {expected_weight_shape}, got {tuple(weight.shape)}"
            )
        if bias is not None and tuple(bias.shape) != (
            module_out_features,
        ):
            raise UnsupportedOpError(
                "Unsupported Linear bias shape: "
                f"expected ({module.out_features},), got {tuple(module.bias.shape)}"
            )
        output_shape = (module_out_features,)
        output = TensorIR(
            name=f"{node_name}_out",
            shape=output_shape,
            dtype=current_tensor.dtype,
        )
        bias_values = None
        if bias is not None:
            bias_values = bias.detach().cpu().numpy().copy()
        op = LinearIR(
            name=node_name,
            input=current_tensor,
            output=output,
            in_features=module_in_features,
            out_features=module_out_features,
            weight=weight.detach().cpu().numpy().copy(),
            bias=bias_values,
        )
        return output, op

    if isinstance(module, nn.Conv2d):
        module_in_channels = _positive_module_integer(
            _required_module_attribute(module, "Conv2d", "in_channels"),
            "Conv2d",
            "in_channels",
        )
        module_out_channels = _positive_module_integer(
            _required_module_attribute(module, "Conv2d", "out_channels"),
            "Conv2d",
            "out_channels",
        )
        groups = _positive_module_integer(
            _required_module_attribute(module, "Conv2d", "groups"),
            "Conv2d",
            "groups",
        )
        weight = _required_parameter(
            _required_module_attribute(module, "Conv2d", "weight"),
            "Conv2d",
            "weight",
            nn,
        )
        bias = _optional_parameter(
            _required_module_attribute(module, "Conv2d", "bias"),
            "Conv2d",
            "bias",
            nn,
        )
        if len(current_tensor.shape) == 4:
            raise UnsupportedOpError(
                "Unsupported Conv2d input: batch dimension is not supported; "
                "use unbatched (C, H, W)"
            )
        if len(current_tensor.shape) != 3:
            raise UnsupportedOpError(
                f"Unsupported Conv2d input rank: expected (C, H, W), got {current_tensor.shape}"
            )
        if groups != 1:
            raise UnsupportedOpError("Unsupported Conv2d groups: only groups=1 is supported")
        padding_mode = _required_module_attribute(module, "Conv2d", "padding_mode")
        if type(padding_mode) is not str or padding_mode != "zeros":
            raise UnsupportedOpError(
                "Unsupported Conv2d padding_mode: only zero padding is supported"
            )
        dilation = _int_pair(
            _required_module_attribute(module, "Conv2d", "dilation"),
            "dilation",
        )
        if dilation != (1, 1):
            raise UnsupportedOpError("Unsupported Conv2d dilation: only dilation=1 is supported")
        padding = _int_pair(
            _required_module_attribute(module, "Conv2d", "padding"),
            "padding",
        )
        stride = _int_pair(
            _required_module_attribute(module, "Conv2d", "stride"),
            "stride",
        )
        kernel_height, kernel_width = _int_pair(
            _required_module_attribute(module, "Conv2d", "kernel_size"),
            "kernel_size",
        )
        _validate_conv_pair("kernel_size", (kernel_height, kernel_width), minimum=1)
        _validate_conv_pair("stride", stride, minimum=1)
        _validate_conv_pair("padding", padding, minimum=0)
        for attribute_name, values in (
            ("kernel_size", (kernel_height, kernel_width)),
            ("stride", stride),
            ("padding", padding),
        ):
            if any(value > _SV_INT_MAX for value in values):
                raise UnsupportedOpError(
                    f"Unsupported Conv2d {attribute_name}: values must fit a "
                    "signed 32-bit SystemVerilog int"
                )
        in_channels, input_height, input_width = (int(dim) for dim in current_tensor.shape)
        if in_channels != module_in_channels:
            raise UnsupportedOpError(
                "Unsupported Conv2d shape: "
                f"expected {module.in_channels} input channels, got {in_channels}"
            )
        output_height = _conv_output_dim(input_height, kernel_height, stride[0], padding[0])
        output_width = _conv_output_dim(input_width, kernel_width, stride[1], padding[1])
        if output_height <= 0 or output_width <= 0:
            raise UnsupportedOpError(
                "Unsupported Conv2d output shape: "
                f"got ({module.out_channels}, {output_height}, {output_width})"
            )
        _validate_tensor_element_count(
            (module_out_channels, output_height, output_width),
            "Conv2d output",
        )
        for output_size, step, kernel_size in (
            (output_height, stride[0], kernel_height),
            (output_width, stride[1], kernel_width),
        ):
            maximum_positive_coordinate = (output_size - 1) * step + kernel_size - 1
            if maximum_positive_coordinate > _SV_INT_MAX:
                raise UnsupportedOpError(
                    "Unsupported Conv2d coordinates: index arithmetic must fit a "
                    "signed 32-bit SystemVerilog int"
                )
        expected_weight_shape = (
            module_out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        )
        actual_weight_shape = tuple(int(dim) for dim in weight.shape)
        if actual_weight_shape != expected_weight_shape:
            raise UnsupportedOpError(
                "Unsupported Conv2d weight shape: "
                f"expected {expected_weight_shape}, got {actual_weight_shape}"
            )
        if bias is not None and tuple(bias.shape) != (
            module_out_channels,
        ):
            raise UnsupportedOpError(
                "Unsupported Conv2d bias shape: "
                f"expected ({module.out_channels},), got {tuple(module.bias.shape)}"
            )
        output = TensorIR(
            name=f"{node_name}_out",
            shape=(module_out_channels, output_height, output_width),
            dtype=current_tensor.dtype,
        )
        bias_values = None
        if bias is not None:
            bias_values = bias.detach().cpu().numpy().copy()
        op = Conv2dIR(
            name=node_name,
            input=current_tensor,
            output=output,
            in_channels=in_channels,
            out_channels=module_out_channels,
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            stride=stride,
            padding=padding,
            weight=weight.detach().cpu().numpy().copy(),
            bias=bias_values,
        )
        return output, op

    if isinstance(module, nn.ReLU):
        inplace = _required_module_attribute(module, "ReLU", "inplace")
        if type(inplace) is not bool:
            raise UnsupportedOpError(
                "Unsupported ReLU inplace: expected a built-in boolean"
            )
        if inplace:
            raise UnsupportedOpError(
                "Unsupported ReLU inplace=True: input mutation is not lowered"
            )
        output = TensorIR(
            name=f"{node_name}_out",
            shape=current_tensor.shape,
            dtype=current_tensor.dtype,
        )
        return output, ReluIR(name=node_name, input=current_tensor, output=output)

    if isinstance(module, nn.Flatten):
        start_dim = _module_integer(
            _required_module_attribute(module, "Flatten", "start_dim"),
            "Flatten",
            "start_dim",
        )
        end_dim = _module_integer(
            _required_module_attribute(module, "Flatten", "end_dim"),
            "Flatten",
            "end_dim",
        )
        output_shape = _flatten_shape(
            shape=current_tensor.shape,
            start_dim=start_dim,
            end_dim=end_dim,
        )
        if len(output_shape) != 1:
            raise UnsupportedOpError(
                "Unsupported Flatten: only flattening the complete tensor to one "
                f"dimension is supported, got output shape {output_shape}"
            )
        output = TensorIR(
            name=f"{node_name}_out",
            shape=output_shape,
            dtype=current_tensor.dtype,
        )
        return output, FlattenIR(name=node_name, input=current_tensor, output=output)

    raise UnsupportedOpError(
        f"Unsupported FX node/op/module: module={type(module).__name__}"
    )


def _flatten_shape(shape: tuple[int, ...], start_dim: int, end_dim: int) -> tuple[int, ...]:
    rank = len(shape)
    if rank == 0:
        return shape
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += rank
    if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
        raise UnsupportedOpError(
            f"Unsupported Flatten dims: start_dim={start_dim}, end_dim={end_dim}"
        )
    flattened = math.prod(shape[start_dim : end_dim + 1])
    return (*shape[:start_dim], flattened, *shape[end_dim + 1 :])


def _validate_model_semantics(
    model: object,
    nn: object,
    torch: object,
    input_dtype: object | None,
) -> str:
    if not isinstance(model, nn.Module):
        raise UnsupportedOpError(
            f"Unsupported model type: expected torch.nn.Module, got {type(model).__name__}"
        )

    module_api = torch.nn.modules.module
    for registry_name in (
        "_global_buffer_registration_hooks",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
    ):
        registry = getattr(module_api, registry_name, {})
        if registry:
            raise UnsupportedOpError(
                f"Unsupported global hook registry {registry_name}: "
                "hook semantics are not lowered"
            )

    lowered_types = (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Flatten)
    modules = tuple(_trusted_named_modules(model, nn))
    for module_name, module in modules:
        display_name = module_name or "<root>"
        module_type = type(module)
        if type(module_type) is not type:
            type_name = type.__getattribute__(module_type, "__name__")
            raise UnsupportedOpError(
                f"Unsupported custom metaclass on module type {type_name}"
            )
        state = _raw_module_state(module)
        _validate_module_dispatch(module, display_name, nn)
        _validate_no_custom_copy_protocol(type(module), nn)
        _validate_no_custom_class_state(type(module), nn)
        _validate_copy_safe_instance_state(module, display_name, torch)
        _validate_forward_python_state(type(module), nn, torch)
        if state.get("_forward_hooks"):
            raise UnsupportedOpError(
                f"Unsupported forward hook on module {display_name}: "
                "hook semantics are not lowered"
            )
        if state.get("_forward_pre_hooks"):
            raise UnsupportedOpError(
                f"Unsupported forward pre-hook on module {display_name}: "
                "hook semantics are not lowered"
            )
        if state.get("_backward_hooks") or state.get("_backward_pre_hooks"):
            raise UnsupportedOpError(
                f"Unsupported backward hook on module {display_name}: "
                "hook semantics are not lowered"
            )
        if isinstance(module, lowered_types):
            if type(module) not in lowered_types:
                raise UnsupportedOpError(
                    f"Unsupported module subclass {type(module).__name__}: "
                    "custom forward semantics are not lowered"
                )

    supported = {
        torch.float32: "float32",
        torch.float64: "float64",
    }
    model_dtypes: set[object] = set()
    for tensor_name, tensor, is_batchnorm_counter in _trusted_named_tensors(
        modules,
        torch,
    ):
        if is_batchnorm_counter:
            continue
        if tensor.dtype not in supported:
            raise UnsupportedOpError(
                f"Unsupported dtype {tensor.dtype} for tensor {tensor_name}; "
                "only float32 and float64 are supported"
            )
        model_dtypes.add(tensor.dtype)
    if len(model_dtypes) > 1:
        names = ", ".join(sorted(str(dtype) for dtype in model_dtypes))
        raise UnsupportedOpError(
            f"Unsupported mixed model dtypes: {names}; use one floating dtype"
        )

    requested_dtype = _normalize_float_dtype(input_dtype, torch, supported)
    model_dtype = next(iter(model_dtypes), None)
    if requested_dtype is not None and model_dtype is not None and requested_dtype != model_dtype:
        raise UnsupportedOpError(
            f"Unsupported input dtype {requested_dtype} for model dtype {model_dtype}"
        )
    resolved = requested_dtype or model_dtype or torch.get_default_dtype()
    if resolved not in supported:
        raise UnsupportedOpError(
            f"Unsupported dtype {resolved}; only float32 and float64 are supported"
        )
    for module_name, module in modules:
        _validate_no_custom_introspection(module, module_name or "<root>", nn)
    return supported[resolved]


def _validate_module_dispatch(
    module: object,
    display_name: str,
    nn: object,
) -> None:
    module_type = type(module)
    state = _raw_module_state(module)
    for attribute in (
        "__call__",
        "_wrapped_call_impl",
        "_call_impl",
        "_compiled_call_impl",
    ):
        if _raw_static_attribute(module_type, attribute) is not _raw_static_attribute(
            nn.Module,
            attribute,
        ):
            label = "compiled call" if attribute == "_compiled_call_impl" else "call"
            raise UnsupportedOpError(
                f"Unsupported custom {label} path ({attribute}) on module "
                f"{display_name}: call semantics are not lowered"
            )
        if attribute in state:
            raise UnsupportedOpError(
                f"Unsupported instance call override ({attribute}) on module "
                f"{display_name}: call semantics are not lowered"
            )
    for attribute in ("__getattribute__", "__getattr__"):
        if _raw_static_attribute(module_type, attribute) is not _raw_static_attribute(
            nn.Module,
            attribute,
        ):
            raise UnsupportedOpError(
                f"Unsupported custom call path ({attribute}) on module "
                f"{display_name}: call semantics are not lowered"
            )
    if "forward" in state:
        raise UnsupportedOpError(
            f"Unsupported instance forward override on module {display_name}"
        )


def _validate_no_custom_introspection(
    module: object,
    display_name: str,
    nn: object,
) -> None:
    state = _raw_module_state(module)
    for attribute in (
        "named_modules",
        "named_parameters",
        "named_buffers",
        "named_children",
        "modules",
        "parameters",
        "buffers",
        "children",
    ):
        if _raw_static_attribute(type(module), attribute) is not _raw_static_attribute(
            nn.Module,
            attribute,
        ) or attribute in state:
            raise UnsupportedOpError(
                f"Unsupported module introspection override ({attribute}) on "
                f"module {display_name}"
            )


def _copy_model_for_tracing(model: object, nn: object, torch: object) -> object:
    before_state = _snapshot_module_state(model, nn, torch)
    class_state = _snapshot_module_class_definitions(model, nn)
    try:
        copied = copy.deepcopy(model)
    except Exception as exc:
        _restore_module_class_definitions(class_state)
        raise UnsupportedOpError(
            f"Unsupported model state: model cannot be copied safely for FX trace: {exc}"
        ) from exc
    class_changed = _restore_module_class_definitions(class_state)
    original_changed = before_state != _snapshot_module_state(model, nn, torch)
    if class_changed or original_changed:
        raise UnsupportedOpError(
            "Unsupported model copy: deepcopy mutated Python module state"
        )
    if not isinstance(copied, nn.Module):
        raise UnsupportedOpError(
            "Unsupported model copy: deepcopy did not return an nn.Module"
        )
    if before_state != _snapshot_module_state(copied, nn, torch):
        raise UnsupportedOpError(
            "Unsupported model copy: deepcopy changed or substituted module semantics"
        )
    return copied


def _validate_no_custom_copy_protocol(module_type: type[object], nn: object) -> None:
    protocol_names = {
        "__deepcopy__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
    }
    framework_classes = _framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        for name in protocol_names & vars(cls).keys():
            raise UnsupportedOpError(
                f"Unsupported custom model copy protocol {cls.__name__}.{name}"
            )


def _validate_no_custom_class_state(root_type: type[object], nn: object) -> None:
    metadata_names = {"__module__", "__qualname__", "__doc__"}
    framework_classes = _framework_module_classes(nn)
    for cls in root_type.__mro__:
        if cls in framework_classes:
            break
        for name, value in vars(cls).items():
            if name in metadata_names:
                continue
            if name == "__annotations__" and type(value) is dict and not value:
                continue
            if type(value) in (staticmethod, classmethod):
                continue
            if isinstance(value, FunctionType):
                if name == "__init__" or not (
                    name.startswith("__") and name.endswith("__")
                ):
                    continue
                raise UnsupportedOpError(
                    f"Unsupported custom class descriptor {cls.__name__}.{name}"
                )
            raise UnsupportedOpError(
                f"Unsupported custom class-level state or descriptor "
                f"{cls.__name__}.{name}"
            )


def _validate_forward_python_state(
    module_type: type[object],
    nn: object,
    torch: object,
) -> None:
    framework_classes = _framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        forward = vars(cls).get("forward")
        if forward is None:
            continue
        _validate_reachable_python_function(
            function=forward,
            label=f"{cls.__name__}.forward",
            module_type=module_type,
            nn=nn,
            torch=torch,
            seen=set(),
        )


def _validate_reachable_python_function(
    function: object,
    label: str,
    module_type: type[object],
    nn: object,
    torch: object,
    seen: set[int],
) -> None:
    if not isinstance(function, FunctionType):
        raise UnsupportedOpError(f"Unsupported callable descriptor in {label}")
    identity = id(function)
    if identity in seen:
        return
    seen.add(identity)
    if "__wrapped__" in function.__dict__:
        raise UnsupportedOpError(f"Unsupported decorated function {label}")
    if function.__dict__:
        raise UnsupportedOpError(f"Unsupported custom function state in {label}")
    if function.__builtins__ is not _TRUSTED_BUILTINS:
        raise UnsupportedOpError(f"Unsupported custom builtins mapping in {label}")
    if builtins.len is not _TRUSTED_LEN:
        raise UnsupportedOpError(f"Unsupported modified builtins binding in {label}")
    if any(_contains_code_object(value) for value in function.__code__.co_consts):
        raise UnsupportedOpError(f"Unsupported nested code object in {label}")
    if any(
        not _is_immutable_python_state(value, set())
        for value in function.__code__.co_consts
    ):
        raise UnsupportedOpError(f"Unsupported code constant in {label}")
    defaults = (
        *tuple(function.__defaults__ or ()),
        *tuple((function.__kwdefaults__ or {}).values()),
    )
    if any(not _is_immutable_python_state(value, set()) for value in defaults):
        raise UnsupportedOpError(f"Unsupported external Python state in {label} defaults")
    if function.__closure__ is not None:
        for cell in function.__closure__:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if not _is_immutable_python_state(value, set()):
                raise UnsupportedOpError(f"Unsupported external Python state in {label}")
    bytecode = dis.Bytecode(function)
    if bytecode.exception_entries:
        raise UnsupportedOpError(f"Unsupported exception handling in {label}")
    instructions = tuple(bytecode)
    for instruction in instructions:
        if instruction.opname == "FORMAT_VALUE":
            raise UnsupportedOpError(
                f"Unsupported tensor-dependent string formatting in {label}"
            )
        if instruction.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"}:
            raise UnsupportedOpError(f"Unsupported external Python state in {label}")
        if instruction.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
            continue
        method_name = str(instruction.argval)
        if method_name in {"format", "node", "tracer"}:
            raise UnsupportedOpError(
                f"Unsupported Python introspection/string formatting attribute "
                f"{method_name!r} in {label}"
            )
        if method_name in {"_parameters", "_buffers", "_modules"}:
            raise UnsupportedOpError(
                f"Unsupported internal module registry access {method_name!r} in {label}"
            )
        if method_name.startswith("__") and method_name.endswith("__"):
            raise UnsupportedOpError(
                f"Unsupported Python introspection attribute {method_name!r} in {label}"
            )
    if any(
        instruction.opcode in dis.hasjabs or instruction.opcode in dis.hasjrel
        for instruction in instructions
    ):
        raise UnsupportedOpError(
            "Unsupported FX symbolic trace control flow/state semantics in "
            f"{label}"
        )
    supported_opcodes = {
        "BINARY_OP",
        "BUILD_TUPLE",
        "CACHE",
        "CALL",
        "COPY_FREE_VARS",
        "EXTENDED_ARG",
        "KW_NAMES",
        "LOAD_ATTR",
        "LOAD_CONST",
        "LOAD_DEREF",
        "LOAD_FAST",
        "LOAD_GLOBAL",
        "LOAD_METHOD",
        "NOP",
        "POP_TOP",
        "PRECALL",
        "PUSH_NULL",
        "RESUME",
        "RETURN_VALUE",
        "STORE_FAST",
    }
    for index, instruction in enumerate(instructions):
        if instruction.opname == "FORMAT_VALUE":
            raise UnsupportedOpError(
                f"Unsupported tensor-dependent string formatting in {label}"
            )
        if instruction.opname == "BINARY_OP" and instruction.argrepr == "%":
            raise UnsupportedOpError(
                f"Unsupported tensor-dependent string formatting in {label}"
            )
        if instruction.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"}:
            raise UnsupportedOpError(f"Unsupported external Python state in {label}")
        if instruction.opname not in supported_opcodes:
            raise UnsupportedOpError(
                f"Unsupported custom Python bytecode {instruction.opname} in {label}"
            )
        if instruction.opname == "LOAD_GLOBAL":
            name = str(instruction.argval)
            if name not in function.__globals__:
                if name == "len":
                    continue
                raise UnsupportedOpError(
                    f"Unsupported external Python state {name!r} in {label}"
                )
            value = function.__globals__[name]
            if value is torch and _next_loaded_attribute(instructions, index) == "argmax":
                continue
            raise UnsupportedOpError(
                f"Unsupported external Python state {name!r} in {label}"
            )
        if instruction.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
            continue
        method_name = str(instruction.argval)
        if method_name in {"format", "node", "tracer"}:
            raise UnsupportedOpError(
                f"Unsupported Python introspection/string formatting attribute "
                f"{method_name!r} in {label}"
            )
        if method_name in {"_parameters", "_buffers", "_modules"}:
            raise UnsupportedOpError(
                f"Unsupported internal module registry access {method_name!r} in {label}"
            )
        if method_name.startswith("__") and method_name.endswith("__"):
            raise UnsupportedOpError(
                f"Unsupported Python introspection attribute {method_name!r} in {label}"
            )
        resolved = _find_custom_method(module_type, method_name, nn)
        if resolved is not None:
            owner, method = resolved
            _validate_reachable_python_function(
                function=method,
                label=f"{owner.__name__}.{method_name}",
                module_type=module_type,
                nn=nn,
                torch=torch,
                seen=seen,
            )


def _find_custom_method(
    module_type: type[object],
    name: str,
    nn: object,
) -> tuple[type[object], object] | None:
    framework_classes = _framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        if name in vars(cls):
            value = vars(cls)[name]
            if isinstance(value, FunctionType):
                return cls, value
            if type(value) in (staticmethod, classmethod):
                return cls, value.__func__
            return None
    return None


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
) -> object:
    # This wrapper is the bootstrap trust boundary. Its references live in
    # closure cells, so replacing their module-global mirrors cannot affect
    # the code that detects the replacement.
    def validate(nn: object) -> None:
        for key in frontend_globals:
            if trusted_type(key) is not trusted_str_type:
                raise trusted_error("Unsupported modified frontend runtime keys")
        for name, value in trusted_global_bindings:
            if frontend_globals.get(name) is not value:
                raise trusted_error(
                    f"Unsupported modified frontend trust root {name}"
                )
        if frontend_globals.get("UnsupportedOpError") is not trusted_error:
            raise trusted_error("Unsupported modified frontend error binding")
        if (
            frontend_globals.get("_parse_model_impl")
            is not parse_implementation
        ):
            raise trusted_error("Unsupported modified frontend parser implementation")
        if (
            frontend_globals.get("_validate_framework_integrity_impl")
            is not implementation
        ):
            raise trusted_error(
                "Unsupported modified frontend integrity implementation"
            )
        if frontend_globals.get("_validate_framework_integrity") is not validate:
            raise trusted_error("Unsupported modified frontend integrity validator")
        if (
            frontend_globals.get("_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY")
            is not validate
        ):
            raise trusted_error("Unsupported modified frontend validator trust root")
        implementation(nn)

    return validate


_validate_framework_integrity = _make_integrity_validator(
    _validate_framework_integrity_impl,
    _parse_model_impl,
    _TRUSTED_FRONTEND_GLOBALS,
    _TRUSTED_TYPE,
    _TRUSTED_STR_TYPE,
    _TRUSTED_UNSUPPORTED_OP_ERROR,
    tuple(
        (name, value)
        for name, value in _TRUSTED_FRONTEND_GLOBALS.items()
        if name.startswith("_TRUSTED_")
    ),
)
_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY = _validate_framework_integrity


def _make_parse_model(
    implementation: object,
    validator: object,
    torch: object,
    nn: object,
    frontend_globals: dict[str, object],
    trusted_type: type[object],
    trusted_str_type: type[str],
    trusted_error: type[UnsupportedOpError],
) -> object:
    def parse_model(
        model: object,
        input_shape: Sequence[int],
        input_dtype: object | None = None,
    ) -> GraphIR:
        for key in frontend_globals:
            if trusted_type(key) is not trusted_str_type:
                raise trusted_error("Unsupported modified frontend runtime keys")
        if frontend_globals.get("parse_model") is not parse_model:
            raise trusted_error("Unsupported modified frontend parser binding")
        validator(nn)
        return implementation(model, input_shape, input_dtype, torch, nn)

    return parse_model


parse_model = _make_parse_model(
    _parse_model_impl,
    _validate_framework_integrity,
    _TRUSTED_TORCH_MODULE,
    _TRUSTED_TORCH_NN_MODULE,
    _TRUSTED_FRONTEND_GLOBALS,
    _TRUSTED_TYPE,
    _TRUSTED_STR_TYPE,
    _TRUSTED_UNSUPPORTED_OP_ERROR,
)
parse_model.__qualname__ = "parse_model"


def _next_loaded_attribute(
    instructions: tuple[dis.Instruction, ...],
    index: int,
) -> str | None:
    for following in instructions[index + 1 :]:
        if following.opname in {"CACHE", "EXTENDED_ARG", "NOP"}:
            continue
        if following.opname in {"LOAD_ATTR", "LOAD_METHOD"}:
            return str(following.argval)
        return None
    return None


def _is_immutable_python_state(value: object, seen: set[int]) -> bool:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return True
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    try:
        if type(value) is tuple:
            return all(_is_immutable_python_state(item, seen) for item in value)
        if type(value) is frozenset:
            return all(_is_immutable_python_state(item, seen) for item in value)
        return False
    finally:
        seen.remove(identity)


def _contains_code_object(value: object) -> bool:
    if isinstance(value, CodeType):
        return True
    if type(value) in (tuple, frozenset):
        return any(_contains_code_object(item) for item in value)
    return False


def _validate_copy_safe_instance_state(
    module: object,
    display_name: str,
    torch: object,
) -> None:
    for name, value in _raw_module_state(module).items():
        if name in {"_modules", "_parameters", "_buffers"}:
            continue
        if not _is_copy_safe_state_value(value, torch, set()):
            raise UnsupportedOpError(
                f"Unsupported external Python state {name!r} on module "
                f"{display_name}: value cannot be copied safely"
            )


def _is_copy_safe_state_value(value: object, torch: object, seen: set[int]) -> bool:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return True
    if type(value) in (torch.Tensor, torch.nn.Parameter, np.ndarray):
        return False
    if type(value) in (torch.dtype, torch.device):
        return True
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    try:
        if type(value) in (dict, OrderedDict):
            return all(
                _is_copy_safe_state_value(key, torch, seen)
                and _is_copy_safe_state_value(item, torch, seen)
                for key, item in value.items()
            )
        if type(value) in (list, tuple, set, frozenset):
            return all(_is_copy_safe_state_value(item, torch, seen) for item in value)
        return False
    finally:
        seen.remove(identity)


def _snapshot_module_class_definitions(
    model: object,
    nn: object,
) -> tuple[tuple[type[object], dict[str, object]], ...]:
    snapshots: list[tuple[type[object], dict[str, object]]] = []
    seen: set[type[object]] = set()
    framework_classes = _framework_module_classes(nn)
    for _, module in _trusted_named_modules(model, nn):
        for cls in type(module).__mro__:
            if cls in framework_classes:
                break
            if cls not in seen:
                snapshots.append((cls, dict(vars(cls))))
                seen.add(cls)
    return tuple(snapshots)


def _restore_module_class_definitions(
    snapshots: tuple[tuple[type[object], dict[str, object]], ...],
) -> bool:
    changed = False
    for cls, before in snapshots:
        current = dict(vars(cls))
        if current.keys() != before.keys() or any(
            current.get(name) is not value for name, value in before.items()
        ):
            changed = True
        for name in current.keys() - before.keys():
            delattr(cls, name)
        for name, value in before.items():
            if current.get(name) is not value:
                setattr(cls, name, value)
    return changed


def _raw_module_state(module: object) -> dict[str, object]:
    state = object.__getattribute__(module, "__dict__")
    if type(state) is not dict:
        raise UnsupportedOpError("Unsupported nn.Module state mapping")
    if any(type(name) is not str for name in state):
        raise UnsupportedOpError("Unsupported non-string nn.Module state key")
    return state


def _trusted_named_modules(model: object, nn: object) -> tuple[tuple[str, object], ...]:
    modules: list[tuple[str, object]] = []
    seen: set[int] = set()

    def visit(name: str, module: object) -> None:
        if not isinstance(module, nn.Module):
            raise UnsupportedOpError(
                f"Unsupported module tree entry {name or '<root>'}: expected nn.Module"
            )
        identity = id(module)
        if identity in seen:
            return
        seen.add(identity)
        modules.append((name, module))
        children = _raw_module_state(module).get("_modules")
        if type(children) is not dict:
            raise UnsupportedOpError(
                f"Unsupported module tree on {name or '<root>'}: _modules must be a dict"
            )
        for child_name, child in children.items():
            if type(child_name) is not str:
                raise UnsupportedOpError("Unsupported non-string module name")
            if child is None:
                continue
            qualified = f"{name}.{child_name}" if name else child_name
            visit(qualified, child)

    visit("", model)
    return tuple(modules)


def _trusted_named_tensors(
    modules: tuple[tuple[str, object], ...],
    torch: object,
) -> tuple[tuple[str, object, bool], ...]:
    tensors: list[tuple[str, object, bool]] = []
    for module_name, module in modules:
        state = _raw_module_state(module)
        for registry_name in ("_parameters", "_buffers"):
            registry = state.get(registry_name)
            if type(registry) is not dict:
                raise UnsupportedOpError(
                    f"Unsupported module tensor registry {registry_name} on "
                    f"{module_name or '<root>'}"
                )
            for tensor_name, tensor in registry.items():
                if type(tensor_name) is not str:
                    raise UnsupportedOpError(
                        f"Unsupported non-string tensor name in {registry_name}"
                    )
                if tensor is None:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    raise UnsupportedOpError(
                        f"Unsupported non-tensor value in {registry_name} on "
                        f"{module_name or '<root>'}"
                    )
                expected_type = (
                    torch.nn.Parameter
                    if registry_name == "_parameters"
                    else torch.Tensor
                )
                if type(tensor) is not expected_type:
                    raise UnsupportedOpError(
                        f"Unsupported tensor subclass {type(tensor).__name__} in "
                        f"{registry_name} on {module_name or '<root>'}"
                    )
                if tensor.device.type != "cpu":
                    raise UnsupportedOpError(
                        f"Unsupported tensor device {tensor.device} for "
                        f"{module_name or '<root>'}.{tensor_name}; only CPU is supported"
                    )
                if tensor.layout is not torch.strided:
                    raise UnsupportedOpError(
                        f"Unsupported tensor layout {tensor.layout} for "
                        f"{module_name or '<root>'}.{tensor_name}; only strided is supported"
                    )
                if not tensor.is_contiguous():
                    raise UnsupportedOpError(
                        f"Unsupported non-contiguous tensor "
                        f"{module_name or '<root>'}.{tensor_name}"
                    )
                if tensor.is_conj():
                    raise UnsupportedOpError(
                        f"Unsupported conjugate view bit on tensor "
                        f"{module_name or '<root>'}.{tensor_name}"
                    )
                if tensor.is_neg():
                    raise UnsupportedOpError(
                        f"Unsupported negative view bit on tensor "
                        f"{module_name or '<root>'}.{tensor_name}"
                    )
                if tensor.grad is not None:
                    raise UnsupportedOpError(
                        f"Unsupported gradient state on tensor "
                        f"{module_name or '<root>'}.{tensor_name}; "
                        "autograd state is not lowered"
                    )
                qualified = f"{module_name}.{tensor_name}" if module_name else tensor_name
                is_batchnorm_counter = (
                    type(module) is torch.nn.BatchNorm2d
                    and registry_name == "_buffers"
                    and tensor_name == "num_batches_tracked"
                )
                if is_batchnorm_counter and (
                    tensor.dtype is not torch.int64 or tuple(tensor.shape) != ()
                ):
                    raise UnsupportedOpError(
                        "Unsupported BatchNorm2d num_batches_tracked: expected "
                        "a scalar int64 Tensor"
                    )
                tensors.append((qualified, tensor, is_batchnorm_counter))
    return tuple(tensors)


def _snapshot_module_state(
    model: object,
    nn: object,
    torch: object,
) -> tuple[object, ...]:
    snapshots: list[object] = []
    for module_name, module in _trusted_named_modules(model, nn):
        state = _raw_module_state(module)
        attributes = tuple(
            sorted(
                (
                    name,
                    _freeze_state_value(value, torch, set()),
                )
                for name, value in state.items()
                if name != "_modules"
            )
        )
        child_registry = state["_modules"]
        children = tuple(
            (name, type(child))
            for name, child in child_registry.items()
            if child is not None
        )
        snapshots.append(
            (
                module_name,
                type(module),
                attributes,
                children,
            )
        )
    return tuple(snapshots)


def _freeze_state_value(value: object, torch: object, seen: set[int]) -> object:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return (type(value).__qualname__, repr(value))
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        data = (
            value.detach()
            .resolve_conj()
            .resolve_neg()
            .cpu()
            .contiguous()
            .numpy()
        )
        gradient = None
        if value.grad is not None:
            gradient = _freeze_state_value(value.grad, torch, seen)
        return (
            "tensor",
            type(value).__module__,
            type(value).__qualname__,
            str(value.dtype),
            str(value.device),
            str(value.layout),
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            value.requires_grad,
            value.is_conj(),
            value.is_neg(),
            gradient,
            data.tobytes(),
        )
    if type(value) is np.ndarray:
        data = np.ascontiguousarray(value)
        return (
            "ndarray",
            str(value.dtype),
            tuple(value.shape),
            tuple(value.strides),
            value.flags.c_contiguous,
            value.flags.f_contiguous,
            value.flags.owndata,
            value.flags.writeable,
            value.flags.aligned,
            data.tobytes(),
        )

    identity = id(value)
    if identity in seen:
        return ("cycle", identity)
    seen.add(identity)
    try:
        if isinstance(value, dict):
            items = [
                (
                    _freeze_state_value(key, torch, seen),
                    _freeze_state_value(item, torch, seen),
                )
                for key, item in value.items()
            ]
            return ("dict", tuple(sorted(items, key=repr)))
        if isinstance(value, (list, tuple)):
            return (
                type(value).__qualname__,
                tuple(_freeze_state_value(item, torch, seen) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            items = [_freeze_state_value(item, torch, seen) for item in value]
            return (type(value).__qualname__, tuple(sorted(items, key=repr)))
        if callable(value):
            return (
                "callable",
                getattr(value, "__module__", ""),
                getattr(value, "__qualname__", repr(value)),
            )
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            return (
                "object",
                type(value).__module__,
                type(value).__qualname__,
                _freeze_state_value(state, torch, seen),
            )
        return ("value", type(value).__module__, type(value).__qualname__, repr(value))
    finally:
        seen.remove(identity)


def _validate_fusion_boundary(
    source_model: object,
    fused_model: object,
    input_shape: tuple[int, ...],
    float_dtype: str,
    nn: object,
    torch: object,
) -> None:
    dtype = torch.float32 if float_dtype == "float32" else torch.float64
    numpy_dtype = np.dtype(float_dtype)
    element_count = math.prod(input_shape)
    probe_values = (
        np.zeros(input_shape, dtype=numpy_dtype),
        np.linspace(-0.75, 0.75, num=element_count, dtype=numpy_dtype).reshape(
            input_shape
        ),
    )
    rtol, atol = (
        (1e-5, 1e-6) if float_dtype == "float32" else (1e-12, 1e-12)
    )
    for values in probe_values:
        probe_input = torch.from_numpy(values.copy()).to(dtype=dtype)
        source_output = _run_concrete_probe(
            source_model,
            probe_input.unsqueeze(0),
            "source singleton-batch",
            nn,
            torch,
        )
        fused_output = _run_concrete_probe(
            fused_model,
            probe_input,
            "fused batchless",
            nn,
            torch,
            copy_model=False,
        )
        if not isinstance(source_output, torch.Tensor) or not isinstance(
            fused_output,
            torch.Tensor,
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source and fused models must "
                "return one Tensor"
            )
        if source_output.dtype != fused_output.dtype:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source and fused output dtypes "
                f"differ ({source_output.dtype} != {fused_output.dtype})"
            )

        source_values = source_output.detach().cpu().numpy()
        fused_values = fused_output.detach().cpu().numpy()
        if source_values.shape == () or fused_values.shape == ():
            if (
                source_values.shape != ()
                or fused_values.shape != ()
                or source_output.dtype is not torch.int64
                or source_values.item() != fused_values.item()
            ):
                raise UnsupportedOpError(
                    "Unsupported fusion semantics: scalar Argmax/class_id "
                    "changed across Conv2d/BatchNorm2d fusion"
                )
            continue

        if source_output.dtype is not dtype or fused_output.dtype is not dtype:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: tensor outputs must preserve "
                f"the model dtype {dtype}"
            )
        if source_values.shape[0] != 1:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source tensor output must have "
                "a leading singleton batch dimension"
            )
        source_batchless = source_values[0]
        if source_batchless.shape != fused_values.shape:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source singleton-batch output "
                f"shape {source_batchless.shape} does not match fused batchless "
                f"shape {fused_values.shape}"
            )
        if not np.all(np.isfinite(source_batchless)) or not np.all(
            np.isfinite(fused_values)
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source and fused outputs must "
                "contain only finite values"
            )
        if not np.allclose(
            source_batchless,
            fused_values,
            rtol=rtol,
            atol=atol,
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source singleton-batch output "
                "does not match fused batchless output"
            )
        if int(np.argmax(source_batchless.reshape(-1))) != int(
            np.argmax(fused_values.reshape(-1))
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: Argmax/class_id changed across "
                "Conv2d/BatchNorm2d fusion"
            )


def _run_concrete_probe(
    model: object,
    probe_input: object,
    label: str,
    nn: object,
    torch: object,
    *,
    copy_model: bool = True,
) -> object:
    probe_model = _copy_model_for_tracing(model, nn, torch) if copy_model else model
    before_state = _snapshot_module_state(probe_model, nn, torch)
    class_state = _snapshot_module_class_definitions(probe_model, nn)
    probe_error: Exception | None = None
    output: object | None = None
    try:
        with torch.no_grad():
            output = probe_model(probe_input)
    except Exception as exc:
        probe_error = exc
    class_changed = _restore_module_class_definitions(class_state)
    instance_changed = before_state != _snapshot_module_state(
        probe_model,
        nn,
        torch,
    )
    if class_changed or instance_changed:
        raise UnsupportedOpError(
            f"Unsupported Python state mutation during {label} semantic probe"
        )
    if probe_error is not None:
        raise UnsupportedOpError(
            f"Unsupported {label} model semantics: {probe_error}"
        ) from probe_error
    return output


def _validate_lowered_semantics(
    model: object,
    graph: GraphIR,
    input_shape: tuple[int, ...],
    float_dtype: str,
    nn: object,
    torch: object,
    *,
    copy_model: bool = True,
) -> None:
    from torch2rtl.quant.reference import infer_float_graph

    dtype = torch.float32 if float_dtype == "float32" else torch.float64
    numpy_dtype = np.dtype(float_dtype)
    element_count = math.prod(input_shape)
    probe_values = (
        np.zeros(input_shape, dtype=numpy_dtype),
        np.linspace(-0.75, 0.75, num=element_count, dtype=numpy_dtype).reshape(
            input_shape
        ),
    )
    for values in probe_values:
        probe_model = (
            _copy_model_for_tracing(model, nn, torch) if copy_model else model
        )
        before_state = _snapshot_module_state(probe_model, nn, torch)
        class_state = _snapshot_module_class_definitions(probe_model, nn)
        probe_error: Exception | None = None
        actual: object | None = None
        probe_input = torch.from_numpy(values.copy()).to(dtype=dtype)
        try:
            with torch.no_grad():
                actual = probe_model(probe_input)
        except Exception as exc:
            probe_error = exc
        class_changed = _restore_module_class_definitions(class_state)
        instance_changed = before_state != _snapshot_module_state(
            probe_model,
            nn,
            torch,
        )
        if class_changed or instance_changed:
            raise UnsupportedOpError(
                "Unsupported Python state mutation during concrete semantic probe"
            )
        if probe_error is not None:
            raise UnsupportedOpError(
                f"Unsupported concrete model semantics: {probe_error}"
            ) from probe_error
        if not isinstance(actual, torch.Tensor):
            raise UnsupportedOpError(
                "Unsupported model output: concrete forward must return one Tensor"
            )

        expected_dtype = torch.int64 if graph.output.dtype == "int64" else dtype
        if actual.dtype != expected_dtype:
            raise UnsupportedOpError(
                "Unsupported model output dtype: concrete forward produced "
                f"{actual.dtype}, GraphIR requires {expected_dtype}"
            )
        actual_values = actual.detach().cpu().numpy()
        expected_values = infer_float_graph(graph, values).output
        if actual_values.shape != expected_values.shape or not np.array_equal(
            actual_values,
            expected_values,
            equal_nan=True,
        ):
            raise UnsupportedOpError(
                "Unsupported model semantics: concrete forward does not match lowered GraphIR"
            )


def _normalize_float_dtype(
    value: object | None,
    torch: object,
    supported: dict[object, str],
) -> object | None:
    if value is None:
        return None
    if value is torch.float32 or value is np.dtype(np.float32):
        return torch.float32
    if value is torch.float64 or value is np.dtype(np.float64):
        return torch.float64
    if type(value) is str:
        if value in {"float32", "torch.float32"}:
            return torch.float32
        if value in {"float64", "torch.float64"}:
            return torch.float64
    raise UnsupportedOpError(
        "Unsupported input dtype; only float32 and float64 are supported"
    )


def _validate_input_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    if type(input_shape) in (str, bytes, bytearray) or not isinstance(
        input_shape,
        SequenceABC,
    ):
        raise UnsupportedOpError(
            "Unsupported input shape: expected a finite sequence of positive integers"
        )
    try:
        shape = tuple(input_shape)
    except Exception as exc:
        raise UnsupportedOpError(
            "Unsupported input shape: expected a finite sequence of positive integers"
        ) from exc
    if len(shape) > _MAX_TENSOR_RANK:
        raise UnsupportedOpError(
            f"Unsupported input shape: rank must not exceed {_MAX_TENSOR_RANK}"
        )
    for index, dim in enumerate(shape):
        if type(dim) is not int or dim <= 0:
            raise UnsupportedOpError(
                "Unsupported input shape: positive integers are required; "
                f"dimension at index {index} is not a positive built-in int"
            )
    _validate_tensor_element_count(shape, "input shape")
    return shape


def _validate_tensor_element_count(shape: Sequence[int], label: str) -> None:
    element_count = 1
    for dimension in shape:
        if dimension > _MAX_TENSOR_ELEMENTS // element_count:
            raise UnsupportedOpError(
                f"Unsupported {label}: total element count exceeds "
                f"the v0.2 limit of {_MAX_TENSOR_ELEMENTS}"
            )
        element_count *= dimension


def _parse_output_node(node: object) -> object:
    args = tuple(getattr(node, "args", ()))
    kwargs = dict(getattr(node, "kwargs", {}))
    if len(args) != 1 or kwargs:
        raise UnsupportedOpError(
            "Unsupported FX output: expected one tensor value without keyword arguments"
        )
    output_value = args[0]
    if not _is_fx_node(output_value):
        raise UnsupportedOpError(
            "Unsupported FX output: tuples, lists, dictionaries, and constants "
            "are not supported"
        )
    return output_value


def _require_sequential_input(
    node: object,
    expected_node: object,
    allow_parameters: bool = False,
) -> None:
    args = tuple(getattr(node, "args", ()))
    kwargs = dict(getattr(node, "kwargs", {}))
    if not args or not _is_fx_node(args[0]):
        raise UnsupportedOpError(
            f"Unsupported FX arguments for node {getattr(node, 'name', '<unknown>')}: "
            "the first argument must be the previous tensor"
        )
    if args[0] is not expected_node:
        raise UnsupportedOpError(
            f"Unsupported FX graph at node {getattr(node, 'name', '<unknown>')}: "
            "only a single sequential dependency chain is supported"
        )
    if not allow_parameters and (len(args) != 1 or kwargs):
        raise UnsupportedOpError(
            f"Unsupported FX arguments for node {getattr(node, 'name', '<unknown>')}: "
            "module calls must receive only the previous tensor"
        )


def _validate_argmax_arguments(node: object, is_method: bool) -> None:
    args = tuple(getattr(node, "args", ()))
    kwargs = dict(getattr(node, "kwargs", {}))
    positional = args[1:]
    if len(positional) > 2:
        raise UnsupportedOpError("Unsupported argmax arguments")

    allowed_kwargs = {"dim", "keepdim"}
    if not is_method:
        allowed_kwargs.add("out")
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        names = ", ".join(sorted(str(name) for name in unexpected))
        raise UnsupportedOpError(f"Unsupported argmax keyword arguments: {names}")

    dim = _positional_or_keyword(positional, 0, kwargs, "dim", None)
    keepdim = _positional_or_keyword(positional, 1, kwargs, "keepdim", False)
    if not is_method and kwargs.get("out") is not None:
        raise UnsupportedOpError("Unsupported argmax out argument")
    if dim is not None or keepdim is not False:
        raise UnsupportedOpError(
            "Unsupported argmax: only global argmax with dim=None and "
            "keepdim=False is supported"
        )


def _positional_or_keyword(
    positional: tuple[object, ...],
    index: int,
    kwargs: dict[object, object],
    name: str,
    default: object,
) -> object:
    if index < len(positional):
        if name in kwargs:
            raise UnsupportedOpError(f"argmax received {name} more than once")
        return positional[index]
    return kwargs.get(name, default)


def _is_fx_node(value: object) -> bool:
    return (
        hasattr(value, "op")
        and hasattr(value, "target")
        and hasattr(value, "args")
        and hasattr(value, "kwargs")
    )


def _conv_output_dim(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


def _int_pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        raise UnsupportedOpError(f"Unsupported Conv2d {name}: string values are not supported")
    if type(value) is int:
        return (value, value)
    if type(value) is tuple and len(value) == 2:
        first, second = value
        if type(first) is int and type(second) is int:
            return (first, second)
    raise UnsupportedOpError(f"Unsupported Conv2d {name}: expected int or int pair")


def _positive_module_integer(value: object, module: str, attribute: str) -> int:
    normalized = _module_integer(value, module, attribute)
    if normalized <= 0:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a positive built-in integer, "
            f"got {value!r}"
        )
    return normalized


def _module_integer(value: object, module: str, attribute: str) -> int:
    if type(value) is not int:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a built-in integer, "
            f"got {value!r}"
        )
    return value


def _required_parameter(
    value: object,
    module: str,
    attribute: str,
    nn: object,
) -> object:
    if type(value) is not nn.Parameter:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a Parameter"
        )
    return value


def _required_module_attribute(
    module_value: object,
    module: str,
    attribute: str,
) -> object:
    try:
        return getattr(module_value, attribute)
    except AttributeError as exc:
        raise UnsupportedOpError(
            f"Unsupported {module}: required attribute {attribute!r} is missing"
        ) from exc


def _optional_parameter(
    value: object,
    module: str,
    attribute: str,
    nn: object,
) -> object | None:
    if value is None:
        return None
    return _required_parameter(value, module, attribute, nn)


def _validate_conv_pair(
    name: str,
    values: tuple[int, int],
    minimum: int,
) -> None:
    if any(value < minimum for value in values):
        relation = "positive" if minimum == 1 else "non-negative"
        raise UnsupportedOpError(
            f"Unsupported Conv2d {name}: values must be {relation}, got {values}"
        )


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


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"op_{cleaned}"
    return cleaned or "op"
