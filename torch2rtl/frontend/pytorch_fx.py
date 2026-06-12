from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np

from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.ir.tensor import TensorIR


class UnsupportedOpError(RuntimeError):
    """Raised when the FX graph contains an operation outside the MVP subset."""


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


def parse_model(model: object, input_shape: Sequence[int]) -> GraphIR:
    import torch
    import torch.nn as nn

    traced = torch.fx.symbolic_trace(model)
    modules = dict(traced.named_modules())
    input_tensor = TensorIR(name="input", shape=tuple(input_shape), dtype="float")
    current_tensor = input_tensor
    ops: list[object] = []
    saw_argmax = False

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            continue
        if node.op == "call_module":
            module = modules[str(node.target)]
            current_tensor, new_op = _parse_module_node(
                node_name=_safe_name(str(node.name)),
                module=module,
                current_tensor=current_tensor,
                nn=nn,
            )
            ops.append(new_op)
            continue
        if node.op == "call_function" and node.target is torch.argmax:
            argmax_input = current_tensor
            current_tensor = TensorIR(name=_safe_name(str(node.name)), shape=(), dtype="uint")
            ops.append(
                ArgmaxIR(
                    name=_safe_name(str(node.name)),
                    input=argmax_input,
                    output=current_tensor,
                )
            )
            saw_argmax = True
            continue
        if node.op == "call_method" and str(node.target) == "argmax":
            argmax_input = current_tensor
            current_tensor = TensorIR(name=_safe_name(str(node.name)), shape=(), dtype="uint")
            ops.append(
                ArgmaxIR(
                    name=_safe_name(str(node.name)),
                    input=argmax_input,
                    output=current_tensor,
                )
            )
            saw_argmax = True
            continue
        raise UnsupportedOpError(
            f"Unsupported FX node/op/module: op={node.op}, target={node.target}"
        )

    if not saw_argmax:
        argmax_input = current_tensor
        current_tensor = TensorIR(name="class_id", shape=(), dtype="uint")
        ops.append(ArgmaxIR(name="argmax", input=argmax_input, output=current_tensor))

    return GraphIR(
        input=input_tensor,
        output=current_tensor,
        ops=tuple(ops),
        metadata={"source": type(model).__name__},
    )


def _parse_module_node(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
    nn: object,
) -> tuple[TensorIR, object]:
    if isinstance(module, nn.Linear):
        if not current_tensor.shape:
            raise UnsupportedOpError("Unsupported Linear input: scalar tensor")
        in_features = int(current_tensor.shape[-1])
        if in_features != int(module.in_features):
            raise UnsupportedOpError(
                "Unsupported Linear shape: "
                f"expected last dim {module.in_features}, got {in_features}"
            )
        prefix_shape = current_tensor.shape[:-1]
        output_shape = (*prefix_shape, int(module.out_features))
        output = TensorIR(name=f"{node_name}_out", shape=output_shape, dtype="float")
        bias = None
        if module.bias is not None:
            bias = module.bias.detach().cpu().numpy().astype(np.float64)
        op = LinearIR(
            name=node_name,
            input=current_tensor,
            output=output,
            in_features=int(module.in_features),
            out_features=int(module.out_features),
            weight=module.weight.detach().cpu().numpy().astype(np.float64),
            bias=bias,
        )
        return output, op

    if isinstance(module, nn.ReLU):
        output = TensorIR(name=f"{node_name}_out", shape=current_tensor.shape, dtype="float")
        return output, ReluIR(name=node_name, input=current_tensor, output=output)

    if isinstance(module, nn.Flatten):
        output_shape = _flatten_shape(
            shape=current_tensor.shape,
            start_dim=int(module.start_dim),
            end_dim=int(module.end_dim),
        )
        output = TensorIR(name=f"{node_name}_out", shape=output_shape, dtype="float")
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
