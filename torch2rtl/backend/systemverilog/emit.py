from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from torch2rtl.eda_tools import detect_eda_tools
from torch2rtl.ir.graph import GraphIR
from torch2rtl.quant.fixed_point import FixedPointConfig, dequantize_array
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    QuantizedConv2dIR,
    QuantizedFlattenIR,
    QuantizedGraph,
    QuantizedLinearIR,
    QuantizedReluIR,
    infer_float_graph,
    infer_quantized,
    quantize_graph,
)
from torch2rtl.visualization.manifest import HTML_NAME, MANIFEST_NAME, write_visualization_artifacts
from torch2rtl.verify.vectors import write_vector_files


@dataclass(frozen=True)
class BuildReport:
    out_dir: Path
    generated_files: tuple[str, ...]
    report_path: Path


@dataclass(frozen=True)
class LayerRenderInfo:
    kind: str
    name: str
    input_signal: str
    output_signal: str
    in_features: int
    out_features: int
    weights_values: tuple[str, ...] = ()
    biases_values: tuple[str, ...] = ()
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    kernel_size: tuple[int, int] = (0, 0)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)


def emit_systemverilog(
    graph: GraphIR,
    cfg: FixedPointConfig,
    out_dir: Path,
    vector_count: int = 16,
    seed: int = 0,
) -> BuildReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(out_dir)
    qgraph = quantize_graph(graph, cfg)
    vector_files = write_vector_files(out_dir, qgraph, cfg, vector_count, seed)

    env = _template_env()
    module_templates = {
        "conv2d_comb.sv": "conv2d_comb.sv.j2",
        "linear_comb.sv": "linear_comb.sv.j2",
        "relu.sv": "relu.sv.j2",
        "argmax.sv": "argmax.sv.j2",
        "tb_top.sv": "tb_top.sv.j2",
    }
    generated: list[str] = []
    for output_name, template_name in module_templates.items():
        content = env.get_template(template_name).render(**_common_context(qgraph))
        (out_dir / output_name).write_text(content, encoding="utf-8")
        generated.append(output_name)

    top_context = _top_context(qgraph)
    top_content = env.get_template("top.sv.j2").render(**top_context)
    (out_dir / "top.sv").write_text(top_content, encoding="utf-8")
    generated.append("top.sv")
    generated.extend(vector_files)
    generated.append("report.json")
    generated.extend([MANIFEST_NAME, HTML_NAME])

    report = _report_payload(graph, qgraph, cfg, out_dir, generated, vector_count)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_visualization_artifacts(
        graph=graph,
        qgraph=qgraph,
        build_dir=out_dir,
        generated_files=generated,
        vector_count=vector_count,
    )
    return BuildReport(
        out_dir=out_dir,
        generated_files=tuple(sorted(generated)),
        report_path=report_path,
    )


def _template_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(disabled_extensions=("sv", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _clear_previous_outputs(out_dir: Path) -> None:
    exact_names = {
        "argmax.sv",
        "conv2d_comb.sv",
        "expected_classes.txt",
        "input_vectors.txt",
        "linear_comb.sv",
        "relu.sv",
        "report.json",
        "simv",
        "tb_top.sv",
        "top.sv",
        "vectors.json",
        "visualization.html",
        "visualization.json",
        "yosys.log",
    }
    for name in exact_names:
        path = out_dir / name
        if path.is_file():
            path.unlink()

    for pattern in ("*_weights.mem", "*_bias.mem"):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _common_context(qgraph: QuantizedGraph) -> dict[str, Any]:
    class_count = _class_count(qgraph)
    return {
        "data_bits": qgraph.cfg.bits,
        "frac_bits": qgraph.cfg.frac_bits,
        "acc_bits": qgraph.cfg.acc_bits,
        "input_size": qgraph.input_size,
        "class_count": class_count,
        "class_bits": _ceil_log2(class_count),
    }


def _top_context(qgraph: QuantizedGraph) -> dict[str, Any]:
    layers: list[LayerRenderInfo] = []
    signals: list[LayerRenderInfo] = []
    current_signal = "in_data"
    current_size = qgraph.input_size
    logits_signal = current_signal

    for idx, op in enumerate(qgraph.ops):
        name = _safe_sv_name(f"{op.name}_{idx}")
        if isinstance(op, QuantizedLinearIR):
            output_signal = f"{name}_out"
            layer = LayerRenderInfo(
                kind="linear",
                name=name,
                input_signal=current_signal,
                output_signal=output_signal,
                in_features=op.in_features,
                out_features=op.out_features,
                weights_values=_sv_array_values(op.weight.reshape(-1), qgraph.cfg.bits),
                biases_values=_sv_array_values(op.bias.reshape(-1), qgraph.cfg.bits),
            )
            layers.append(layer)
            signals.append(layer)
            current_signal = output_signal
            current_size = op.out_features
            logits_signal = current_signal
        elif isinstance(op, QuantizedConv2dIR):
            output_signal = f"{name}_out"
            layer = LayerRenderInfo(
                kind="conv2d",
                name=name,
                input_signal=current_signal,
                output_signal=output_signal,
                in_features=op.input_size,
                out_features=op.output_size,
                weights_values=_sv_array_values(op.weight.reshape(-1), qgraph.cfg.bits),
                biases_values=_sv_array_values(op.bias.reshape(-1), qgraph.cfg.bits),
                input_shape=(op.in_channels, op.input_height, op.input_width),
                output_shape=(op.out_channels, op.output_height, op.output_width),
                kernel_size=(op.kernel_height, op.kernel_width),
                stride=op.stride,
                padding=op.padding,
            )
            layers.append(layer)
            signals.append(layer)
            current_signal = output_signal
            current_size = op.output_size
            logits_signal = current_signal
        elif isinstance(op, QuantizedReluIR):
            output_signal = f"{name}_out"
            layer = LayerRenderInfo(
                kind="relu",
                name=name,
                input_signal=current_signal,
                output_signal=output_signal,
                in_features=current_size,
                out_features=current_size,
            )
            layers.append(layer)
            signals.append(layer)
            current_signal = output_signal
            logits_signal = current_signal
        elif isinstance(op, QuantizedFlattenIR):
            current_size = qgraph.input_size if current_signal == "in_data" else current_size
        elif isinstance(op, QuantizedArgmaxIR):
            layer = LayerRenderInfo(
                kind="argmax",
                name=name,
                input_signal=current_signal,
                output_signal="class_id",
                in_features=current_size,
                out_features=1,
            )
            layers.append(layer)
        else:
            raise TypeError(f"Unsupported quantized op for SV: {type(op).__name__}")

    context = _common_context(qgraph)
    context.update(
        {
            "layers": layers,
            "signals": signals,
            "logits_signal": logits_signal,
            "logits_size": current_size,
        }
    )
    return context


def _report_payload(
    graph: GraphIR,
    qgraph: QuantizedGraph,
    cfg: FixedPointConfig,
    build_dir: Path,
    generated: list[str],
    vector_count: int,
) -> dict[str, Any]:
    return {
        "tool": "torch2rtl",
        "version": "0.2.0",
        "quant": asdict(cfg),
        "graph": {
            "input_shape": list(graph.input.shape),
            "output_shape": list(graph.output.shape),
            "ops": [type(op).__name__ for op in graph.ops],
        },
        "metrics": _metrics_payload(qgraph),
        "reference": _reference_payload(graph, qgraph, cfg, build_dir, vector_count),
        "vectors": {"count": vector_count},
        "generated_files": sorted(generated),
        "tools": detect_eda_tools(),
        "status": {},
    }


def _class_count(qgraph: QuantizedGraph) -> int:
    last_size = qgraph.input_size
    for op in qgraph.ops:
        if isinstance(op, QuantizedLinearIR):
            last_size = op.out_features
        elif isinstance(op, QuantizedConv2dIR):
            last_size = op.output_size
    return last_size


def _metrics_payload(qgraph: QuantizedGraph) -> dict[str, Any]:
    total_parameters = 0
    total_macs = 0
    ops: list[dict[str, Any]] = []
    current_size = qgraph.input_size
    for op in qgraph.ops:
        if isinstance(op, QuantizedLinearIR):
            parameters = int(op.weight.size + op.bias.size)
            macs = int(op.in_features * op.out_features)
            current_size = op.out_features
            shape = [op.out_features]
        elif isinstance(op, QuantizedConv2dIR):
            parameters = op.parameter_count
            macs = op.mac_count
            current_size = op.output_size
            shape = [op.out_channels, op.output_height, op.output_width]
        elif isinstance(op, QuantizedReluIR):
            parameters = 0
            macs = 0
            shape = [current_size]
        elif isinstance(op, QuantizedFlattenIR):
            parameters = 0
            macs = 0
            shape = [current_size]
        elif isinstance(op, QuantizedArgmaxIR):
            parameters = 0
            macs = 0
            shape = []
        else:
            raise TypeError(f"Unsupported quantized op for metrics: {type(op).__name__}")
        total_parameters += parameters
        total_macs += macs
        ops.append(
            {
                "name": op.name,
                "kind": type(op).__name__,
                "parameters": parameters,
                "macs": macs,
                "activation_size": current_size,
                "activation_shape": shape,
            }
        )
    return {
        "parameters": total_parameters,
        "macs": total_macs,
        "activation_values": [qgraph.input_size, *[op["activation_size"] for op in ops]],
        "ops": ops,
    }


def _reference_payload(
    graph: GraphIR,
    qgraph: QuantizedGraph,
    cfg: FixedPointConfig,
    build_dir: Path,
    vector_count: int,
) -> dict[str, Any]:
    input_path = build_dir / "input_vectors.txt"
    if not input_path.exists():
        return {"kind": "fixed_vs_float_ir", "status": "missing_vectors"}
    inputs = _load_input_vectors(input_path, qgraph.input_size)
    class_matches = 0
    abs_errors: list[float] = []
    for row in inputs:
        fixed = infer_quantized(qgraph, row)
        float_input = dequantize_array(np.asarray(row, dtype=np.int64), cfg)
        floating = infer_float_graph(graph, float_input.reshape(qgraph.input_shape))
        if fixed.class_id == floating.class_id:
            class_matches += 1
        fixed_logits = dequantize_array(fixed.logits.reshape(-1), cfg)
        float_logits = floating.logits.reshape(-1)
        if fixed_logits.size == float_logits.size:
            abs_errors.extend(np.abs(fixed_logits - float_logits).astype(float).tolist())
    error_array = np.asarray(abs_errors, dtype=np.float64)
    return {
        "kind": "fixed_vs_float_ir",
        "source": "PyTorch weights lowered through GraphIR",
        "vectors": min(vector_count, int(len(inputs))),
        "class_matches": class_matches,
        "class_mismatches": int(len(inputs) - class_matches),
        "mean_abs_logit_error": round(float(error_array.mean()), 6)
        if error_array.size
        else 0.0,
        "max_abs_logit_error": round(float(error_array.max()), 6)
        if error_array.size
        else 0.0,
    }


def _load_input_vectors(path: Path, input_size: int) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    data = np.asarray(values, dtype=np.int64)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        if input_size == 1:
            return data.reshape(-1, 1)
        return data.reshape(1, input_size)
    return data


def _ceil_log2(value: int) -> int:
    return max(1, math.ceil(math.log2(max(value, 1))))


def _sv_array_values(values: np.ndarray, bits: int) -> tuple[str, ...]:
    return tuple(_sv_signed_literal(int(value), bits) for value in values.reshape(-1))


def _sv_signed_literal(value: int, bits: int) -> str:
    if value < 0:
        return f"-{bits}'sd{abs(value)}"
    return f"{bits}'sd{value}"


def _safe_sv_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"op_{cleaned}"
    return cleaned or "op"
