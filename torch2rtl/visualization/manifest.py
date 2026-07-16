from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from torch2rtl.ir.graph import GraphIR
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    QuantizedConv2dIR,
    QuantizedFlattenIR,
    QuantizedGraph,
    QuantizedLinearIR,
    QuantizedOp,
    QuantizedReluIR,
)
from torch2rtl.visualization.html import render_visualization_html
from torch2rtl.visualization.trace import (
    array_stats,
    qgraph_from_manifest,
    trace_collection_payload,
    trace_payload,
)

MANIFEST_NAME = "visualization.json"
HTML_NAME = "visualization.html"
SOURCE_PREVIEW_MAX_LINES = 200
SOURCE_PREVIEW_MAX_CHARS = 20_000
SOURCE_PREVIEW_CHUNK_SIZE = 8_192


def write_visualization_artifacts(
    graph: GraphIR,
    qgraph: QuantizedGraph,
    build_dir: Path,
    generated_files: Sequence[str],
    vector_count: int,
    vector_index: int = 0,
) -> tuple[str, str]:
    manifest = build_visualization_manifest(
        graph=graph,
        qgraph=qgraph,
        build_dir=build_dir,
        generated_files=generated_files,
        vector_count=vector_count,
        vector_index=vector_index,
    )
    manifest_path = build_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    html_path = build_dir / HTML_NAME
    html_path.write_text(render_visualization_html(manifest), encoding="utf-8")
    return (MANIFEST_NAME, HTML_NAME)


def render_visualization_from_build(
    build_dir: Path,
    out_path: Path | None = None,
    vector_index: int = 0,
) -> Path:
    manifest_path = build_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{MANIFEST_NAME} not found in {build_dir}; run torch2rtl compile first"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qgraph = qgraph_from_manifest(manifest, build_dir)
    manifest["trace"] = trace_payload(qgraph, manifest, build_dir, vector_index)
    manifest["traces"] = trace_collection_payload(
        qgraph,
        manifest,
        build_dir,
        selected_index=vector_index,
    )
    manifest["build"] = _build_payload(
        build_dir=build_dir,
        generated_files=manifest.get("build", {}).get("generated_files", []),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output = out_path if out_path is not None else build_dir / HTML_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_visualization_html(manifest), encoding="utf-8")
    return output


def build_visualization_manifest(
    graph: GraphIR,
    qgraph: QuantizedGraph,
    build_dir: Path,
    generated_files: Sequence[str],
    vector_count: int,
    vector_index: int = 0,
) -> dict[str, Any]:
    blocks, connections, ops = _circuit_payload(qgraph)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tool": "torch2rtl",
        "title": "Обозреватель схемы Torch2RTL",
        "quant": asdict(qgraph.cfg),
        "graph": {
            "input_shape": list(graph.input.shape),
            "output_shape": list(graph.output.shape),
            "ops": [type(op).__name__ for op in graph.ops],
        },
        "build": _build_payload(build_dir, generated_files),
        "vectors": {
            "count": vector_count,
            "selected_index": vector_index,
            "input_file": "input_vectors.txt",
            "expected_file": "expected_classes.txt",
        },
        "blocks": blocks,
        "connections": connections,
        "ops": ops,
    }
    manifest["trace"] = trace_payload(qgraph, manifest, build_dir, vector_index)
    manifest["traces"] = trace_collection_payload(
        qgraph,
        manifest,
        build_dir,
        selected_index=vector_index,
    )
    return manifest


def _circuit_payload(
    qgraph: QuantizedGraph,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = [
        {
            "id": "input",
            "kind": "input",
            "lane": "main",
            "label": "Шина входа",
            "name": "in_data",
            "signal_out": "in_data",
            "features": qgraph.input_size,
            "shape": list(qgraph.input_shape),
            "file": "input_vectors.txt",
        }
    ]
    connections: list[dict[str, Any]] = []
    ops: list[dict[str, Any]] = []
    previous_id = "input"
    current_signal = "in_data"
    current_size = qgraph.input_size

    for idx, op in enumerate(qgraph.ops):
        layer_id = _safe_id(f"{op.name}_{idx}")
        if isinstance(op, QuantizedLinearIR):
            output_signal = f"{layer_id}_out"
            block = _main_block(
                block_id=layer_id,
                kind="linear",
                label="Linear / MAC",
                name=layer_id,
                input_signal=current_signal,
                output_signal=output_signal,
                in_features=op.in_features,
                out_features=op.out_features,
                params={
                    "weights": op.in_features * op.out_features,
                    "biases": op.out_features,
                    "macs": op.in_features * op.out_features,
                },
            )
            blocks.append(block)
            blocks.extend(_memory_blocks(layer_id, op))
            connections.append(_data_connection(previous_id, layer_id, current_signal))
            connections.extend(_memory_connections(layer_id))
            ops.append(_op_record(block, op.name, idx, op))
            previous_id = layer_id
            current_signal = output_signal
            current_size = op.out_features
        elif isinstance(op, QuantizedConv2dIR):
            output_signal = f"{layer_id}_out"
            block = _main_block(
                block_id=layer_id,
                kind="conv2d",
                label="Conv2d / MAC",
                name=layer_id,
                input_signal=current_signal,
                output_signal=output_signal,
                in_features=op.input_size,
                out_features=op.output_size,
                params={
                    "weights": int(op.weight.size),
                    "biases": int(op.bias.size),
                    "macs": op.mac_count,
                    "in_channels": op.in_channels,
                    "out_channels": op.out_channels,
                    "kernel_size": f"{op.kernel_height}x{op.kernel_width}",
                    "stride": f"{op.stride[0]}x{op.stride[1]}",
                    "padding": f"{op.padding[0]}x{op.padding[1]}",
                    "output_shape": (
                        f"{op.out_channels}x{op.output_height}x{op.output_width}"
                    ),
                },
            )
            blocks.append(block)
            blocks.extend(_memory_blocks(layer_id, op))
            connections.append(_data_connection(previous_id, layer_id, current_signal))
            connections.extend(_memory_connections(layer_id))
            ops.append(_op_record(block, op.name, idx, op))
            previous_id = layer_id
            current_signal = output_signal
            current_size = op.output_size
        elif isinstance(op, QuantizedReluIR):
            output_signal = f"{layer_id}_out"
            block = _main_block(
                block_id=layer_id,
                kind="relu",
                label="ReLU: отсечка ниже 0",
                name=layer_id,
                input_signal=current_signal,
                output_signal=output_signal,
                in_features=current_size,
                out_features=current_size,
                params={"threshold": 0},
            )
            blocks.append(block)
            connections.append(_data_connection(previous_id, layer_id, current_signal))
            ops.append(_op_record(block, op.name, idx, op))
            previous_id = layer_id
            current_signal = output_signal
        elif isinstance(op, QuantizedFlattenIR):
            block = _main_block(
                block_id=layer_id,
                kind="flatten",
                label="Flatten: проводная развёртка",
                name=layer_id,
                input_signal=current_signal,
                output_signal=current_signal,
                in_features=current_size,
                out_features=current_size,
                params={"hardware": "wire-only"},
            )
            blocks.append(block)
            connections.append(_data_connection(previous_id, layer_id, current_signal))
            ops.append(_op_record(block, op.name, idx, op))
            previous_id = layer_id
        elif isinstance(op, QuantizedArgmaxIR):
            block = _main_block(
                block_id=layer_id,
                kind="argmax",
                label="Argmax",
                name=layer_id,
                input_signal=current_signal,
                output_signal="class_id",
                in_features=current_size,
                out_features=1,
                params={"classes": current_size},
            )
            blocks.append(block)
            connections.append(_data_connection(previous_id, layer_id, current_signal))
            ops.append(_op_record(block, op.name, idx, op))
            previous_id = layer_id
        else:
            raise TypeError(f"Unsupported quantized op for visualization: {type(op).__name__}")

    blocks.append(
        {
            "id": "output",
            "kind": "output",
            "lane": "main",
            "label": "Выход класса",
            "name": "class_id",
            "signal_in": "class_id",
            "features": 1,
            "shape": [],
            "file": "expected_classes.txt",
        }
    )
    connections.append(_data_connection(previous_id, "output", "class_id"))
    return blocks, connections, ops


def _main_block(
    block_id: str,
    kind: str,
    label: str,
    name: str,
    input_signal: str,
    output_signal: str,
    in_features: int,
    out_features: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": block_id,
        "kind": kind,
        "lane": "main",
        "label": label,
        "name": name,
        "signal_in": input_signal,
        "signal_out": output_signal,
        "in_features": in_features,
        "out_features": out_features,
        "shape": [in_features, out_features],
        "params": params,
    }


def _memory_blocks(
    layer_id: str,
    op: QuantizedLinearIR | QuantizedConv2dIR,
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{layer_id}_weights",
            "kind": "memory",
            "lane": "memory",
            "label": "веса .mem",
            "name": f"{op.name}_weights.mem",
            "parent": layer_id,
            "features": int(op.weight.size),
            "file": f"{op.name}_weights.mem",
            "stats": array_stats(op.weight),
        },
        {
            "id": f"{layer_id}_bias",
            "kind": "memory",
            "lane": "memory",
            "label": "смещения .mem",
            "name": f"{op.name}_bias.mem",
            "parent": layer_id,
            "features": int(op.bias.size),
            "file": f"{op.name}_bias.mem",
            "stats": array_stats(op.bias),
        },
    ]


def _memory_connections(layer_id: str) -> list[dict[str, Any]]:
    return [
        {"from": f"{layer_id}_weights", "to": layer_id, "signal": "weights", "kind": "param"},
        {"from": f"{layer_id}_bias", "to": layer_id, "signal": "biases", "kind": "param"},
    ]


def _data_connection(source: str, target: str, signal: str) -> dict[str, Any]:
    return {"from": source, "to": target, "signal": signal, "kind": "data"}


def _op_record(
    block: dict[str, Any],
    source_name: str,
    index: int,
    op: QuantizedOp,
) -> dict[str, Any]:
    record = {
        "id": block["id"],
        "kind": block["kind"],
        "source_name": source_name,
        "index": index,
        "in_features": block["in_features"],
        "out_features": block["out_features"],
    }
    if isinstance(op, QuantizedLinearIR):
        record.update(
            {
                "weight_file": f"{source_name}_weights.mem",
                "bias_file": f"{source_name}_bias.mem",
            }
        )
    elif isinstance(op, QuantizedConv2dIR):
        record.update(
            {
                "in_channels": op.in_channels,
                "out_channels": op.out_channels,
                "input_height": op.input_height,
                "input_width": op.input_width,
                "output_height": op.output_height,
                "output_width": op.output_width,
                "kernel_height": op.kernel_height,
                "kernel_width": op.kernel_width,
                "stride": list(op.stride),
                "padding": list(op.padding),
                "weight_file": f"{source_name}_weights.mem",
                "bias_file": f"{source_name}_bias.mem",
            }
        )
    return record


def _build_payload(build_dir: Path, generated_files: Sequence[str]) -> dict[str, Any]:
    report_path = build_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_generated_files = report.get("generated_files", list(generated_files))
        return {
            "generated_files": report_generated_files,
            "tools": report.get("tools", {}),
            "environment": report.get("environment", {}),
            "status": report.get("status", {}),
            "demo": report.get("demo", {}),
            "metrics": report.get("metrics", {}),
            "reference": report.get("reference", {}),
            "simulation": report.get("simulation", {}),
            "synthesis": report.get("synthesis", {}),
            "artifacts": _artifact_payload(build_dir, report_generated_files),
            "source_previews": _source_preview_payload(
                build_dir, report_generated_files
            ),
        }
    return {
        "generated_files": list(generated_files),
        "tools": {},
        "status": {},
        "source_previews": _source_preview_payload(build_dir, generated_files),
    }


def _source_preview_payload(
    build_dir: Path, generated_files: Sequence[str]
) -> dict[str, dict[str, Any]]:
    build_root = build_dir.resolve()
    previews: dict[str, dict[str, Any]] = {}
    for file_name in dict.fromkeys(str(name) for name in generated_files):
        relative_path = Path(file_name)
        if relative_path.is_absolute() or relative_path.suffix.lower() != ".sv":
            continue

        source_path = (build_root / relative_path).resolve()
        if not source_path.is_relative_to(build_root) or not source_path.is_file():
            continue

        try:
            previews[file_name] = _read_source_preview(source_path)
        except OSError:
            continue
    return previews


def _read_source_preview(source_path: Path) -> dict[str, Any]:
    preview_parts: list[str] = []
    preview_chars = 0
    preview_newlines = 0
    total_chars = 0
    total_newlines = 0
    last_char = ""

    with source_path.open(encoding="utf-8", errors="replace") as source:
        while chunk := source.read(SOURCE_PREVIEW_CHUNK_SIZE):
            total_chars += len(chunk)
            total_newlines += chunk.count("\n")
            last_char = chunk[-1]

            if (
                preview_chars >= SOURCE_PREVIEW_MAX_CHARS
                or preview_newlines >= SOURCE_PREVIEW_MAX_LINES
            ):
                continue

            candidate = chunk[: SOURCE_PREVIEW_MAX_CHARS - preview_chars]
            newline_budget = SOURCE_PREVIEW_MAX_LINES - preview_newlines
            if candidate.count("\n") >= newline_budget:
                end = 0
                for _ in range(newline_budget):
                    end = candidate.find("\n", end) + 1
                candidate = candidate[:end]

            preview_parts.append(candidate)
            preview_chars += len(candidate)
            preview_newlines += candidate.count("\n")

    text = "".join(preview_parts)
    line_count = total_newlines + int(total_chars > 0 and last_char != "\n")
    shown_lines = text.count("\n") + int(bool(text) and not text.endswith("\n"))
    return {
        "text": text,
        "line_count": line_count,
        "shown_lines": shown_lines,
        "truncated": total_chars > len(text),
    }


def _artifact_payload(build_dir: Path, generated_files: Sequence[str]) -> list[dict[str, Any]]:
    names = list(dict.fromkeys([*generated_files, "yosys.log"]))
    return [
        {"file": name, "size_bytes": int((build_dir / name).stat().st_size)}
        for name in names
        if (build_dir / name).is_file()
    ]


def _safe_id(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        return f"op_{cleaned}"
    return cleaned or "op"
