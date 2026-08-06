from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array
from torch2rtl.quant.reference import QuantizedGraph, infer_quantized


@dataclass(frozen=True)
class _VerificationVectors:
    inputs: np.ndarray
    logits: np.ndarray
    classes: np.ndarray
    source: str
    seed: int | None

    @property
    def count(self) -> int:
        return int(self.inputs.shape[0])


def write_vector_files(
    out_dir: Path,
    qgraph: QuantizedGraph,
    cfg: FixedPointConfig,
    vector_count: int | None = None,
    seed: int | None = None,
    *,
    input_vectors: np.ndarray | None = None,
    vector_source: str | None = None,
) -> list[str]:
    vectors = _prepare_verification_vectors(
        qgraph=qgraph,
        cfg=cfg,
        vector_count=vector_count,
        seed=seed,
        input_vectors=input_vectors,
        vector_source=vector_source,
    )
    return _write_prepared_vector_files(out_dir, qgraph, cfg, vectors)


def _prepare_verification_vectors(
    qgraph: QuantizedGraph,
    cfg: FixedPointConfig,
    vector_count: int | None = None,
    seed: int | None = None,
    *,
    input_vectors: np.ndarray | None = None,
    vector_source: str | None = None,
) -> _VerificationVectors:
    inputs, source, resolved_seed = _resolve_inputs(
        input_size=qgraph.input_size,
        cfg=cfg,
        vector_count=vector_count,
        seed=seed,
        input_vectors=input_vectors,
        vector_source=vector_source,
    )
    results = [infer_quantized(qgraph, row) for row in inputs]
    classes = np.asarray(
        [result.class_id for result in results],
        dtype=np.int64,
    )
    logits = np.stack(
        [np.asarray(result.logits, dtype=np.int64).reshape(-1) for result in results]
    )
    return _VerificationVectors(
        inputs=inputs,
        logits=logits,
        classes=classes,
        source=source,
        seed=resolved_seed,
    )


def _write_prepared_vector_files(
    out_dir: Path,
    qgraph: QuantizedGraph,
    cfg: FixedPointConfig,
    vectors: _VerificationVectors,
) -> list[str]:
    input_path = out_dir / "input_vectors.txt"
    expected_classes = out_dir / "expected_classes.txt"
    expected_logits = out_dir / "expected_logits.txt"
    np.savetxt(input_path, vectors.inputs, fmt="%d")
    np.savetxt(expected_classes, vectors.classes.reshape(-1, 1), fmt="%d")
    np.savetxt(expected_logits, vectors.logits, fmt="%d")

    generated = ["input_vectors.txt", "expected_classes.txt", "expected_logits.txt"]
    for op in qgraph.ops:
        if hasattr(op, "weight") and hasattr(op, "bias"):
            weight_path = out_dir / f"{op.name}_weights.mem"
            bias_path = out_dir / f"{op.name}_bias.mem"
            np.savetxt(weight_path, op.weight.reshape(-1), fmt="%d")
            np.savetxt(bias_path, op.bias.reshape(-1), fmt="%d")
            generated.extend([weight_path.name, bias_path.name])

    metadata = {
        "count": vectors.count,
        "input_size": qgraph.input_size,
        "logits_size": int(vectors.logits.shape[1]),
        "class_count": int(vectors.logits.shape[1]),
        "data_bits": cfg.bits,
        "source": vectors.source,
    }
    if vectors.seed is not None:
        metadata["seed"] = vectors.seed
    metadata_path = out_dir / "vectors.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generated.append("vectors.json")
    return generated


def _resolve_inputs(
    *,
    input_size: int,
    cfg: FixedPointConfig,
    vector_count: int | None,
    seed: int | None,
    input_vectors: np.ndarray | None,
    vector_source: str | None,
) -> tuple[np.ndarray, str, int | None]:
    if input_vectors is None:
        if vector_source is not None:
            raise ValueError("vector_source requires input_vectors")
        resolved_count = 16 if vector_count is None else vector_count
        resolved_seed = 0 if seed is None else seed
        inputs = generate_random_inputs(
            input_size,
            resolved_count,
            cfg,
            resolved_seed,
        )
        return inputs, "random_uniform", resolved_seed

    if vector_count is not None:
        raise ValueError("vector_count cannot be used with input_vectors")
    if seed is not None:
        raise ValueError("seed cannot be used with input_vectors")
    if type(vector_source) is not str or not vector_source.strip():
        raise ValueError(
            "vector_source must be a non-empty string when input_vectors are provided"
        )
    inputs = _validate_custom_inputs(input_vectors, input_size, cfg)
    return inputs, vector_source, None


def _validate_custom_inputs(
    input_vectors: np.ndarray,
    input_size: int,
    cfg: FixedPointConfig,
) -> np.ndarray:
    try:
        inputs = np.asarray(input_vectors)
    except ValueError as exc:
        raise ValueError("input_vectors must be a rectangular two-dimensional array") from exc
    if inputs.ndim != 2:
        raise ValueError(
            "input_vectors must be a two-dimensional array, "
            f"got shape {inputs.shape}"
        )
    if inputs.shape[0] == 0:
        raise ValueError("input_vectors must contain at least one vector")
    if inputs.shape[1] != input_size:
        raise ValueError(
            f"input_vectors must have input_size={input_size} columns, "
            f"got {inputs.shape[1]}"
        )
    if inputs.dtype.kind not in "iu":
        raise TypeError(
            "input_vectors must have an integer dtype, "
            f"got {inputs.dtype}"
        )

    minimum = int(inputs.min())
    maximum = int(inputs.max())
    if minimum < cfg.min_int or maximum > cfg.max_int:
        raise ValueError(
            f"input_vectors values must fit signed {cfg.bits}-bit range "
            f"[{cfg.min_int}, {cfg.max_int}], got [{minimum}, {maximum}]"
        )
    return inputs.astype(np.int64, copy=True)


def generate_random_inputs(
    input_size: int,
    vector_count: int,
    cfg: FixedPointConfig,
    seed: int,
) -> np.ndarray:
    _validate_vector_dimensions(input_size, vector_count)
    rng = np.random.default_rng(seed)
    values = rng.uniform(low=-1.0, high=1.0, size=(vector_count, input_size))
    return quantize_array(values, cfg)


def _validate_vector_dimensions(input_size: int, vector_count: int) -> None:
    if isinstance(vector_count, bool) or not isinstance(vector_count, int):
        raise TypeError("vector_count must be an integer")
    if vector_count <= 0:
        raise ValueError(f"vector_count must be positive, got {vector_count}")
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"input_size must be a positive integer, got {input_size}")
