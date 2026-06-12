from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array
from torch2rtl.quant.reference import QuantizedGraph, infer_quantized


def write_vector_files(
    out_dir: Path,
    qgraph: QuantizedGraph,
    cfg: FixedPointConfig,
    vector_count: int,
    seed: int,
) -> list[str]:
    inputs = generate_random_inputs(qgraph.input_size, vector_count, cfg, seed)
    expected = np.asarray(
        [infer_quantized(qgraph, row).class_id for row in inputs],
        dtype=np.int64,
    )

    input_path = out_dir / "input_vectors.txt"
    expected_path = out_dir / "expected_classes.txt"
    np.savetxt(input_path, inputs, fmt="%d")
    np.savetxt(expected_path, expected.reshape(-1, 1), fmt="%d")

    generated = ["input_vectors.txt", "expected_classes.txt"]
    for op in qgraph.ops:
        if hasattr(op, "weight") and hasattr(op, "bias"):
            weight_path = out_dir / f"{op.name}_weights.mem"
            bias_path = out_dir / f"{op.name}_bias.mem"
            np.savetxt(weight_path, op.weight.reshape(-1), fmt="%d")
            np.savetxt(bias_path, op.bias.reshape(-1), fmt="%d")
            generated.extend([weight_path.name, bias_path.name])

    metadata = {
        "count": vector_count,
        "input_size": qgraph.input_size,
        "seed": seed,
    }
    metadata_path = out_dir / "vectors.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generated.append("vectors.json")
    return generated


def generate_random_inputs(
    input_size: int,
    vector_count: int,
    cfg: FixedPointConfig,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(low=-1.0, high=1.0, size=(vector_count, input_size))
    return quantize_array(values, cfg)
