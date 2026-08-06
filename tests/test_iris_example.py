from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("sklearn") is None:
    pytest.skip("Iris example dependencies are unavailable", allow_module_level=True)

from examples.iris_mlp.model import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    OUTPUT_SIZE,
    PARAMETER_COUNT,
    IrisMLP,
    trainable_parameter_count,
)
from torch2rtl.eda_tools import detect_eda_tools
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import infer_quantized
from torch2rtl.synth import run_yosys
from torch2rtl.visualization.trace import qgraph_from_manifest
from torch2rtl.verify import run_simulation


IRIS_VECTOR_SOURCE = "iris_test_split"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IrisArtifacts:
    checkpoint_path: Path
    test_accuracy: float
    training_stdout: str
    build_dir: Path
    compilation_stdout: str


@pytest.fixture(scope="module")
def iris_artifacts(tmp_path_factory: pytest.TempPathFactory) -> IrisArtifacts:
    root = tmp_path_factory.mktemp("iris_mlp")
    checkpoint_path = root / "iris_mlp.pt"
    training_process = subprocess.run(
        [
            sys.executable,
            "-B",
            "examples/iris_mlp/train.py",
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert training_process.returncode == 0, training_process.stderr
    accuracy_match = re.search(
        r"^test accuracy: ([0-9.]+)%$",
        training_process.stdout,
        re.M,
    )
    assert accuracy_match is not None, training_process.stdout
    test_accuracy = float(accuracy_match.group(1)) / 100.0
    build_dir = root / "rtl_q8_4"
    compilation_process = subprocess.run(
        [
            sys.executable,
            "-B",
            "examples/iris_mlp/compile.py",
            "--checkpoint",
            str(checkpoint_path),
            "--out",
            str(build_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compilation_process.returncode == 0, compilation_process.stderr
    return IrisArtifacts(
        checkpoint_path,
        test_accuracy,
        training_process.stdout,
        build_dir,
        compilation_process.stdout,
    )


def test_iris_model_architecture_and_parameter_count() -> None:
    model = IrisMLP()

    assert model.fc1.in_features == INPUT_SIZE == 4
    assert model.fc1.out_features == HIDDEN_SIZE == 8
    assert model.fc2.in_features == HIDDEN_SIZE == 8
    assert model.fc2.out_features == OUTPUT_SIZE == 3
    assert isinstance(model.relu, torch.nn.ReLU)
    assert trainable_parameter_count(model) == PARAMETER_COUNT == 67


def test_iris_split_is_deterministic_stratified_and_disjoint() -> None:
    payload = _run_iris_json(
        """
import json
import torch
from examples.iris_mlp.data import prepare_iris_data

first = prepare_iris_data()
second = prepare_iris_data()
indices = [first.train_indices, first.validation_indices, first.test_indices]
print(json.dumps({
    "deterministic": indices == [
        second.train_indices,
        second.validation_indices,
        second.test_indices,
    ],
    "sizes": [len(values) for values in indices],
    "disjoint": all(
        not set(indices[left]) & set(indices[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ),
    "complete": set().union(*map(set, indices)) == set(range(150)),
    "class_counts": [
        torch.bincount(targets, minlength=3).tolist()
        for targets in (
            first.train_targets,
            first.validation_targets,
            first.test_targets,
        )
    ],
}))
"""
    )

    assert payload == {
        "deterministic": True,
        "sizes": [90, 30, 30],
        "disjoint": True,
        "complete": True,
        "class_counts": [[30, 30, 30], [10, 10, 10], [10, 10, 10]],
    }


def test_iris_normalization_is_fitted_only_on_train() -> None:
    payload = _run_iris_json(
        """
import json
import numpy as np
import torch
from examples.iris_mlp.data import prepare_iris_data

data = prepare_iris_data()
raw_train = torch.tensor(data.train_features.to_numpy(), dtype=torch.float32)
raw_all = torch.tensor(np.concatenate([
    data.train_features.to_numpy(),
    data.validation_features.to_numpy(),
    data.test_features.to_numpy(),
]), dtype=torch.float32)
print(json.dumps({
    "min_is_train": torch.equal(data.train_min, raw_train.min(dim=0).values),
    "max_is_train": torch.equal(data.train_max, raw_train.max(dim=0).values),
    "min_is_not_global": not torch.equal(data.train_min, raw_all.min(dim=0).values),
    "normalized_min": data.train_inputs.min(dim=0).values.tolist(),
    "normalized_max": data.train_inputs.max(dim=0).values.tolist(),
}))
"""
    )

    assert payload["min_is_train"] is True
    assert payload["max_is_train"] is True
    assert payload["min_is_not_global"] is True
    np.testing.assert_allclose(payload["normalized_min"], [-1.0] * 4)
    np.testing.assert_allclose(payload["normalized_max"], [1.0] * 4)


def test_iris_checkpoint_loads_into_new_exact_instance(
    iris_artifacts: IrisArtifacts,
) -> None:
    assert iris_artifacts.checkpoint_path.exists()
    assert iris_artifacts.test_accuracy >= 0.90
    assert (
        "checkpoint reload: PASS (new instance, exact parameters/logits/classes)"
        in iris_artifacts.training_stdout
    )


def test_iris_compile_uses_test_split_and_internal_fixed_oracle(
    iris_artifacts: IrisArtifacts,
) -> None:
    build_dir = iris_artifacts.build_dir
    cfg = FixedPointConfig(8, 4, 32)
    written_inputs = np.loadtxt(build_dir / "input_vectors.txt", dtype=np.int64)
    written_logits = np.loadtxt(build_dir / "expected_logits.txt", dtype=np.int64)
    written_classes = np.loadtxt(build_dir / "expected_classes.txt", dtype=np.int64)
    assert written_inputs.shape == (30, 4)

    visualization = json.loads(
        (build_dir / "visualization.json").read_text(encoding="utf-8")
    )
    qgraph = qgraph_from_manifest(visualization, build_dir)
    oracle = [infer_quantized(qgraph, row) for row in written_inputs]
    np.testing.assert_array_equal(
        written_logits,
        np.stack([sample.logits for sample in oracle]),
    )
    np.testing.assert_array_equal(
        written_classes,
        np.asarray([sample.class_id for sample in oracle]),
    )
    saturation = {"fc1": {"min": 0, "max": 0}, "fc2": {"min": 0, "max": 0}}
    for sample in oracle:
        activations = {
            activation.name: activation.values for activation in sample.activations
        }
        for name, counts in saturation.items():
            counts["min"] += int(np.count_nonzero(activations[name] == cfg.min_int))
            counts["max"] += int(np.count_nonzero(activations[name] == cfg.max_int))
    assert saturation == {
        "fc1": {"min": 0, "max": 0},
        "fc2": {"min": 0, "max": 0},
    }
    assert iris_artifacts.test_accuracy >= 0.90
    assert (
        "PyTorch float vs GraphIR float: 30/30 classes match"
        in iris_artifacts.compilation_stdout
    )
    assert (
        "PyTorch float vs Q8.4 fixed: 30/30 classes match"
        in iris_artifacts.compilation_stdout
    )

    vectors = json.loads((build_dir / "vectors.json").read_text(encoding="utf-8"))
    report = json.loads((build_dir / "report.json").read_text(encoding="utf-8"))
    assert vectors["source"] == IRIS_VECTOR_SOURCE
    assert report["vectors"] == {"count": 30, "source": IRIS_VECTOR_SOURCE}
    assert visualization["vectors"]["source"] == IRIS_VECTOR_SOURCE
    assert visualization["trace"]["input"]["values"] == written_inputs[0].tolist()


def test_iris_rtl_passes_when_a_simulator_is_available(
    iris_artifacts: IrisArtifacts,
) -> None:
    tools = detect_eda_tools()
    if not ((tools["iverilog"] and tools["vvp"]) or tools["verilator"]):
        pytest.skip("Icarus Verilog and Verilator are unavailable")

    simulation = run_simulation(iris_artifacts.build_dir)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=30" in simulation.stdout


def test_iris_generic_yosys_synthesis_when_available(
    iris_artifacts: IrisArtifacts,
) -> None:
    if not detect_eda_tools()["yosys"]:
        pytest.skip("Yosys is unavailable")

    synthesis = run_yosys(iris_artifacts.build_dir)

    assert synthesis.ok, synthesis.stdout + synthesis.stderr
    assert synthesis.status == "passed"
    assert synthesis.metrics["cells"] > 0


def _run_iris_json(code: str) -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert type(payload) is dict
    return payload
