from __future__ import annotations

import ast
import importlib
from pathlib import Path


TESTS = Path(__file__).resolve().parent
RELEASE_CASE_MODULES = (
    "_release_contract_frontend_outputs.py",
    "_release_contract_model_contract.py",
    "_release_contract_runtime_integrity.py",
    "_release_contract_structural_limits.py",
    "_release_contract_fixed_point.py",
    "_release_contract_vector_report_rtl.py",
    "_release_contract_end_to_end_package.py",
)
BATCHNORM_CASE_MODULES = (
    "_batchnorm_fusion_structure.py",
    "_batchnorm_schema_cases.py",
    "_batchnorm_runtime_bindings.py",
    "_batchnorm_copy_dis_builtins.py",
    "_batchnorm_bootstrap_runtime.py",
    "_batchnorm_internal_component.py",
    "_batchnorm_semantic_boundary.py",
)


def _tree(filename: str) -> ast.Module:
    return ast.parse((TESTS / filename).read_text(encoding="utf-8"))


def test_god_modules_are_thin_stable_collection_facades() -> None:
    expected = {
        "test_release_contract.py": tuple(
            name.removesuffix(".py") for name in RELEASE_CASE_MODULES
        ),
        "test_batchnorm_adversarial.py": tuple(
            name.removesuffix(".py") for name in BATCHNORM_CASE_MODULES
        ),
    }
    for filename, module_names in expected.items():
        path = TESTS / filename
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 20
        tree = _tree(filename)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
        imported = tuple(
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        )
        assert imported == module_names


def test_area_case_modules_each_own_test_bodies() -> None:
    for filename in RELEASE_CASE_MODULES + BATCHNORM_CASE_MODULES:
        tests = tuple(
            node.name
            for node in _tree(filename).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
        assert tests, filename


def test_batchnorm_differential_cases_have_stable_unique_ids() -> None:
    module = importlib.import_module("test_batchnorm_fusion")
    ids = tuple(case.id for case in module._BATCHNORM_SWEEP_CASES)
    expected = tuple(
        f"stats-{variant}-{dtype}-pairs-{pairs}-conv-{bias}"
        for variant in range(4)
        for dtype in ("float32", "float64")
        for pairs in (1, 2)
        for bias in ("no-bias", "bias")
    )
    assert ids == expected
    assert len(ids) == len(set(ids)) == 32
