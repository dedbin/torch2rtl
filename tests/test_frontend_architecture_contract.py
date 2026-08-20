from __future__ import annotations

import ast
from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "torch2rtl" / "frontend"
PYTORCH_FX = FRONTEND / "pytorch_fx.py"
COMPONENTS = (
    "_errors.py",
    "_state_guard.py",
    "_model_contract.py",
    "_fusion.py",
    "_lowering.py",
    "_semantics.py",
    "_pipeline.py",
)
ALLOWED_INTERNAL_IMPORTS = {
    "_errors.py": frozenset(),
    "_state_guard.py": frozenset({"torch2rtl.frontend._errors"}),
    "_model_contract.py": frozenset(
        {
            "torch2rtl.frontend._errors",
            "torch2rtl.frontend._state_guard",
        }
    ),
    "_fusion.py": frozenset(
        {
            "torch2rtl.frontend._errors",
            "torch2rtl.frontend._state_guard",
        }
    ),
    "_lowering.py": frozenset({"torch2rtl.frontend._errors"}),
    "_semantics.py": frozenset(
        {
            "torch2rtl.frontend._errors",
            "torch2rtl.frontend._state_guard",
        }
    ),
    "_pipeline.py": frozenset(
        {
            "torch2rtl.frontend._errors",
            "torch2rtl.frontend._fusion",
            "torch2rtl.frontend._lowering",
            "torch2rtl.frontend._model_contract",
            "torch2rtl.frontend._semantics",
            "torch2rtl.frontend._state_guard",
        }
    ),
}
FORBIDDEN_INTERNAL_IMPORTS = frozenset(
    {
        "torch2rtl.frontend",
        "torch2rtl.frontend.pytorch_fx",
    }
)
PYTORCH_FX_MAX_LINES = 2301
PYTORCH_FX_LARGE_FUNCTION_MAX_LOC = {
    "_definition_fingerprint": 76,
    "_validate_framework_integrity_impl": 215,
    "_make_integrity_validator": 261,
    "_make_parse_model": 138,
}
COMPLEXITY_EXCEPTIONS = {
    ("_fusion.py", "_fuse_conv_batchnorm_eval"): (90, 16),
    ("_fusion.py", "_validate_conv_batchnorm_pair"): (222, 46),
    ("_lowering.py", "lower_fx_graph"): (128, 23),
    ("_lowering.py", "_lower_conv2d_module"): (159, 21),
    ("_model_contract.py", "_validate_model_semantics"): (105, 24),
    ("_model_contract.py", "_validate_reachable_python_function"): (156, 48),
    ("_pipeline.py", "_parse_model_impl"): (87, 9),
    ("_semantics.py", "_validate_fusion_boundary"): (114, 19),
    ("_state_guard.py", "_trusted_named_tensors"): (82, 30),
    ("_state_guard.py", "_freeze_state_value"): (85, 16),
}


def _tree(filename: str) -> ast.Module:
    return ast.parse((FRONTEND / filename).read_text(encoding="utf-8"))


def _complexity(function: ast.FunctionDef) -> int:
    branch_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.BoolOp,
        ast.IfExp,
        ast.comprehension,
        ast.Match,
    )
    return 1 + sum(isinstance(node, branch_nodes) for node in ast.walk(function))


def _frontend_imports(tree: ast.Module) -> frozenset[str]:
    imported: set[str] = set()
    prefix = "torch2rtl.frontend"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name
                for alias in node.names
                if alias.name == prefix or alias.name.startswith(f"{prefix}.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level:
                package = prefix.split(".")
                retained = len(package) - (node.level - 1)
                relative_base = package[:retained]
                module = ".".join(
                    relative_base + ([] if module is None else module.split("."))
                )
            if module == prefix:
                imported.add(prefix)
                imported.update(f"{prefix}.{alias.name}" for alias in node.names)
            elif module is not None and module.startswith(f"{prefix}."):
                imported.add(module)
    return frozenset(imported)


def test_internal_frontend_dependency_dag_is_exact() -> None:
    assert tuple(
        path.name
        for path in sorted(FRONTEND.glob("_*.py"))
        if path.name != "__init__.py"
    ) == tuple(sorted(COMPONENTS))
    for filename in COMPONENTS:
        imported = _frontend_imports(_tree(filename))
        assert imported == ALLOWED_INTERNAL_IMPORTS[filename]
        assert imported.isdisjoint(FORBIDDEN_INTERNAL_IMPORTS)


def test_forbidden_back_edges_are_detected_independent_of_import_form() -> None:
    samples = (
        "from torch2rtl.frontend import pytorch_fx",
        "import torch2rtl.frontend.pytorch_fx",
        "from . import pytorch_fx",
        "from ..frontend import pytorch_fx",
    )
    for source in samples:
        assert not _frontend_imports(ast.parse(source)).isdisjoint(
            FORBIDDEN_INTERNAL_IMPORTS
        )


def test_pytorch_fx_file_and_large_functions_do_not_grow() -> None:
    source = PYTORCH_FX.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= PYTORCH_FX_MAX_LINES
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    large_functions = {
        name
        for name, node in functions.items()
        if (node.end_lineno or node.lineno) - node.lineno + 1 > 75
    }
    assert large_functions <= set(PYTORCH_FX_LARGE_FUNCTION_MAX_LOC)
    for name, max_loc in PYTORCH_FX_LARGE_FUNCTION_MAX_LOC.items():
        node = functions[name]
        loc = (node.end_lineno or node.lineno) - node.lineno + 1
        assert loc <= max_loc


def test_internal_components_do_not_define_public_parser_entrypoints() -> None:
    for filename in COMPONENTS:
        names = {
            node.name
            for node in _tree(filename).body
            if isinstance(node, ast.FunctionDef)
        }
        assert "parse_model" not in names


def test_component_complexity_does_not_grow() -> None:
    seen_exceptions: set[tuple[str, str]] = set()
    for filename in COMPONENTS:
        for node in _tree(filename).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            key = (filename, node.name)
            loc = (node.end_lineno or node.lineno) - node.lineno + 1
            complexity = _complexity(node)
            if key in COMPLEXITY_EXCEPTIONS:
                max_loc, max_complexity = COMPLEXITY_EXCEPTIONS[key]
                assert loc <= max_loc
                assert complexity <= max_complexity
                seen_exceptions.add(key)
            else:
                assert loc <= 75, key
                assert complexity <= 15, key
    assert seen_exceptions == set(COMPLEXITY_EXCEPTIONS)


def test_no_unified_operator_registry_was_introduced() -> None:
    for filename in COMPONENTS:
        source = (FRONTEND / filename).read_text(encoding="utf-8")
        assert "SUPPORTED_OPERATORS" not in source
        assert "register_operator" not in source
