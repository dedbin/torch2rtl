from __future__ import annotations

import inspect
import itertools
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import torch2rtl
import torch2rtl.frontend as frontend
import torch2rtl.frontend._errors as error_component
import torch2rtl.frontend.pytorch_fx as pytorch_fx


def test_public_frontend_objects_and_metadata_are_stable() -> None:
    assert torch2rtl.parse_model is frontend.parse_model is pytorch_fx.parse_model
    assert (
        torch2rtl.UnsupportedOpError
        is frontend.UnsupportedOpError
        is pytorch_fx.UnsupportedOpError
        is error_component.UnsupportedOpError
    )
    assert inspect.signature(pytorch_fx.parse_model) == inspect.Signature(
        parameters=(
            inspect.Parameter("model", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="object"),
            inspect.Parameter("input_shape", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="Sequence[int]"),
            inspect.Parameter("input_dtype", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None, annotation="object | None"),
        ),
        return_annotation="GraphIR",
    )
    assert pytorch_fx.parse_model.__name__ == "parse_model"
    assert pytorch_fx.parse_model.__qualname__ == "parse_model"
    assert pytorch_fx.parse_model.__module__ == "torch2rtl.frontend.pytorch_fx"
    assert pytorch_fx.UnsupportedOpError.__name__ == "UnsupportedOpError"
    assert pytorch_fx.UnsupportedOpError.__qualname__ == "UnsupportedOpError"
    assert pytorch_fx.UnsupportedOpError.__module__ == "torch2rtl.frontend.pytorch_fx"
    assert issubclass(pytorch_fx.UnsupportedOpError, RuntimeError)
    assert pickle.loads(pickle.dumps(pytorch_fx.parse_model)) is pytorch_fx.parse_model
    restored = pickle.loads(pickle.dumps(pytorch_fx.UnsupportedOpError("example")))
    assert type(restored) is pytorch_fx.UnsupportedOpError
    assert restored.args == ("example",)


def test_public_all_contract_does_not_invent_loader_at_root() -> None:
    assert torch2rtl.__all__ == [
        "FixedPointConfig",
        "UnsupportedOpError",
        "parse_model",
        "quantize_array",
    ]
    assert frontend.__all__ == [
        "UnsupportedOpError",
        "load_model_from_file",
        "parse_model",
    ]
    assert error_component.__all__ == ["UnsupportedOpError"]
    assert not hasattr(torch2rtl, "load_model_from_file")
    assert frontend.load_model_from_file is pytorch_fx.load_model_from_file
    assert str(inspect.signature(pytorch_fx.load_model_from_file)) == (
        "(path: 'Path') -> 'object'"
    )


@pytest.mark.parametrize("order", tuple(itertools.permutations(("root", "frontend", "module"))))
def test_cold_import_order_preserves_one_public_object(order: tuple[str, ...]) -> None:
    script = """
import importlib, json
names = {
    'root': 'torch2rtl',
    'frontend': 'torch2rtl.frontend',
    'module': 'torch2rtl.frontend.pytorch_fx',
}
loaded = [importlib.import_module(names[name]) for name in ORDER]
root = importlib.import_module(names['root'])
frontend = importlib.import_module(names['frontend'])
module = importlib.import_module(names['module'])
print(json.dumps({
    'parse_identity': root.parse_model is frontend.parse_model is module.parse_model,
    'error_identity': root.UnsupportedOpError is frontend.UnsupportedOpError is module.UnsupportedOpError,
    'parse_module': module.parse_model.__module__,
    'error_module': module.UnsupportedOpError.__module__,
}))
""".replace("ORDER", repr(order))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "parse_identity": True,
        "error_identity": True,
        "parse_module": "torch2rtl.frontend.pytorch_fx",
        "error_module": "torch2rtl.frontend.pytorch_fx",
    }


def test_loader_prefers_factory_calls_eval_and_restores_sys_path(tmp_path: Path) -> None:
    model_file = tmp_path / "loader_model.py"
    model_file.write_text(
        """
class Model:
    def __init__(self, source):
        self.source = source
        self.training = True
    def eval(self):
        self.training = False
        return self
model = Model('global')
def create_model():
    return Model('factory')
""",
        encoding="utf-8",
    )
    before = list(sys.path)
    loaded = pytorch_fx.load_model_from_file(model_file)
    assert list(sys.path) == before
    assert loaded.source == "factory"
    assert loaded.training is False


def test_loader_uses_model_fallback_and_calls_eval(tmp_path: Path) -> None:
    model_file = tmp_path / "global_model.py"
    model_file.write_text(
        """
class Model:
    def __init__(self):
        self.training = True
        self.eval_calls = 0
    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self
model = Model()
""",
        encoding="utf-8",
    )

    loaded = pytorch_fx.load_model_from_file(model_file)

    assert loaded.training is False
    assert loaded.eval_calls == 1


@pytest.mark.parametrize(
    ("source", "error_type", "message"),
    [
        ("value = 1\n", ValueError, "must define create_model"),
        ("raise LookupError('import failed')\n", LookupError, "import failed"),
    ],
)
def test_loader_errors_restore_sys_path(
    tmp_path: Path,
    source: str,
    error_type: type[Exception],
    message: str,
) -> None:
    model_file = tmp_path / "bad_model.py"
    model_file.write_text(source, encoding="utf-8")
    before = list(sys.path)
    with pytest.raises(error_type, match=message):
        pytorch_fx.load_model_from_file(model_file)
    assert list(sys.path) == before


def test_loader_missing_path_is_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError) as captured:
        pytorch_fx.load_model_from_file(missing)
    assert captured.value.filename is None
    assert captured.value.args == (missing.resolve(),)
