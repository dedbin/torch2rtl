from __future__ import annotations

from pathlib import Path

from torch2rtl.verify.simulator import _patch_vvp_shebang, _simulation_ok


def test_patch_vvp_shebang_uses_forward_slashes(tmp_path: Path) -> None:
    script = tmp_path / "simv"
    script.write_text("#! \n:ivl_version \"test\";\n", encoding="utf-8")

    _patch_vvp_shebang(script, r"C:\tools\oss-cad-suite\bin\vvp.exe")

    first_line = script.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "#! C:/tools/oss-cad-suite/bin/vvp.exe"


def test_simulation_ok_requires_pass_without_failures() -> None:
    assert _simulation_ok(0, "PASS vectors=16\n")
    assert not _simulation_ok(0, "FAIL vector=16 expected=2 got=1\nFAIL errors=1\n")
    assert not _simulation_ok(1, "PASS vectors=16\n")
