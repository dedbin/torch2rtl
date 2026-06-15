from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch2rtl.eda_tools import detect_eda_tools


def update_report(build_dir: Path, section: str, payload: dict[str, Any]) -> None:
    report_path = build_dir / "report.json"
    report = _read_report(report_path)
    report["tools"] = detect_eda_tools()
    status = report.setdefault("status", {})
    status[section] = payload
    report[section] = payload
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _refresh_visualization(build_dir)


def update_report_metadata(build_dir: Path, payload: dict[str, Any]) -> None:
    report_path = build_dir / "report.json"
    report = _read_report(report_path)
    report.update(payload)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _refresh_visualization(build_dir)


def _read_report(report_path: Path) -> dict[str, Any]:
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {"status": {}}


def _refresh_visualization(build_dir: Path) -> None:
    manifest_path = build_dir / "visualization.json"
    if not manifest_path.exists():
        return

    from torch2rtl.visualization.manifest import render_visualization_from_build

    render_visualization_from_build(build_dir)
