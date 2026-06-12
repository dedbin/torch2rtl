from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch2rtl.eda_tools import detect_eda_tools


def update_report(build_dir: Path, section: str, payload: dict[str, Any]) -> None:
    report_path = build_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"status": {}}
    report["tools"] = detect_eda_tools()
    status = report.setdefault("status", {})
    status[section] = payload
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
