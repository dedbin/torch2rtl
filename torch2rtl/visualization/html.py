from __future__ import annotations

import json
from typing import Any

from torch2rtl.visualization.script import SCRIPT
from torch2rtl.visualization.styles import STYLES


def render_visualization_html(manifest: dict[str, Any]) -> str:
    return HTML_TEMPLATE.format(
        manifest_json=_json_for_html(manifest),
        script=SCRIPT,
        styles=STYLES,
    )


def _json_for_html(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Обозреватель схемы Torch2RTL</title>
  <style>
{styles}
  </style>
</head>
<body>
  <script id="manifest-data" type="application/json">{manifest_json}</script>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">артефакт сборки torch2rtl</p>
        <h1>Обозреватель схемы Torch2RTL</h1>
      </div>
      <div class="quant-readout" id="quant-readout"></div>
    </header>
    <section class="status-strip" id="status-strip" aria-label="Статус сборки"></section>
    <section class="workspace">
      <article class="panel canvas-panel">
        <div class="canvas-head">
          <strong>Путь данных RTL</strong>
          <span id="graph-summary"></span>
        </div>
        <div class="canvas-scroll">
          <svg id="circuit-root" role="img" aria-label="Схема Torch2RTL"></svg>
        </div>
      </article>
      <aside class="panel inspector">
        <div class="inspector-head">
          <strong>Сведения</strong>
          <span id="selected-label"></span>
        </div>
        <div class="inspector-body" id="inspector-body"></div>
      </aside>
    </section>
    <section class="trace-panel">
      <div class="trace-head">
        <strong>Эталонный проход</strong>
        <span id="trace-status"></span>
      </div>
      <div class="trace-grid" id="trace-grid"></div>
    </section>
    <section class="report-grid">
      <article class="panel report-panel">
        <div class="report-head">
          <strong>Ресурсы синтеза</strong>
          <span id="resource-status"></span>
        </div>
        <div class="report-body" id="resource-summary"></div>
      </article>
      <article class="panel report-panel">
        <div class="report-head">
          <strong>Артефакты</strong>
          <span id="artifact-status"></span>
        </div>
        <div class="report-body" id="artifact-links"></div>
      </article>
    </section>
  </main>
  <script>
{script}
  </script>
</body>
</html>
"""
