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
  <meta name="color-scheme" content="light">
  <title>Обозреватель схемы Torch2RTL</title>
  <style>
{styles}
  </style>
</head>
<body>
  <a class="skip-link" href="#visualization-canvas">К схеме</a>
  <script id="manifest-data" type="application/json">{manifest_json}</script>
  <main class="app-shell" id="app-shell">
    <header class="topbar">
      <div class="brand-lockup">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
        <div>
          <p class="eyebrow">python → fixed point → rtl</p>
          <h1>torch2rtl <span>Hardware Explorer</span></h1>
        </div>
      </div>
      <div class="build-identity" id="build-identity"></div>
      <div class="topbar-actions">
        <div class="quant-readout" id="quant-readout"></div>
        <div class="segmented" aria-label="Режим подробности">
          <button type="button" data-action="set-mode" data-value="explain">Объяснение</button>
          <button type="button" data-action="set-mode" data-value="engineer">Инженерный</button>
        </div>
        <div class="segmented" aria-label="Представление схемы">
          <button type="button" data-action="set-representation" data-value="rtl">RTL logical</button>
          <button type="button" data-action="set-representation" data-value="yosys">Yosys coarse</button>
        </div>
      </div>
    </header>

    <nav class="context-bar" aria-label="Контекст визуализации">
      <div class="breadcrumbs" id="breadcrumbs"></div>
      <div class="context-tools">
        <button class="context-button vector-trigger" type="button" data-action="open-vectors">
          <span>Вектор</span><strong id="vector-label">#00</strong><i aria-hidden="true">⌄</i>
        </button>
        <div class="zoom-control" aria-label="Масштаб">
          <button type="button" data-action="zoom-out" aria-label="Уменьшить">−</button>
          <output id="zoom-label">100%</output>
          <button type="button" data-action="zoom-in" aria-label="Увеличить">+</button>
        </div>
        <button class="context-button icon-button" type="button" data-action="fit" aria-label="Вписать схему">Fit</button>
        <button class="context-button mobile-inspector-button" type="button" data-action="toggle-inspector">Сведения</button>
      </div>
    </nav>

    <section class="status-strip" id="status-strip" aria-label="Статус сборки"></section>

    <section class="workspace">
      <aside class="panel layer-rail" aria-label="Структура модели">
        <div class="panel-head compact">
          <div><p class="panel-kicker">Структура</p><strong>Модель и RTL</strong></div>
          <span id="layer-count"></span>
        </div>
        <nav class="layer-tree" id="layer-tree" aria-label="Слои модели"></nav>
        <div class="rail-foot" id="rail-summary"></div>
      </aside>

      <article class="panel canvas-panel" aria-labelledby="view-title">
        <div class="canvas-head">
          <div><p class="panel-kicker" id="view-kicker">Логическая схема</p><strong id="view-title">Путь данных RTL</strong></div>
          <div class="canvas-head-meta"><span id="graph-summary"></span><span class="live-indicator"><i></i> interactive</span></div>
        </div>
        <div class="canvas-viewport" id="circuit-viewport">
          <div class="visualization-canvas" id="visualization-canvas" tabindex="0">
            <svg id="circuit-root" role="img" aria-label="Схема Torch2RTL"></svg>
          </div>
          <div class="canvas-hint" id="canvas-hint"></div>
          <div class="minimap" id="minimap" aria-hidden="true"></div>
        </div>
      </article>

      <aside class="panel inspector" id="inspector">
        <div class="inspector-head">
          <div><p class="panel-kicker">Выбранный узел</p><strong>Сведения</strong></div>
          <span id="selected-label"></span>
          <button class="close-inspector" type="button" data-action="toggle-inspector" aria-label="Закрыть сведения">×</button>
        </div>
        <div class="inspector-body" id="inspector-body"></div>
      </aside>
    </section>

    <section class="transport" id="transport" aria-label="Пошаговая трассировка">
      <div class="transport-controls">
        <button type="button" data-action="back" class="transport-button secondary">← Назад</button>
        <button type="button" data-action="play" class="transport-button play-button" aria-label="Запустить трассировку">▶</button>
        <button type="button" data-action="next-step" class="transport-button secondary">Вперёд →</button>
      </div>
      <div class="timeline-wrap">
        <div class="timeline-meta"><strong id="step-label">Шаг 6/6</strong><span id="trace-status" aria-live="polite"></span></div>
        <div class="timeline" id="timeline"></div>
      </div>
      <div class="result-readout" id="result-readout"></div>
    </section>

    <section class="support-data" hidden aria-hidden="true">
      <div id="trace-grid"></div>
      <span id="resource-status"></span><div id="resource-summary"></div>
      <span id="artifact-status"></span><div id="artifact-links"></div>
    </section>
  </main>

  <dialog class="vector-dialog" id="vector-dialog" aria-labelledby="vector-dialog-title">
    <form method="dialog" class="dialog-shell">
      <div class="dialog-head">
        <div><p class="panel-kicker">Проверочные данные</p><h2 id="vector-dialog-title">Выберите входной вектор</h2></div>
        <button value="cancel" aria-label="Закрыть">×</button>
      </div>
      <div class="vector-list" id="vector-list"></div>
    </form>
  </dialog>

  <div class="drawer-backdrop" id="drawer-backdrop" data-action="close-source" hidden></div>
  <aside class="source-drawer" id="source-drawer" aria-labelledby="source-title" aria-hidden="true">
    <div class="drawer-head">
      <div><p class="panel-kicker">Сгенерированный RTL</p><h2 id="source-title">Исходный модуль</h2></div>
      <button type="button" data-action="close-source" aria-label="Закрыть исходник">×</button>
    </div>
    <div class="source-tabs" id="source-tabs"></div>
    <pre class="source-code"><code id="source-code"></code></pre>
  </aside>

  <div class="signal-tooltip" id="signal-tooltip" role="tooltip" hidden></div>
  <div class="screen-reader-status" id="a11y-status" aria-live="polite"></div>
  <script>
{script}
  </script>
</body>
</html>
"""
