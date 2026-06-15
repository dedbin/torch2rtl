SCRIPT = """
const manifest = JSON.parse(document.getElementById("manifest-data").textContent);
const svg = document.getElementById("circuit-root");
const state = { selectedId: "input" };
const NS = "http://www.w3.org/2000/svg";
const mainBlocks = manifest.blocks.filter((block) => block.lane === "main");
const byId = new Map(manifest.blocks.map((block) => [block.id, block]));
const traceByBlock = new Map((manifest.trace.steps || []).map((step) => [step.block_id, step]));
const kindLabels = {
  input: "вход",
  output: "выход",
  linear: "Linear / MAC",
  conv2d: "Conv2d / MAC",
  relu: "ReLU",
  flatten: "Flatten",
  argmax: "Argmax",
  memory: "память",
};
const detailLabels = {
  name: "имя",
  kind: "тип",
  inputSignal: "входной сигнал",
  outputSignal: "выходной сигнал",
  features: "размер",
  values: "значения",
  file: "файл",
  weights: "веса",
  biases: "смещения",
  macs: "операции MAC",
  in_channels: "входные каналы",
  out_channels: "выходные каналы",
  kernel_size: "ядро",
  stride: "шаг",
  padding: "padding",
  output_shape: "форма выхода",
  threshold: "порог",
  hardware: "аппаратно",
  classes: "классы",
  count: "количество",
  min: "минимум",
  max: "максимум",
  zero_count: "нули",
  outputPreview: "выход",
  range: "диапазон",
};
const statusLabels = {
  passed: "пройдено",
  failed: "ошибка",
  pending: "нет данных",
  not_found: "не найдено",
};

function h(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

function svgEl(name, attrs = {}, text = null) {
  const el = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text !== null) el.textContent = text;
  return el;
}

function shortValues(summary) {
  const preview = (summary.preview || []).join(", ");
  const suffix = summary.size > (summary.preview || []).length ? ", ..." : "";
  return `[${preview}${suffix}]`;
}

function statusClass(status) {
  if (status === "passed" || status === true) return "status-ok";
  if (status === "failed" || status === false) return "status-failed";
  return "status-pending";
}

function statusLabel(status) {
  return statusLabels[status] || status;
}

function statusMessage(message) {
  const text = String(message || "нет данных");
    return text
    .replace("simulation passed", "симуляция пройдена")
    .replace("synthesis passed", "синтез пройден")
    .replace("synthesis failed", "синтез завершился ошибкой")
    .replace("simulator not found", "симулятор не найден")
    .replace("yosys not found", "Yosys не найден")
    .replace("passed", "пройдено")
    .replace("failed", "ошибка")
    .replace("pending", "нет данных");
}

function statusText(name, value) {
  if (!value) return { title: name, status: "pending", message: "нет данных" };
  return {
    title: name,
    status: value.status || (value.ok ? "passed" : "failed"),
    message: statusMessage(value.message || String(value.ok ?? "pending"))
  };
}

function renderReadouts() {
  const q = manifest.quant;
  const items = [
    `${q.bits} бит`,
    `дробных: ${q.frac_bits}`,
    `аккумулятор: ${q.acc_bits}`,
  ];
  document.getElementById("quant-readout").innerHTML = items
    .map((item) => `<span class="chip">${h(item)}</span>`).join("");
  document.getElementById("graph-summary").textContent =
    `${manifest.graph.input_shape.join("x") || "скаляр"} вход -> ${mainBlocks.length - 2} RTL-блока`;

  const build = manifest.build || {};
  const tools = build.tools || {};
  const status = build.status || {};
  const rows = [
    { title: "сборка", status: "passed", message: `файлов: ${(build.generated_files || []).length}` },
    statusText("симуляция", status.simulation),
    statusText("синтез", status.synthesis),
    { title: "EDA-инструменты", status: Object.values(tools).some(Boolean) ? "passed" : "pending",
      message: Object.entries(tools).map(([key, val]) => `${key}:${val ? "да" : "нет"}`).join(" ") || "неизвестно" }
  ];
  document.getElementById("status-strip").innerHTML = rows.map((row) => `
    <div class="status-item ${statusClass(row.status)}">
      <strong>${h(row.title)}</strong><span>${h(statusLabel(row.status))} / ${h(row.message)}</span>
    </div>`).join("");
}

function layoutBlocks() {
  const positions = new Map();
  const gap = 190;
  const w = 142;
  const hgt = 76;
  const startX = 70;
  mainBlocks.forEach((block, index) => {
    positions.set(block.id, { x: startX + index * gap, y: 185, w, h: hgt });
  });
  manifest.blocks.filter((block) => block.lane === "memory").forEach((block) => {
    const parent = positions.get(block.parent);
    const isBias = block.id.endsWith("_bias");
    if (parent) {
      positions.set(block.id, {
        x: parent.x + (isBias ? 78 : -14),
        y: 58,
        w: 106,
        h: 56
      });
    }
  });
  return positions;
}

function drawGrid(width, height) {
  const grid = svgEl("g", { class: "board-grid" });
  for (let x = 0; x <= width; x += 32) {
    grid.appendChild(svgEl("line", { x1: x, y1: 0, x2: x, y2: height }));
  }
  for (let y = 0; y <= height; y += 32) {
    grid.appendChild(svgEl("line", { x1: 0, y1: y, x2: width, y2: y }));
  }
  svg.appendChild(grid);
}

function connectionPath(fromPos, toPos, kind) {
  if (kind === "param") {
    const x1 = fromPos.x + fromPos.w / 2;
    const y1 = fromPos.y + fromPos.h;
    const x2 = toPos.x + toPos.w / 2;
    const y2 = toPos.y;
    const mid = (y1 + y2) / 2;
    return `M ${x1} ${y1} L ${x1} ${mid} L ${x2} ${mid} L ${x2} ${y2}`;
  }
  const x1 = fromPos.x + fromPos.w;
  const y1 = fromPos.y + fromPos.h / 2;
  const x2 = toPos.x;
  const y2 = toPos.y + toPos.h / 2;
  const mid = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
}

function renderCircuit() {
  svg.replaceChildren();
  const positions = layoutBlocks();
  const width = Math.max(980, 140 + mainBlocks.length * 190);
  const height = 430;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  drawGrid(width, height);

  manifest.connections.forEach((conn) => {
    const fromPos = positions.get(conn.from);
    const toPos = positions.get(conn.to);
    if (!fromPos || !toPos) return;
    const active = conn.kind === "data" && conn.to !== "output";
    svg.appendChild(svgEl("path", {
      d: connectionPath(fromPos, toPos, conn.kind),
      class: `wire ${conn.kind}${active ? " active" : ""}`
    }));
    if (conn.kind === "data") {
      const labelX = (fromPos.x + fromPos.w + toPos.x) / 2 - 28;
      const labelY = fromPos.y + fromPos.h / 2 - 11;
      svg.appendChild(svgEl("text", { class: "signal-label", x: labelX, y: labelY }, conn.signal));
    }
  });

  manifest.blocks.forEach((block) => {
    const pos = positions.get(block.id);
    if (!pos) return;
    const group = svgEl("g", {
      class: `block kind-${block.kind}${block.id === state.selectedId ? " selected" : ""}`,
      tabindex: "0",
      role: "button",
      "aria-label": `${block.label} ${block.name}`
    });
    group.appendChild(svgEl("rect", { x: pos.x, y: pos.y, width: pos.w, height: pos.h, rx: 8 }));
    group.appendChild(svgEl("text", { x: pos.x + 12, y: pos.y + 25 }, block.label));
    group.appendChild(svgEl("text", { class: "subtext", x: pos.x + 12, y: pos.y + 47 },
      block.kind === "memory" ? `${block.features} знач.` : `${block.in_features ?? block.features} -> ${block.out_features ?? block.features}`
    ));
    group.addEventListener("click", () => selectBlock(block.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectBlock(block.id);
      }
    });
    svg.appendChild(group);
  });
}

function detailsForBlock(block) {
  const rows = [];
  rows.push([detailLabels.name, block.name || block.id]);
  rows.push([detailLabels.kind, kindLabels[block.kind] || block.kind]);
  if (block.signal_in) rows.push([detailLabels.inputSignal, block.signal_in]);
  if (block.signal_out) rows.push([detailLabels.outputSignal, block.signal_out]);
  if (block.in_features !== undefined) rows.push([detailLabels.features, `${block.in_features} -> ${block.out_features}`]);
  if (block.features !== undefined) rows.push([detailLabels.values, block.features]);
  if (block.file) {
    rows.push([detailLabels.file, { html: `<a href="${encodeURI(block.file)}">${h(block.file)}</a>` }]);
  }
  Object.entries(block.params || {}).forEach(([key, value]) => {
    rows.push([detailLabels[key] || key, value]);
  });
  Object.entries(block.stats || {}).forEach(([key, value]) => {
    rows.push([detailLabels[key] || key, value]);
  });
  const trace = traceByBlock.get(block.id);
  if (trace) {
    rows.push([detailLabels.outputPreview, { html: `<code>${h(shortValues(trace.output))}</code>` }]);
    rows.push([detailLabels.range, `${trace.output.min}..${trace.output.max}`]);
  }
  return rows;
}

function renderInspector() {
  const block = byId.get(state.selectedId) || byId.get("input");
  document.getElementById("selected-label").textContent = block.id;
  const rows = detailsForBlock(block).map(([key, value]) =>
    `<dt>${h(key)}</dt><dd>${value && value.html ? value.html : h(value)}</dd>`
  ).join("");
  document.getElementById("inspector-body").innerHTML = `
    <h2 class="detail-title">${h(block.label)}</h2>
    <p class="detail-kind">${h(kindLabels[block.kind] || block.kind)}</p>
    <dl class="kv">${rows}</dl>
  `;
}

function renderTrace() {
  const trace = manifest.trace;
  const match = trace.matched ? "совпало" : "не совпало";
  document.getElementById("trace-status").textContent =
    `вектор ${trace.vector_index} / ожидалось ${trace.expected_class ?? "н/д"} / ${match}`;
  const steps = (trace.steps || []).map((step) => `
    <button class="trace-step" type="button" data-block="${h(step.block_id)}">
      <strong>${h(kindLabels[step.kind] || step.kind)}</strong>
      <span>${h(shortValues(step.output))}</span>
      <span>диапазон ${h(step.output.min)}..${h(step.output.max)}</span>
    </button>`).join("");
  document.getElementById("trace-grid").innerHTML = `
    <div class="trace-summary">
      <p class="class-readout">класс ${h(trace.class_id)}</p>
      <div class="kv">
        <dt>вход</dt><dd><code>${h(shortValues(trace.input))}</code></dd>
        <dt>логиты</dt><dd><code>${h(shortValues(trace.logits))}</code></dd>
        <dt>ненулевые</dt><dd>${h(trace.logits.nonzero)} / ${h(trace.logits.size)}</dd>
      </div>
    </div>
    <div class="trace-steps">${steps}</div>
  `;
  document.querySelectorAll(".trace-step").forEach((el) => {
    el.addEventListener("click", () => selectBlock(el.dataset.block));
  });
}

function renderResources() {
  const build = manifest.build || {};
  const synthesis = build.synthesis || (build.status || {}).synthesis || {};
  const resource = synthesis.metrics || {};
  const model = build.metrics || {};
  const reference = build.reference || {};
  const rows = [
    ["параметры модели", model.parameters],
    ["MAC в модели", model.macs],
    ["ячейки Yosys", resource.cells],
    ["провода", resource.wires],
    ["битов проводов", resource.wire_bits],
    ["порты", resource.ports],
    ["совпадения классов", reference.class_matches],
    ["ошибки классов", reference.class_mismatches],
    ["средняя ошибка логитов", reference.mean_abs_logit_error],
    ["макс. ошибка логитов", reference.max_abs_logit_error],
  ].filter(([, value]) => value !== undefined && value !== null);

  const cellTypes = Object.entries(resource.cell_types || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  const cellRows = cellTypes.map(([name, value]) => `<dt>${h(name)}</dt><dd>${h(value)}</dd>`).join("");
  document.getElementById("resource-status").textContent = statusMessage(synthesis.message);
  document.getElementById("resource-summary").innerHTML = rows.length
    ? `<dl class="kv">${rows.map(([key, value]) => `<dt>${h(key)}</dt><dd>${h(value)}</dd>`).join("")}${cellRows}</dl>`
    : `<p class="empty-note">нет данных синтеза</p>`;
}

function renderArtifacts() {
  const artifacts = (manifest.build || {}).artifacts || [];
  document.getElementById("artifact-status").textContent = `${artifacts.length} файлов`;
  document.getElementById("artifact-links").innerHTML = artifacts.length
    ? `<div class="artifact-list">${artifacts.map((item) => `
        <a href="${encodeURI(item.file)}">
          <strong>${h(item.file)}</strong>
          <span>${h(item.size_bytes)} байт</span>
        </a>`).join("")}</div>`
    : `<p class="empty-note">артефакты не найдены</p>`;
}

function selectBlock(id) {
  state.selectedId = id;
  renderCircuit();
  renderInspector();
}

renderReadouts();
renderCircuit();
renderInspector();
renderTrace();
renderResources();
renderArtifacts();
"""
