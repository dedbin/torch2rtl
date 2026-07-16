SCRIPT = r"""
const manifest = JSON.parse(document.getElementById("manifest-data").textContent);
const mainBlocks = manifest.blocks.filter((block) => block.lane === "main");
const memoryBlocks = manifest.blocks.filter((block) => block.lane === "memory");
const byId = new Map(manifest.blocks.map((block) => [block.id, block]));
const traceList = Array.isArray(manifest.traces?.traces) && manifest.traces.traces.length
  ? manifest.traces.traces
  : [manifest.trace];
const sourcePreviews = manifest.build?.source_previews || {};
const kindLabels = {
  input: "Шина входа",
  output: "Выход класса",
  linear: "Linear / MAC",
  conv2d: "Conv2d / MAC",
  relu: "ReLU",
  flatten: "Flatten",
  argmax: "Argmax",
  memory: "Память параметров",
};
const routeLabels = {
  overview: "Модель",
  conv: "Conv2d",
  mac: "MAC",
  linear: "Linear + Argmax",
  yosys: "Yosys coarse",
  mapping: "Logical ↔ Yosys",
};
const statusLabels = {
  passed: "пройдено",
  failed: "ошибка",
  pending: "нет данных",
  not_found: "не найдено",
};
const timelineLabels = {
  overview: ["Вход", "Conv", "ReLU", "Linear", "Argmax", "Класс"],
  conv: ["Окно", "Веса", "×", "Σ", ">>>", "Выход"],
  mac: ["Операнды", "×", "Частичные", "ACC", ">>>", "Результат"],
  linear: ["Вектор", "Матрица", "MAC", "Логиты", "Сравнение", "Класс"],
  yosys: ["RTL", "prep", "cells", "nets", "mapping", "review"],
  mapping: ["Logical", "порты", "cell class", "source", "net", "mapping"],
};

if (!window.location.hash) history.replaceState(null, "", "#overview");

const state = {
  route: parseRoute(),
  mode: "explain",
  representation: parseRoute().name === "yosys" || parseRoute().name === "mapping" ? "yosys" : "rtl",
  vectorIndex: Number(manifest.trace?.vector_index || 0),
  selectedId: "input",
  selectedOutput: null,
  macStep: parseRoute().name === "mac" ? 0 : 5,
  playing: false,
  zoom: 1,
  inspectorOpen: false,
  sourceFile: Object.keys(sourcePreviews)[0] || "",
};

let playTimer = null;

function h(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function parseRoute() {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const name = routeLabels[parts[0]] ? parts[0] : "overview";
  return {
    name,
    id: parts[1] ? decodeURIComponent(parts[1]) : null,
    output: Number.isFinite(Number(parts[2])) ? Number(parts[2]) : null,
    cell: parts[3] ? decodeURIComponent(parts[3]) : null,
  };
}

function routeHash(name, id = null, output = null, cell = null) {
  const parts = [name];
  if (id !== null && id !== undefined) parts.push(encodeURIComponent(id));
  if (output !== null && output !== undefined) parts.push(String(output));
  if (cell !== null && cell !== undefined) parts.push(encodeURIComponent(cell));
  return `#${parts.join("/")}`;
}

function navigate(hash) {
  if (window.location.hash === hash) {
    state.route = parseRoute();
    renderAll();
    return;
  }
  window.location.hash = hash;
}

function currentTrace() {
  return traceList.find((trace) => Number(trace.vector_index) === Number(state.vectorIndex))
    || traceList[0]
    || manifest.trace;
}

function traceStep(blockOrKind) {
  const trace = currentTrace();
  return (trace.steps || []).find((step) => step.block_id === blockOrKind)
    || (trace.steps || []).find((step) => step.kind === blockOrKind)
    || null;
}

function summaryValues(summary) {
  return Array.isArray(summary?.values) ? summary.values : (summary?.preview || []);
}

function summaryText(summary, limit = 8) {
  const values = summaryValues(summary);
  const shown = values.slice(0, limit).join(", ");
  return `[${shown}${Number(summary?.size || 0) > limit ? ", …" : ""}]`;
}

function qScale(extraFracBits = 0) {
  return 2 ** (Number(manifest.quant.frac_bits) + extraFracBits);
}

function decoded(raw, extraFracBits = 0, digits = 6) {
  const value = Number(raw) / qScale(extraFracBits);
  return Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function signedHex(value, bits = 32) {
  const width = Math.ceil(bits / 4);
  const modulus = 2n ** BigInt(bits);
  let number = BigInt(Math.trunc(Number(value)));
  if (number < 0) number = modulus + number;
  return `0x${number.toString(16).toUpperCase().padStart(width, "0").slice(-width)}`;
}

function valueClass(value) {
  return Number(value) < 0 ? "negative" : Number(value) > 0 ? "positive" : "zero";
}

function statusClass(status) {
  if (status === "passed" || status === true) return "status-ok";
  if (status === "failed" || status === false) return "status-failed";
  return "status-pending";
}

function statusLabel(status) {
  return statusLabels[status] || status || "нет данных";
}

function statusMessage(message) {
  return String(message || "нет данных")
    .replace("simulation passed", "симуляция пройдена")
    .replace("synthesis passed", "синтез пройден")
    .replace("synthesis failed", "синтез завершился ошибкой")
    .replace("simulator not found", "симулятор не найден")
    .replace("yosys not found", "Yosys не найден")
    .replace("install Icarus Verilog or Verilator", "установите Icarus или Verilator")
    .replace("install Yosys to run synthesis", "установите Yosys для синтеза");
}

function activeBlockId() {
  if (["conv", "mac", "linear", "mapping"].includes(state.route.name) && state.route.id) {
    return state.route.id;
  }
  if (state.route.name === "yosys") return "yosys";
  return state.selectedId;
}

function routeForBlock(block) {
  if (!block) return "#overview";
  if (block.kind === "conv2d") return routeHash("conv", block.id);
  if (block.kind === "linear" || block.kind === "argmax") {
    const linear = block.kind === "linear" ? block : mainBlocks.find((item) => item.kind === "linear");
    return routeHash("linear", linear?.id || block.id);
  }
  if (block.kind === "memory") return routeForBlock(byId.get(block.parent));
  return "#overview";
}

function selectedHardwareEntry(blockId, requested = null) {
  const step = traceStep(blockId);
  const entries = step?.hardware?.output_entries || [];
  if (!entries.length) return null;
  const outputValues = summaryValues(step.output);
  const fallbackIndex = outputValues.length
    ? outputValues.indexOf(Math.max(...outputValues))
    : 0;
  const flatIndex = requested ?? state.selectedOutput ?? fallbackIndex;
  return entries.find((entry) => Number(entry.flat_index) === Number(flatIndex)) || entries[0];
}

function renderReadouts() {
  const q = manifest.quant;
  const demo = manifest.build?.demo || {};
  document.getElementById("build-identity").innerHTML = `
    <strong>${h(demo.name || "custom-build")}</strong>
    <span>${h(demo.model_path || "generated RTL")}</span>`;
  const simulation = manifest.build?.status?.simulation;
  const synthesis = manifest.build?.status?.synthesis;
  const items = [
    [`Q${q.bits}.${q.frac_bits}`, ""],
    [`ACC ${q.acc_bits}`, ""],
    [simulation?.status === "passed" ? "SIM PASS" : "SIM DATA", simulation?.status === "passed" ? "ok" : ""],
    [synthesis?.status === "passed" ? "SYNTH PASS" : "RTL READY", synthesis?.status === "passed" ? "ok" : ""],
  ];
  document.getElementById("quant-readout").innerHTML = items
    .map(([label, cls]) => `<span class="chip ${cls}">${h(label)}</span>`).join("");
  document.querySelectorAll('[data-action="set-mode"]').forEach((button) => {
    button.classList.toggle("active", button.dataset.value === state.mode);
    button.setAttribute("aria-pressed", String(button.dataset.value === state.mode));
  });
  document.querySelectorAll('[data-action="set-representation"]').forEach((button) => {
    button.classList.toggle("active", button.dataset.value === state.representation);
    button.setAttribute("aria-pressed", String(button.dataset.value === state.representation));
  });
  document.getElementById("vector-label").textContent = `#${String(state.vectorIndex).padStart(2, "0")}`;
  document.getElementById("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
}

function renderStatuses() {
  const build = manifest.build || {};
  const statuses = build.status || {};
  const rows = [
    {title: "сборка", status: "passed", message: `${(build.generated_files || []).length} файлов`},
    {title: "симуляция", status: statuses.simulation?.status || "pending", message: statusMessage(statuses.simulation?.message)},
    {title: "синтез", status: statuses.synthesis?.status || "pending", message: statusMessage(statuses.synthesis?.message)},
    {title: "эталон", status: currentTrace()?.matched ? "passed" : "failed", message: currentTrace()?.matched ? "class MATCH" : "class MISMATCH"},
  ];
  document.getElementById("status-strip").innerHTML = rows.map((row) => `
    <div class="status-item ${statusClass(row.status)}" title="${h(row.message)}">
      <strong>${h(row.title)}</strong><span>${h(statusLabel(row.status))} · ${h(row.message)}</span>
    </div>`).join("");
}

function breadcrumbsForRoute() {
  const route = state.route;
  const crumbs = [{label: "Модель", hash: "#overview"}];
  const block = route.id ? byId.get(route.id) : null;
  if (route.name === "conv") crumbs.push({label: block?.name || "Conv2d", hash: routeHash("conv", route.id)});
  if (route.name === "mac") {
    crumbs.push({label: block?.kind === "linear" ? "Linear" : "Conv2d", hash: routeForBlock(block)});
    crumbs.push({label: `MAC out[${route.output ?? state.selectedOutput ?? 0}]`, hash: window.location.hash});
  }
  if (route.name === "linear") crumbs.push({label: block?.name || "Linear", hash: window.location.hash});
  if (route.name === "yosys") crumbs.push({label: "Yosys coarse", hash: "#yosys"});
  if (route.name === "mapping") {
    crumbs.push({label: "Yosys coarse", hash: "#yosys"});
    crumbs.push({label: "Сопоставление", hash: window.location.hash});
  }
  return crumbs;
}

function renderBreadcrumbs() {
  const crumbs = breadcrumbsForRoute();
  document.getElementById("breadcrumbs").innerHTML = crumbs.map((crumb, index) => {
    const tail = index === crumbs.length - 1;
    const label = tail
      ? `<strong>${h(crumb.label)}</strong>`
      : `<button type="button" data-action="route" data-route="${h(crumb.hash)}">${h(crumb.label)}</button>`;
    return `${index ? '<span class="crumb-separator">/</span>' : ""}${label}`;
  }).join("");
}

function renderLayerTree() {
  const active = activeBlockId();
  const main = mainBlocks.map((block) => `
    <button class="layer-item kind-${h(block.kind)} ${active === block.id ? "active" : ""}"
      type="button" data-action="route" data-route="${h(routeForBlock(block))}" data-block="${h(block.id)}">
      <strong>${h(kindLabels[block.kind] || block.label)}</strong>
      <span>${h(block.name || block.id)}</span>
    </button>`).join("");
  const memories = memoryBlocks.map((block) => `
    <button class="layer-item kind-memory ${active === block.id ? "active" : ""}"
      type="button" data-action="route" data-route="${h(routeForBlock(block))}" data-block="${h(block.id)}">
      <strong>${h(block.label)}</strong><span>${h(block.name)}</span>
    </button>`).join("");
  document.getElementById("layer-tree").innerHTML = `
    <div class="rail-group-label">Data path</div>${main}
    <div class="rail-group-label">Constant banks</div>${memories}
    <div class="rail-group-label">Synthesis</div>
    <button class="layer-item ${active === "yosys" ? "active" : ""}" type="button" data-action="route" data-route="#yosys">
      <strong>Yosys coarse</strong><span>prep · cells · nets</span>
    </button>`;
  document.getElementById("layer-count").textContent = `${mainBlocks.length} + ${memoryBlocks.length}`;
  const metrics = manifest.build?.metrics || {};
  document.getElementById("rail-summary").innerHTML = [
    [metrics.parameters ?? "—", "параметров"],
    [metrics.macs ?? "—", "MAC"],
    [manifest.vectors?.count ?? traceList.length, "векторов"],
  ].map(([value, label]) => `<div class="rail-stat"><strong>${h(value)}</strong><span>${h(label)}</span></div>`).join("");
}

function updateCanvasHeader(kicker, title, summary, hint) {
  document.getElementById("view-kicker").textContent = kicker;
  document.getElementById("view-title").textContent = title;
  document.getElementById("graph-summary").textContent = summary;
  document.getElementById("canvas-hint").innerHTML = hint;
}

function setCanvas(markup, minimumWidth = 760) {
  const canvas = document.getElementById("visualization-canvas");
  canvas.style.setProperty("--canvas-zoom", state.zoom);
  canvas.style.minWidth = `${minimumWidth}px`;
  canvas.innerHTML = markup;
}

function renderOverview() {
  const trace = currentTrace();
  const width = Math.max(980, 90 + mainBlocks.length * 158);
  const height = 560;
  const positions = new Map();
  mainBlocks.forEach((block, index) => positions.set(block.id, {x: 45 + index * 158, y: 260, w: 122, h: 74}));
  memoryBlocks.forEach((block) => {
    const parent = positions.get(block.parent);
    if (!parent) return;
    const bias = block.id.endsWith("_bias");
    positions.set(block.id, {x: parent.x + (bias ? 66 : -6), y: 104, w: 96, h: 54});
  });
  const marker = `<defs><marker id="arrow-copper" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#c77a3c"></path></marker></defs>`;
  const dataWires = manifest.connections.filter((item) => item.kind === "data").map((conn) => {
    const from = positions.get(conn.from);
    const to = positions.get(conn.to);
    if (!from || !to) return "";
    const x1 = from.x + from.w;
    const y1 = from.y + from.h / 2;
    const x2 = to.x;
    const y2 = to.y + to.h / 2;
    const mid = (x1 + x2) / 2;
    return `<path class="schematic-wire active" d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}" marker-end="url(#arrow-copper)" data-signal="${h(conn.signal)}"></path>
      <text class="schematic-label" x="${mid - 24}" y="${y1 - 10}">${h(conn.signal)}</text>`;
  }).join("");
  const paramWires = manifest.connections.filter((item) => item.kind === "param").map((conn) => {
    const from = positions.get(conn.from);
    const to = positions.get(conn.to);
    if (!from || !to) return "";
    const x1 = from.x + from.w / 2;
    const y1 = from.y + from.h;
    const x2 = to.x + to.w / 2;
    const y2 = to.y;
    const mid = (y1 + y2) / 2;
    return `<path class="schematic-wire param" d="M${x1},${y1} L${x1},${mid} L${x2},${mid} L${x2},${y2}" data-signal="${h(conn.signal)}"></path>`;
  }).join("");
  const nodes = manifest.blocks.map((block) => {
    const pos = positions.get(block.id);
    if (!pos) return "";
    const isMemory = block.kind === "memory";
    const route = routeForBlock(block);
    const step = traceStep(block.id);
    const output = step ? summaryText(step.output, 4) : block.kind === "output" ? `class ${trace.class_id}` : `${block.in_features ?? block.features ?? ""} → ${block.out_features ?? ""}`;
    return `<g class="schematic-node kind-${h(block.kind)} ${state.selectedId === block.id ? "selected" : ""}"
      role="button" tabindex="0" data-action="route" data-route="${h(route)}" data-block="${h(block.id)}">
      <rect x="${pos.x}" y="${pos.y}" width="${pos.w}" height="${pos.h}" rx="9"></rect>
      <circle class="schematic-port" cx="${pos.x}" cy="${pos.y + pos.h / 2}" r="4"></circle>
      <circle class="schematic-port" cx="${pos.x + pos.w}" cy="${pos.y + pos.h / 2}" r="4"></circle>
      <text x="${pos.x + 10}" y="${pos.y + (isMemory ? 22 : 26)}">${h(block.label)}</text>
      <text class="subtext" x="${pos.x + 10}" y="${pos.y + (isMemory ? 40 : 48)}">${h(output)}</text>
      ${!isMemory ? `<text class="subtext" x="${pos.x + 10}" y="${pos.y + 63}">${h(block.name)}</text>` : ""}
    </g>`;
  }).join("");
  const note = `<div class="board-note"><strong>Макет логического data path.</strong> Нажмите Conv2d или Linear, чтобы развернуть операторы ×, +, сдвиг и saturation. Это RTL-структура, не физическое размещение на FPGA.</div>`;
  setCanvas(`${note}<svg id="circuit-root" viewBox="0 0 ${width} ${height}" data-view="overview" aria-label="Обзор пути данных">${marker}${dataWires}${paramWires}${nodes}</svg>`, width);
  updateCanvasHeader(
    "Модель / RTL logical",
    "Путь данных от входа до class_id",
    `${manifest.graph.input_shape.join("×")} → ${mainBlocks.length - 2} операций`,
    "Клик по вычислительному блоку раскрывает его структуру · <strong>зелёным</strong> показан активный вектор"
  );
}

function matrixCells(values, columns, options = {}) {
  const receptive = options.receptive || new Set();
  return values.map((value, index) => {
    const key = options.keys?.[index] || String(index);
    const selected = Number(options.selected) === index;
    const classes = ["matrix-cell", valueClass(value), receptive.has(key) ? "receptive" : "", selected ? "selected" : ""].filter(Boolean).join(" ");
    const attrs = options.clickable
      ? `type="button" data-action="select-output" data-output="${index}"`
      : `role="gridcell"`;
    const tag = options.clickable ? "button" : "div";
    return `<${tag} class="${classes}" ${attrs} data-signal="raw ${h(value)} · ${h(decoded(value))}">
      <span>${h(value)}</span><small>${h(decoded(value, 0, 4))}</small>
    </${tag}>`;
  }).join("");
}

function renderConv() {
  const block = byId.get(state.route.id) || mainBlocks.find((item) => item.kind === "conv2d");
  if (!block) return renderOverview();
  state.selectedId = block.id;
  const step = traceStep(block.id);
  const hardware = step?.hardware;
  if (!hardware) return renderDataUnavailable(block, "Детальная Conv2d-трассировка отсутствует в этой сборке.");
  const inputValues = summaryValues(step.input);
  const outputValues = summaryValues(step.output);
  const inputCols = Number(hardware.input_shape?.at(-1) || Math.ceil(Math.sqrt(inputValues.length)));
  const outputCols = Number(hardware.output_shape?.at(-1) || Math.ceil(Math.sqrt(outputValues.length)));
  const kernelCols = Number(hardware.weight_shape?.at(-1) || 1);
  const fallback = outputValues.indexOf(Math.max(...outputValues));
  const selected = state.selectedOutput ?? fallback;
  state.selectedOutput = selected;
  const entry = selectedHardwareEntry(block.id, selected);
  const receptive = new Set((entry?.taps || []).map((tap) => tap.input_index.join(":")));
  const inputKeys = inputValues.map((_, index) => {
    const spatial = inputCols ? index % (inputCols * Number(hardware.input_shape?.at(-2) || 1)) : index;
    const y = Math.floor(spatial / inputCols);
    const x = spatial % inputCols;
    return `0:${y}:${x}`;
  });
  const kernelCount = (entry?.taps || []).length || hardware.weights.length;
  const kernel = (entry?.taps || []).map((tap) => tap.weight_raw).slice(0, kernelCount);
  const outIndex = entry?.index || [0, 0, selected];
  const note = `<div class="board-note"><strong>Receptive field → один MAC.</strong> Выберите ячейку выхода: зелёная рамка покажет четыре входа, которые физически приходят на её множители.</div>`;
  const markup = `${note}<div class="conv-layout" data-view="conv">
    <section class="matrix-section">
      <h3>Входной регистр</h3><p>signed[${manifest.quant.bits - 1}:0] · ${hardware.input_shape.join("×")}</p>
      <div class="matrix-grid cols-${Math.min(inputCols, 4)}" role="grid">${matrixCells(inputValues, inputCols, {receptive, keys: inputKeys})}</div>
    </section>
    <div class="route-stack" aria-hidden="true"><span>window</span><strong>↘</strong></div>
    <section class="matrix-section">
      <h3>Constant bank / kernel</h3><p>${hardware.weight_shape.join("×")} · Q${manifest.quant.bits}.${manifest.quant.frac_bits}</p>
      <div class="matrix-grid cols-${Math.min(kernelCols, 4)}" role="grid">${matrixCells(kernel, kernelCols)}</div>
    </section>
    <span class="matrix-operator" aria-hidden="true">⊛</span>
    <section class="matrix-section">
      <h3>Карта выхода</h3><p>${hardware.output_shape.join("×")} · выбрано [${outIndex.join(",")}]</p>
      <div class="matrix-grid cols-${Math.min(outputCols, 4)}" role="grid">${matrixCells(outputValues, outputCols, {clickable: true, selected})}</div>
      <button class="canvas-primary conv-action" type="button" data-action="open-mac" data-block="${h(block.id)}" data-output="${selected}">Раскрыть этот MAC →</button>
    </section>
  </div><svg id="circuit-root" width="1" height="1" aria-hidden="true"></svg>`;
  setCanvas(markup, 860);
  updateCanvasHeader(
    "Модель / Conv2d / output map",
    `${block.name}: окно → kernel → выход`,
    `${block.params?.macs || hardware.output_entry_count * (entry?.tap_count || 0)} MAC`,
    `Выбрано <strong>out[${h(outIndex.join(","))}] = ${h(entry?.output_raw)}</strong> · нажмите другую выходную ячейку для смены окна`
  );
}

function nodeStateClass(requiredStep) {
  if (state.macStep > requiredStep) return "done-node";
  if (state.macStep === requiredStep) return "active-node";
  return "";
}

function wireStateClass(requiredStep, value = null) {
  const progress = state.macStep >= requiredStep ? "active" : "muted";
  return `${progress}${Number(value) < 0 ? " negative" : Number(value) > 0 && requiredStep > 0 ? " positive" : ""}`;
}

function renderMac() {
  const block = byId.get(state.route.id) || mainBlocks.find((item) => ["conv2d", "linear"].includes(item.kind));
  if (!block) return renderOverview();
  state.selectedId = block.id;
  state.selectedOutput = state.route.output ?? state.selectedOutput ?? 0;
  const step = traceStep(block.id);
  const entry = selectedHardwareEntry(block.id, state.selectedOutput);
  if (!step?.hardware || !entry) return renderDataUnavailable(block, "MAC-трассировка недоступна для выбранного узла.");
  const taps = entry.taps || [];
  const rowGap = 84;
  const top = 84;
  const height = Math.max(530, top + taps.length * rowGap + 100);
  const width = 1160;
  const rows = taps.map((tap, index) => {
    const y = top + index * rowGap;
    const inputSub = state.mode === "engineer" ? signedHex(tap.input_raw, manifest.quant.bits) : decoded(tap.input_raw);
    const weightSub = state.mode === "engineer" ? signedHex(tap.weight_raw, manifest.quant.bits) : decoded(tap.weight_raw);
    const productSub = state.mode === "engineer" ? signedHex(tap.product_raw, 32) : decoded(tap.product_raw, manifest.quant.frac_bits);
    const accSub = state.mode === "engineer" ? signedHex(tap.accumulator_raw, manifest.quant.acc_bits) : decoded(tap.accumulator_raw, manifest.quant.frac_bits);
    const previousY = index ? top + (index - 1) * rowGap : y - 45;
    const previousX = index ? 700 : 475;
    const previousLabel = index ? "acc" : "bias << frac";
    return `
      <g class="schematic-node ${nodeStateClass(0)}" data-signal="input ${tap.input_index.join(".")} · raw ${tap.input_raw}">
        <rect x="25" y="${y - 25}" width="92" height="50" rx="7"></rect><text x="37" y="${y - 5}">in ${h(tap.input_raw)}</text><text class="subtext" x="37" y="${y + 13}">${h(inputSub)}</text>
      </g>
      <g class="schematic-node kind-memory ${nodeStateClass(0)}" data-signal="weight ${tap.weight_index.join(".")} · raw ${tap.weight_raw}">
        <rect x="145" y="${y - 25}" width="92" height="50" rx="7"></rect><text x="157" y="${y - 5}">w ${h(tap.weight_raw)}</text><text class="subtext" x="157" y="${y + 13}">${h(weightSub)}</text>
      </g>
      <path class="schematic-wire ${wireStateClass(0)}" d="M117,${y} L266,${y}"></path>
      <path class="schematic-wire param ${state.macStep >= 0 ? "active" : "muted"}" d="M237,${y} L266,${y}"></path>
      <g class="schematic-node ${nodeStateClass(1)}"><circle cx="287" cy="${y}" r="21"></circle><text class="operator-symbol" x="287" y="${y}">×</text></g>
      <path class="schematic-wire ${wireStateClass(1, tap.product_raw)}" d="M308,${y} L342,${y}"></path>
      <g class="schematic-node ${nodeStateClass(1)}" data-signal="product raw ${tap.product_raw}">
        <rect x="342" y="${y - 25}" width="122" height="50" rx="7"></rect><text x="354" y="${y - 5}">prod ${h(tap.product_raw)}</text><text class="subtext" x="354" y="${y + 13}">${h(productSub)}</text>
      </g>
      <path class="schematic-wire ${wireStateClass(2, tap.product_raw)}" d="M464,${y} L532,${y}"></path>
      ${index === 0 ? `<g class="schematic-node ${nodeStateClass(2)}"><rect x="430" y="${y - 67}" width="92" height="30" rx="6"></rect><text class="subtext" x="441" y="${y - 48}">${previousLabel} ${entry.initial_accumulator_raw}</text></g>` : ""}
      <path class="schematic-wire ${wireStateClass(2)}" d="M${previousX},${previousY + (index ? 25 : 0)} L${previousX},${y - 34} L552,${y - 34} L552,${y - 21}"></path>
      <g class="schematic-node ${nodeStateClass(index < taps.length - 1 ? 2 : 3)}"><circle cx="552" cy="${y}" r="21"></circle><text class="operator-symbol" x="552" y="${y}">+</text></g>
      <path class="schematic-wire ${wireStateClass(index < taps.length - 1 ? 2 : 3, tap.accumulator_raw)}" d="M573,${y} L595,${y}"></path>
      <g class="schematic-node ${nodeStateClass(index < taps.length - 1 ? 2 : 3)}" data-signal="accumulator raw ${tap.accumulator_raw}">
        <rect x="595" y="${y - 25}" width="105" height="50" rx="7"></rect><text x="607" y="${y - 5}">Σ ${h(tap.accumulator_raw)}</text><text class="subtext" x="607" y="${y + 13}">${h(accSub)}</text>
      </g>`;
  }).join("");
  const finalY = top + Math.max(0, taps.length - 1) * rowGap;
  const finalPath = `
    <path class="schematic-wire ${wireStateClass(4)}" d="M700,${finalY} L750,${finalY}"></path>
    <g class="schematic-node ${nodeStateClass(4)}"><rect x="750" y="${finalY - 31}" width="118" height="62" rx="9"></rect><text x="766" y="${finalY - 7}">&gt;&gt;&gt; ${manifest.quant.frac_bits}</text><text class="subtext" x="766" y="${finalY + 14}">${h(entry.accumulator_raw)} → ${h(entry.shifted_raw)}</text></g>
    <path class="schematic-wire ${wireStateClass(5)}" d="M868,${finalY} L900,${finalY}"></path>
    <g class="schematic-node ${nodeStateClass(5)}"><rect x="900" y="${finalY - 31}" width="110" height="62" rx="9"></rect><text x="916" y="${finalY - 7}">SAT s${manifest.quant.bits}</text><text class="subtext" x="916" y="${finalY + 14}">${entry.saturation_flag ? "clamped" : "unchanged"}</text></g>
    <path class="schematic-wire ${wireStateClass(5)}" d="M1010,${finalY} L1035,${finalY}"></path>
    <g class="schematic-node kind-output ${nodeStateClass(5)}"><rect x="1035" y="${finalY - 31}" width="100" height="62" rx="9"></rect><text class="value-positive" x="1051" y="${finalY - 7}">out ${h(entry.output_raw)}</text><text class="subtext" x="1051" y="${finalY + 14}">${h(decoded(entry.output_raw))}</text></g>`;
  const note = `<div class="board-note"><strong>Один аппаратный MAC, без магии.</strong> Каждый tap проходит через ×, затем входит в упорядоченный аккумулятор. После суммы выполняются арифметический сдвиг и signed saturation.</div>`;
  const title = block.kind === "conv2d" ? `out[${entry.index.join(",")}]` : `neuron[${entry.index.join(",")}]`;
  setCanvas(`${note}<svg id="circuit-root" viewBox="0 0 ${width} ${height}" data-view="mac" aria-label="Внутренняя структура MAC">
    <text class="schematic-overline" x="25" y="55">sources · signed[${manifest.quant.bits - 1}:0]</text>
    <text class="schematic-overline" x="268" y="55">multipliers · signed[${manifest.quant.bits * 2 - 1}:0]</text>
    <text class="schematic-overline" x="520" y="55">ordered accumulator · signed[${manifest.quant.acc_bits - 1}:0]</text>
    ${rows}${finalPath}</svg>`, width);
  updateCanvasHeader(
    `${block.kind === "conv2d" ? "Conv2d" : "Linear"} / MAC / ${title}`,
    state.mode === "engineer" ? "MAC · инженерные разрядности" : "MAC · пошаговый вычислительный путь",
    `${entry.tap_count} × · ${entry.tap_count} + · Q${manifest.quant.bits}.${manifest.quant.frac_bits}`,
    state.macStep < 5 ? `Шаг <strong>${state.macStep + 1}/6</strong> · нажмите ▶ или Вперёд` : `<strong>${entry.accumulator_raw} &gt;&gt;&gt; ${manifest.quant.frac_bits} = ${entry.output_raw}</strong> · ${entry.saturation_flag ? "сработало насыщение" : "без насыщения"}`
  );
}

function renderLinear() {
  const block = byId.get(state.route.id) || mainBlocks.find((item) => item.kind === "linear");
  if (!block) return renderOverview();
  state.selectedId = block.id;
  const step = traceStep(block.id);
  const hardware = step?.hardware;
  if (!hardware) return renderDataUnavailable(block, "Детальная Linear-трассировка отсутствует.");
  const inputs = summaryValues(step.input);
  const logits = summaryValues(step.output);
  const winner = currentTrace().class_id;
  const inCount = Number(hardware.weight_shape?.[1] || inputs.length);
  const rowButtons = (hardware.output_entries || []).map((entry) => {
    const weights = (entry.taps || []).map((tap) => tap.weight_raw);
    return `<button class="mac-row-button ${Number(entry.flat_index) === Number(winner) ? "winner" : ""}" type="button"
      data-action="open-mac" data-block="${h(block.id)}" data-output="${entry.flat_index}">
      <strong>${entry.flat_index}</strong><code>[${h(weights.join(", "))}] · x</code><span>${h(entry.output_raw)}</span>
    </button>`;
  }).join("");
  const maxAbs = Math.max(1, ...logits.map((value) => Math.abs(Number(value))));
  const bars = logits.map((value, index) => `
    <div class="logit-row ${index === winner ? "winner" : ""}"><span>c${index}</span><div class="logit-track"><div class="logit-fill" style="width:${Math.max(2, Math.abs(Number(value)) / maxAbs * 100)}%"></div></div><strong>${h(value)}</strong></div>`).join("");
  const activations = inputs.map((value) => `<span class="activation-pill ${valueClass(value)}">${h(value)}</span>`).join("");
  const note = `<div class="board-note"><strong>Linear — это группа MAC.</strong> Каждая строка constant matrix считает один логит. Argmax сравнивает их и выдаёт номер максимального.</div>`;
  const markup = `${note}<div class="linear-layout" data-view="linear">
    <section class="linear-card"><h3>ReLU / Flatten</h3><p>activation bus · ${inputs.length} × signed8</p><div class="activation-vector">${activations}</div></section>
    <span class="matrix-operator" aria-hidden="true">×</span>
    <section class="linear-card" style="min-width:280px"><h3>Constant matrix + MAC rows</h3><p>${hardware.weight_shape.join("×")} · нажмите строку</p><div class="mac-row-list">${rowButtons}</div></section>
    <section class="linear-card"><h3>Логиты</h3><p>raw / Q${manifest.quant.bits}.${manifest.quant.frac_bits}</p><div class="logit-bars">${bars}</div></section>
    <section class="argmax-stack"><div class="compare-node">max</div><span class="schematic-label">tie → first index</span><div class="winner-readout">class_id ${h(winner)}</div></section>
  </div><svg id="circuit-root" width="1" height="1" aria-hidden="true"></svg>`;
  setCanvas(markup, 980);
  updateCanvasHeader(
    "Модель / Linear / logits / Argmax",
    `${block.name}: ${inCount} входов → ${logits.length} классов`,
    `${block.params?.macs || inCount * logits.length} MAC · class ${winner}`,
    `Победил <strong>class ${winner}</strong> со значением <strong>${h(logits[winner])}</strong> · нажмите строку матрицы, чтобы раскрыть MAC`
  );
}

function inferredCells(block) {
  if (["conv2d", "linear"].includes(block.kind)) return ["$mul", "$add", "$memrd", "$shiftx"];
  if (block.kind === "relu") return ["$lt", "$mux"];
  if (block.kind === "argmax") return ["$gt", "$mux"];
  if (block.kind === "flatten") return ["$slice"];
  return ["$wire"];
}

function renderYosys() {
  state.selectedId = "yosys";
  const synth = manifest.build?.synthesis || manifest.build?.status?.synthesis || {};
  const metrics = synth.metrics || {};
  const hasYosys = synth.status === "passed" || Boolean(metrics.cells);
  const clusters = mainBlocks.filter((block) => !["input", "output"].includes(block.kind)).map((block) => {
    const cells = inferredCells(block);
    return `<section class="yosys-cluster"><h3>${h(block.name)}</h3><p>${h(block.kind)} · ${hasYosys ? "coarse cells" : "ожидаемые cell classes"}</p>
      <div class="cell-grid">${cells.map((cell, index) => `<button class="yosys-cell ${index === 0 && ["conv2d","linear"].includes(block.kind) ? "active" : ""}" type="button"
        data-action="open-mapping" data-block="${h(block.id)}" data-cell="${h(cell)}"><strong>${h(cell)}</strong><span>${hasYosys ? "coarse netlist" : "RTL inference"}</span></button>`).join("")}</div>
    </section>`;
  }).join("");
  const warning = hasYosys
    ? "Coarse netlist после prep — логические ячейки, не FPGA placement и не разводка платы."
    : "Yosys в этой среде не найден. Ниже показана честная проекция ожидаемых классов ячеек из RTL; точные cell IDs появятся после synth с netlist JSON.";
  setCanvas(`<div class="yosys-warning">${h(warning)}</div><div class="yosys-layout" data-view="yosys">${clusters}</div><svg id="circuit-root" width="1" height="1" aria-hidden="true"></svg>`, 900);
  updateCanvasHeader(
    "Synthesis / Yosys coarse",
    hasYosys ? "Coarse netlist · modules, cells, nets" : "Yosys недоступен · логическая проекция",
    hasYosys ? `${metrics.cells || 0} cells · ${metrics.wires || 0} wires` : "RTL logical продолжает работать",
    hasYosys ? "Нажмите ячейку, чтобы увидеть сопоставление с logical operator" : "Жёлтое предупреждение отделяет <strong>подтверждённые данные</strong> от ожидаемой структуры"
  );
}

function defaultSourceForBlock(block) {
  if (!block) return Object.keys(sourcePreviews)[0] || "";
  const preferred = block.kind === "conv2d" ? "conv2d_comb.sv"
    : block.kind === "linear" ? "linear_comb.sv"
    : block.kind === "relu" ? "relu.sv"
    : block.kind === "argmax" ? "argmax.sv"
    : "top.sv";
  return sourcePreviews[preferred] ? preferred : Object.keys(sourcePreviews)[0] || "";
}

function renderMapping() {
  const block = byId.get(state.route.id) || mainBlocks.find((item) => ["conv2d", "linear"].includes(item.kind));
  if (!block) return renderYosys();
  state.selectedId = block.id;
  const cell = state.route.cell || inferredCells(block)[0];
  const synth = manifest.build?.synthesis || manifest.build?.status?.synthesis || {};
  const hasYosys = synth.status === "passed" || Boolean(synth.metrics?.cells);
  const entry = selectedHardwareEntry(block.id, state.selectedOutput);
  const tap = entry?.taps?.at(-1);
  const mappingRows = tap ? [
    ["A", `signed[${manifest.quant.bits - 1}:0] = ${tap.input_raw}`],
    ["B", `signed[${manifest.quant.bits - 1}:0] = ${tap.weight_raw}`],
    ["Y", `signed[${manifest.quant.bits * 2 - 1}:0] = ${tap.product_raw}`],
  ] : [["signal_in", block.signal_in || "—"], ["signal_out", block.signal_out || "—"]];
  const source = defaultSourceForBlock(block);
  const rightTitle = hasYosys ? `Coarse cell ${cell}` : `Ожидаемый cell class ${cell}`;
  const certainty = hasYosys ? "coarse-class confirmed" : "provisional · inferred from RTL";
  const markup = `<div class="board-note"><strong>Сопоставление уровней.</strong> Слева — оператор из logical RTL, справа — coarse-класс. Физический DSP/LUT можно утверждать только после выбора target и place & route.</div>
    <div class="mapping-layout" data-view="mapping">
      <section class="mapping-card"><h3>Logical operator</h3><p>${h(block.name)} · ${h(block.kind)}</p>
        <div class="mapping-list">${mappingRows.map(([key,value]) => `<div class="mapping-row"><span>${h(key)}</span><code>${h(value)}</code></div>`).join("")}</div>
        <div class="inspector-actions"><button class="canvas-secondary" type="button" data-action="open-source" data-source="${h(source)}">Открыть RTL source</button></div>
      </section>
      <div class="mapping-bridge"><strong>${h(certainty)}</strong></div>
      <section class="mapping-card"><h3>${h(rightTitle)}</h3><p>${hasYosys ? "Yosys prep / coarse" : "точный cell ID пока отсутствует"}</p>
        <div class="mapping-list">
          <div class="mapping-row"><span>module</span><code>top</code></div>
          <div class="mapping-row"><span>source</span><code>${h(source || "generated RTL")}</code></div>
          <div class="mapping-row"><span>mapping</span><code>${h(certainty)}</code></div>
          <div class="mapping-row"><span>target</span><code>none</code></div>
        </div>
      </section>
    </div><svg id="circuit-root" width="1" height="1" aria-hidden="true"></svg>`;
  setCanvas(markup, 900);
  updateCanvasHeader(
    "Logical RTL / source map / Yosys",
    `${block.name} ↔ ${cell}`,
    certainty,
    hasYosys ? "Сопоставление ограничено coarse-уровнем" : "Пунктирная логика означает: <strong>это ожидаемый класс, не подтверждённый cell ID</strong>"
  );
}

function renderDataUnavailable(block, message) {
  setCanvas(`<div class="empty-state-card" data-view="empty"><div class="empty-state-icon">∿</div><h3>${h(block.label)}</h3><p>${h(message)}</p><button class="canvas-primary" type="button" data-action="route" data-route="#overview">Вернуться к модели</button></div><svg id="circuit-root" width="1" height="1" aria-hidden="true"></svg>`, 760);
  updateCanvasHeader("Данные", "Подробная трассировка недоступна", block.name, "Логический обзор остаётся доступен");
}

function renderView() {
  state.route = parseRoute();
  if (state.route.name === "overview") renderOverview();
  else if (state.route.name === "conv") renderConv();
  else if (state.route.name === "mac") renderMac();
  else if (state.route.name === "linear") renderLinear();
  else if (state.route.name === "yosys") renderYosys();
  else if (state.route.name === "mapping") renderMapping();
}

function kvRows(rows) {
  return `<dl class="kv">${rows.filter(([,value]) => value !== undefined && value !== null).map(([key,value]) => `<dt>${h(key)}</dt><dd>${value?.html ? value.html : h(value)}</dd>`).join("")}</dl>`;
}

function blockInspector(block) {
  const step = traceStep(block.id);
  const rows = [
    ["имя", block.name || block.id],
    ["тип", kindLabels[block.kind] || block.kind],
    ["вход", block.signal_in],
    ["выход", block.signal_out],
    ["размер", block.in_features !== undefined ? `${block.in_features} → ${block.out_features}` : block.features],
    ...Object.entries(block.params || {}),
  ];
  if (step) rows.push(["значения", {html: `<code>${h(summaryText(step.output))}</code>`}]);
  return `${kvRows(rows)}${["conv2d","linear"].includes(block.kind) ? `<div class="inspector-actions"><button class="primary-button" type="button" data-action="route" data-route="${h(routeForBlock(block))}">Исследовать структуру →</button></div>` : ""}`;
}

function renderInspector() {
  const route = state.route;
  const trace = currentTrace();
  const block = route.id ? byId.get(route.id) : byId.get(state.selectedId) || byId.get("input");
  const selectedLabel = document.getElementById("selected-label");
  selectedLabel.textContent = route.name === "yosys" ? "coarse" : block?.id || route.name;
  let html = "";
  if (route.name === "overview") {
    const target = block || byId.get("input");
    const metrics = manifest.build?.metrics || {};
    const synth = manifest.build?.synthesis || manifest.build?.status?.synthesis || {};
    html = `<h2 class="detail-title">${h(target.label)}</h2><p class="detail-kind">${h(kindLabels[target.kind] || target.kind)}</p>
      ${blockInspector(target)}
      <section class="inspector-section"><h3 class="inspector-section-title">Итог модели</h3>${kvRows([
        ["логиты", {html:`<code>${h(summaryText(trace.logits))}</code>`}],
        ["class_id", trace.class_id], ["ожидалось", trace.expected_class], ["проверка", trace.matched ? "MATCH" : "MISMATCH"],
      ])}</section>
      <section class="inspector-section"><h3 class="inspector-section-title">Ресурсы синтеза</h3>${kvRows([
        ["параметры", metrics.parameters], ["MAC", metrics.macs], ["Yosys cells", synth.metrics?.cells ?? "нет данных"],
      ])}</section>
      <section class="inspector-section"><h3 class="inspector-section-title">Артефакты</h3>
        <div class="artifact-mini">${(manifest.build?.artifacts || []).slice(0, 6).map((item) => `<a href="${encodeURI(item.file)}"><span>${h(item.file)}</span><small>${h(item.size_bytes)} B</small></a>`).join("") || '<span class="empty-note">нет файлов</span>'}</div>
      </section>`;
  } else if (route.name === "conv") {
    const entry = selectedHardwareEntry(block.id, state.selectedOutput);
    html = `<h2 class="detail-title">${h(block.label)}</h2><p class="detail-kind">output map · selected MAC</p>
      ${kvRows([
        ["ядро", block.params?.kernel_size], ["stride", block.params?.stride], ["padding", block.params?.padding],
        ["выход", block.params?.output_shape], ["всего MAC", block.params?.macs], ["ячейка", entry ? `[${entry.index.join(",")}]` : "—"], ["raw", entry?.output_raw], ["decoded", entry ? decoded(entry.output_raw) : "—"],
      ])}
      <p class="inspector-callout">Подсвеченное окно — физические операнды выбранного MAC. Оно меняется вместе с ячейкой выхода.</p>
      <div class="inspector-actions"><button class="primary-button" type="button" data-action="open-mac" data-block="${h(block.id)}" data-output="${entry?.flat_index || 0}">Раскрыть этот MAC →</button></div>`;
  } else if (route.name === "mac") {
    const entry = selectedHardwareEntry(block.id, route.output);
    const taps = entry?.taps || [];
    const equation = taps.map((tap) => `${tap.input_raw}×${tap.weight_raw}`).join(" + ");
    html = `<h2 class="detail-title">MAC ${h(entry?.index?.join(",") || "")}</h2><p class="detail-kind">${state.mode === "engineer" ? "engineering view" : "fixed-point explanation"}</p>
      <div class="equation">${h(equation)}<br>= <strong>${h(entry?.accumulator_raw)}</strong> &gt;&gt;&gt; ${manifest.quant.frac_bits}<br>= <strong>${h(entry?.output_raw)}</strong> / ${h(decoded(entry?.output_raw || 0))}</div>
      ${kvRows(state.mode === "engineer" ? [
        ["Circuit ID", `${block.id}.mac[${entry?.index?.join(",")}]`], ["ACC raw", entry?.accumulator_raw], ["ACC hex", signedHex(entry?.accumulator_raw || 0, manifest.quant.acc_bits)],
        ["input / weight", `signed[${manifest.quant.bits - 1}:0]`], ["product", `signed[${manifest.quant.bits * 2 - 1}:0]`], ["accumulator", `signed[${manifest.quant.acc_bits - 1}:0]`],
        ["after shift", `${entry?.shifted_raw} / ${signedHex(entry?.shifted_raw || 0, manifest.quant.bits)}`],
      ] : [["операндов", entry?.tap_count], ["частичных сумм", entry?.tap_count], ["ACC", entry?.accumulator_raw], ["после >>>", entry?.shifted_raw], ["результат", entry?.output_raw]])}
      <div class="badge-row"><span class="status-badge ${entry?.saturation_flag ? "warn" : "good"}">${entry?.saturation_flag ? "Сработало насыщение" : "Без насыщения"}</span><span class="status-badge good">Без переполнения ACC</span></div>
      <p class="inspector-callout">Шаг ${state.macStep + 1}: ${h(timelineLabels.mac[state.macStep])}. Зелёный путь показывает уже вычисленную часть схемы.</p>
      <div class="inspector-actions"><button class="secondary-button" type="button" data-action="open-source" data-source="${h(defaultSourceForBlock(block))}">Открыть RTL source</button>${state.macStep === 5 ? `<button class="primary-button" type="button" data-action="route" data-route="${h(mainBlocks.some((item)=>item.kind==="linear") ? routeHash("linear", mainBlocks.find((item)=>item.kind==="linear").id) : "#overview")}">К выходу модели →</button>` : ""}</div>`;
  } else if (route.name === "linear") {
    const step = traceStep(block.id);
    const logits = summaryValues(step?.output);
    html = `<h2 class="detail-title">Linear + Argmax</h2><p class="detail-kind">матрица MAC и comparator tree</p>
      <div class="logit-table">${logits.map((value,index) => `<div class="logit-table-row ${index === trace.class_id ? "winner" : ""}"><span>class ${index}</span><code>${h(value)}</code><strong>${index === trace.class_id ? "← max" : ""}</strong></div>`).join("")}</div>
      <p class="inspector-callout">При равенстве выигрывает первый индекс: используется последовательное сравнение.</p>
      ${kvRows([["входов", block.in_features], ["выходов", block.out_features], ["MAC", block.params?.macs], ["class_id", trace.class_id]])}`;
  } else if (route.name === "yosys") {
    const synth = manifest.build?.synthesis || manifest.build?.status?.synthesis || {};
    html = `<h2 class="detail-title">Yosys coarse</h2><p class="detail-kind">prep · logical cells</p>
      ${kvRows([["module", "top"], ["stage", "coarse"], ["driver", "prep"], ["target", "none"], ["cells", synth.metrics?.cells ?? "нет данных"], ["wires", synth.metrics?.wires ?? "нет данных"], ["status", statusMessage(synth.message)]])}
      <p class="inspector-callout warning">Coarse cells не являются FPGA placement. Для макета распайки позже понадобится target-specific mapping и pinout.</p>
      <div class="inspector-actions"><button class="secondary-button" type="button" data-action="route" data-route="#overview">Вернуться к RTL logical</button></div>`;
  } else if (route.name === "mapping") {
    const synth = manifest.build?.synthesis || manifest.build?.status?.synthesis || {};
    const confirmed = synth.status === "passed" || Boolean(synth.metrics?.cells);
    html = `<h2 class="detail-title">Logical ↔ ${h(route.cell || "$cell")}</h2><p class="detail-kind">source map</p>
      ${kvRows([["Logical", block.id], ["cell class", route.cell || inferredCells(block)[0]], ["mapping", confirmed ? "coarse confirmed" : "provisional"], ["stage", confirmed ? "Yosys coarse" : "RTL inference"], ["target", "none"]])}
      <p class="inspector-callout ${confirmed ? "" : "warning"}">${confirmed ? "Связь подтверждена на coarse-уровне." : "Yosys netlist отсутствует: класс ячейки выведен из RTL, точный ID не придуман."}</p>
      <div class="inspector-actions"><button class="secondary-button" type="button" data-action="open-source" data-source="${h(defaultSourceForBlock(block))}">Открыть RTL source</button><button class="primary-button" type="button" data-action="route" data-route="#yosys">Назад к netlist</button></div>`;
  }
  document.getElementById("inspector-body").innerHTML = html;
}

function routeProgress() {
  if (state.route.name === "mac") return state.macStep;
  if (state.route.name === "conv") return 1;
  if (state.route.name === "linear") return 4;
  if (state.route.name === "yosys") return 2;
  if (state.route.name === "mapping") return 4;
  const block = byId.get(state.selectedId);
  const kindIndex = {input:0, conv2d:1, relu:2, flatten:2, linear:3, argmax:4, output:5};
  return kindIndex[block?.kind] ?? 5;
}

function renderTransport() {
  const labels = timelineLabels[state.route.name] || timelineLabels.overview;
  const progress = routeProgress();
  document.getElementById("step-label").textContent = `Шаг ${progress + 1}/6`;
  const trace = currentTrace();
  document.getElementById("trace-status").textContent = `вектор ${trace.vector_index} · expected ${trace.expected_class ?? "—"} · ${trace.matched ? "MATCH" : "MISMATCH"}`;
  document.getElementById("timeline").innerHTML = labels.map((label,index) => `
    <button class="timeline-step ${index < progress ? "done" : ""} ${index === progress ? "active" : ""}" type="button" data-action="timeline-step" data-step="${index}" aria-label="Шаг ${index + 1}: ${h(label)}"><i></i><span>${h(label)}</span></button>`).join("");
  const playButton = document.querySelector('[data-action="play"]');
  playButton.textContent = state.playing ? "❚❚" : "▶";
  playButton.setAttribute("aria-label", state.playing ? "Пауза" : "Запустить трассировку");
  const next = document.querySelector('[data-action="next-step"]');
  next.disabled = state.route.name === "mac" && state.macStep >= 5;
  const back = document.querySelector('[data-action="back"]');
  back.disabled = state.route.name === "overview" && activeBlockId() === "input";
  document.getElementById("result-readout").innerHTML = `<span class="logits-compact">logits ${h(summaryText(trace.logits, 6))}</span><div class="class-chip"><span>class_id</span><strong>${h(trace.class_id)} ${trace.matched ? "✓" : "!"}</strong></div>`;
}

function renderVectorList() {
  document.getElementById("vector-list").innerHTML = traceList.map((trace) => `
    <button class="vector-option ${Number(trace.vector_index) === Number(state.vectorIndex) ? "active" : ""}" type="button" data-action="select-vector" data-vector="${trace.vector_index}">
      <span class="vector-index">#${String(trace.vector_index).padStart(2,"0")}</span>
      <span><strong>${h(summaryText(trace.input, 9))}</strong><span>range ${h(trace.input.min)}…${h(trace.input.max)}</span></span>
      <strong class="vector-result">class ${h(trace.class_id)} ${trace.matched ? "✓" : "!"}</strong>
    </button>`).join("");
}

function renderSourceDrawer() {
  const files = Object.keys(sourcePreviews);
  const active = sourcePreviews[state.sourceFile] ? state.sourceFile : files[0];
  state.sourceFile = active || "";
  document.getElementById("source-tabs").innerHTML = files.length
    ? files.map((file) => `<button class="source-tab ${file === active ? "active" : ""}" type="button" data-action="select-source" data-source="${h(file)}">${h(file)}</button>`).join("")
    : `<span class="status-badge warn">RTL preview не встроен</span>`;
  const preview = sourcePreviews[active];
  document.getElementById("source-title").textContent = active || "Исходный модуль";
  document.getElementById("source-code").textContent = preview
    ? `${preview.text}${preview.truncated ? `\n\n// … показано ${preview.shown_lines} из ${preview.line_count} строк` : ""}`
    : "RTL source preview недоступен. Откройте соответствующий .sv рядом с visualization.html.";
}

function openSource(file) {
  if (file) state.sourceFile = file;
  renderSourceDrawer();
  document.getElementById("source-drawer").classList.add("open");
  document.getElementById("source-drawer").setAttribute("aria-hidden", "false");
  document.getElementById("drawer-backdrop").hidden = false;
}

function closeSource() {
  document.getElementById("source-drawer").classList.remove("open");
  document.getElementById("source-drawer").setAttribute("aria-hidden", "true");
  document.getElementById("drawer-backdrop").hidden = true;
}

function openVectors() {
  renderVectorList();
  const dialog = document.getElementById("vector-dialog");
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function closeVectors() {
  const dialog = document.getElementById("vector-dialog");
  if (typeof dialog.close === "function" && dialog.open) dialog.close();
  else dialog.removeAttribute("open");
}

function announce(text) {
  document.getElementById("a11y-status").textContent = text;
}

function stopPlayback() {
  if (playTimer) window.clearInterval(playTimer);
  playTimer = null;
  state.playing = false;
}

function stepForward() {
  if (state.route.name === "mac") {
    state.macStep = clamp(state.macStep + 1, 0, 5);
    if (state.macStep >= 5) stopPlayback();
    renderView();
    renderInspector();
    renderTransport();
    announce(`Шаг ${state.macStep + 1}: ${timelineLabels.mac[state.macStep]}`);
    return;
  }
  if (state.route.name === "overview") {
    const index = Math.max(0, mainBlocks.findIndex((block) => block.id === state.selectedId));
    const next = mainBlocks[Math.min(index + 1, mainBlocks.length - 1)];
    state.selectedId = next.id;
    renderAll();
    return;
  }
  if (state.route.name === "conv") {
    const entry = selectedHardwareEntry(state.route.id, state.selectedOutput);
    navigate(routeHash("mac", state.route.id, entry?.flat_index || 0));
  }
}

function stepBackward() {
  stopPlayback();
  if (state.route.name === "mac") {
    if (state.macStep > 0) {
      state.macStep -= 1;
      renderView();
      renderInspector();
      renderTransport();
      announce(`Шаг ${state.macStep + 1}: ${timelineLabels.mac[state.macStep]}`);
      return;
    }
    const block = byId.get(state.route.id);
    navigate(block?.kind === "linear"
      ? routeHash("linear", block.id)
      : routeHash("conv", block?.id || state.route.id));
    return;
  }
  if (state.route.name === "mapping") {
    navigate("#yosys");
    return;
  }
  if (["conv", "linear", "yosys"].includes(state.route.name)) {
    if (state.route.id) state.selectedId = state.route.id;
    navigate("#overview");
    return;
  }
  const index = mainBlocks.findIndex((block) => block.id === state.selectedId);
  const previous = mainBlocks[Math.max(0, index - 1)] || mainBlocks[0];
  state.selectedId = previous.id;
  renderAll();
  announce(`Выбран предыдущий блок: ${kindLabels[previous.kind] || previous.label}`);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    renderTransport();
    return;
  }
  if (state.route.name !== "mac") {
    if (state.route.name === "conv") stepForward();
    return;
  }
  if (state.macStep >= 5) state.macStep = 0;
  state.playing = true;
  renderAll();
  playTimer = window.setInterval(stepForward, 720);
}

function setTimelineStep(step) {
  const value = clamp(Number(step), 0, 5);
  if (state.route.name === "mac") {
    stopPlayback();
    state.macStep = value;
    renderAll();
  } else if (state.route.name === "overview") {
    const targets = ["input", mainBlocks.find((b)=>b.kind==="conv2d")?.id, mainBlocks.find((b)=>b.kind==="relu")?.id, mainBlocks.find((b)=>b.kind==="linear")?.id, mainBlocks.find((b)=>b.kind==="argmax")?.id, "output"].filter(Boolean);
    state.selectedId = targets[value] || targets.at(-1);
    renderAll();
  }
}

function renderAll() {
  state.route = parseRoute();
  state.representation = ["yosys", "mapping"].includes(state.route.name) ? "yosys" : "rtl";
  document.body.classList.toggle("inspector-open", state.inspectorOpen);
  renderReadouts();
  renderStatuses();
  renderBreadcrumbs();
  renderLayerTree();
  renderView();
  renderInspector();
  renderTransport();
  renderVectorList();
  renderSourceDrawer();
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  if (action === "route") {
    if (target.dataset.block) state.selectedId = target.dataset.block;
    navigate(target.dataset.route || "#overview");
  }
  else if (action === "set-mode") {
    state.mode = target.dataset.value;
    renderAll();
  } else if (action === "set-representation") {
    if (target.dataset.value === "yosys") navigate("#yosys");
    else navigate(state.route.name === "mapping" ? routeHash("mac", state.route.id, state.selectedOutput || 0) : "#overview");
  } else if (action === "open-vectors") openVectors();
  else if (action === "select-vector") {
    state.vectorIndex = Number(target.dataset.vector);
    state.selectedOutput = null;
    state.macStep = state.route.name === "mac" ? 0 : 5;
    stopPlayback();
    closeVectors();
    renderAll();
    announce(`Выбран вектор ${state.vectorIndex}`);
  } else if (action === "select-output") {
    state.selectedOutput = Number(target.dataset.output);
    renderView();
    renderInspector();
  } else if (action === "open-mac") {
    state.macStep = 0;
    navigate(routeHash("mac", target.dataset.block, Number(target.dataset.output || 0)));
  } else if (action === "open-mapping") {
    navigate(routeHash("mapping", target.dataset.block, state.selectedOutput || 0, target.dataset.cell));
  } else if (action === "open-source") openSource(target.dataset.source);
  else if (action === "close-source") closeSource();
  else if (action === "select-source") {
    state.sourceFile = target.dataset.source;
    renderSourceDrawer();
  } else if (action === "zoom-in") {
    state.zoom = clamp(state.zoom + 0.1, 0.6, 1.6);
    renderReadouts(); renderView();
  } else if (action === "zoom-out") {
    state.zoom = clamp(state.zoom - 0.1, 0.6, 1.6);
    renderReadouts(); renderView();
  } else if (action === "fit") {
    const viewport = document.getElementById("circuit-viewport");
    const canvas = document.getElementById("visualization-canvas");
    const naturalWidth = Number.parseFloat(canvas.style.minWidth) || canvas.scrollWidth || 760;
    state.zoom = clamp((viewport.clientWidth - 24) / naturalWidth, 0.6, 1);
    renderReadouts(); renderView();
    document.getElementById("circuit-viewport").scrollTo({left: 0, top: 0, behavior: "smooth"});
  } else if (action === "toggle-inspector") {
    state.inspectorOpen = !state.inspectorOpen;
    document.body.classList.toggle("inspector-open", state.inspectorOpen);
  } else if (action === "back") {
    stepBackward();
  } else if (action === "play") togglePlayback();
  else if (action === "next-step") stepForward();
  else if (action === "timeline-step") setTimelineStep(target.dataset.step);
});

document.addEventListener("keydown", (event) => {
  const interactive = event.target.closest?.("button, a, input, select, textarea, [role=button]");
  if ((event.key === "Enter" || event.key === " ") && event.target.matches?.('[data-action]:not(button)')) {
    event.preventDefault();
    event.target.dispatchEvent(new MouseEvent("click", {bubbles: true}));
    return;
  }
  if (event.key === "Escape") {
    closeSource();
    closeVectors();
    state.inspectorOpen = false;
    document.body.classList.remove("inspector-open");
    return;
  }
  if (interactive) return;
  if (event.key === "ArrowRight") { event.preventDefault(); stepForward(); }
  if (event.key === "ArrowLeft") { event.preventDefault(); stepBackward(); }
  if (event.key === " ") { event.preventDefault(); togglePlayback(); }
});

document.addEventListener("pointerover", (event) => {
  const target = event.target.closest?.("[data-signal]");
  if (!target) return;
  const tooltip = document.getElementById("signal-tooltip");
  tooltip.textContent = target.dataset.signal;
  tooltip.hidden = false;
  const rect = target.getBoundingClientRect();
  tooltip.style.left = `${clamp(rect.left + rect.width / 2, 8, window.innerWidth - 270)}px`;
  tooltip.style.top = `${clamp(rect.top - 42, 8, window.innerHeight - 80)}px`;
});

document.addEventListener("pointerout", (event) => {
  if (event.target.closest?.("[data-signal]")) document.getElementById("signal-tooltip").hidden = true;
});

window.addEventListener("hashchange", () => {
  stopPlayback();
  state.route = parseRoute();
  state.macStep = state.route.name === "mac" ? 0 : 5;
  state.selectedOutput = state.route.output ?? state.selectedOutput;
  renderAll();
  document.getElementById("circuit-viewport").scrollTo({left: 0, top: 0});
});

renderAll();
"""
