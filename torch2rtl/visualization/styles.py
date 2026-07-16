STYLES = """
:root {
  --app: #eef2ec;
  --panel: #f7faf6;
  --raised: #ffffff;
  --board: #0b1511;
  --board-raised: #10201a;
  --board-soft: #162820;
  --ink: #142018;
  --muted: #67736b;
  --muted-2: #8a958e;
  --on-board: #eaf4ee;
  --on-board-muted: #91a79b;
  --line: #d5ded7;
  --board-line: #294138;
  --copper: #c77a3c;
  --active: #47e08a;
  --positive: #f2b84b;
  --negative: #5ca8ff;
  --danger: #ff5d6c;
  --violet: #9b7bff;
  --teal: #20bfa9;
  --blue: #478bff;
  --shadow: 0 18px 46px rgba(14, 28, 20, 0.10);
  --shadow-high: 0 28px 80px rgba(6, 17, 11, 0.24);
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --header-h: 72px;
  --context-h: 48px;
  --transport-h: 96px;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--app);
}

* { box-sizing: border-box; }

html,
body { height: 100%; }

body {
  margin: 0;
  overflow: hidden;
  background:
    radial-gradient(circle at 6% 2%, rgba(71, 224, 138, 0.10), transparent 24rem),
    radial-gradient(circle at 96% 94%, rgba(199, 122, 60, 0.10), transparent 30rem),
    var(--app);
}

button,
input,
output { font: inherit; }

button { color: inherit; }

button:focus-visible,
[tabindex]:focus-visible,
a:focus-visible {
  outline: 3px solid rgba(71, 224, 138, 0.68);
  outline-offset: 2px;
}

button:disabled { cursor: not-allowed; opacity: 0.42; }

.skip-link {
  position: fixed;
  z-index: 1000;
  top: 8px;
  left: 8px;
  transform: translateY(-160%);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  background: var(--ink);
  color: white;
  text-decoration: none;
}

.skip-link:focus { transform: translateY(0); }

.app-shell {
  height: 100vh;
  min-width: 320px;
  display: grid;
  grid-template-rows: var(--header-h) var(--context-h) auto minmax(0, 1fr) var(--transport-h);
}

.topbar {
  z-index: 20;
  min-width: 0;
  padding: 0 18px;
  border-bottom: 1px solid var(--line);
  display: grid;
  grid-template-columns: minmax(250px, 1fr) auto minmax(520px, 1.3fr);
  align-items: center;
  gap: 18px;
  background: rgba(247, 250, 246, 0.94);
  backdrop-filter: blur(16px);
}

.brand-lockup {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-mark {
  position: relative;
  flex: 0 0 38px;
  width: 38px;
  height: 38px;
  border: 1px solid #203a2d;
  border-radius: 11px;
  display: grid;
  grid-template-columns: repeat(2, 8px);
  place-content: center;
  gap: 5px;
  background: var(--board);
  box-shadow: inset 0 0 0 1px rgba(71, 224, 138, 0.07);
}

.brand-mark::before,
.brand-mark::after {
  content: "";
  position: absolute;
  background: var(--copper);
}

.brand-mark::before { width: 48px; height: 1px; top: 18px; left: -6px; }
.brand-mark::after { width: 1px; height: 48px; top: -6px; left: 18px; }

.brand-mark i {
  z-index: 1;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: var(--active);
  box-shadow: 0 0 9px rgba(71, 224, 138, 0.56);
}

.eyebrow,
.panel-kicker {
  margin: 0 0 3px;
  color: var(--muted);
  font-size: 10px;
  line-height: 1.2;
  font-weight: 760;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  overflow: hidden;
  color: var(--ink);
  font-size: 20px;
  line-height: 1.05;
  font-weight: 820;
  letter-spacing: -0.03em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

h1 span { color: var(--muted); font-weight: 620; }

.build-identity {
  min-width: 160px;
  padding: 7px 12px;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: var(--raised);
  text-align: center;
}

.build-identity strong,
.build-identity span { display: block; }
.build-identity strong { font-size: 12px; }
.build-identity span { margin-top: 1px; color: var(--muted); font-size: 10px; }

.topbar-actions {
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.quant-readout { display: flex; gap: 5px; }

.chip,
.status-badge,
.width-badge {
  min-height: 26px;
  padding: 5px 8px;
  border: 1px solid var(--line);
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(255, 255, 255, 0.74);
  color: var(--muted);
  font-size: 10px;
  font-weight: 760;
  line-height: 1;
  white-space: nowrap;
}

.chip.ok::before,
.status-badge.ok::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: var(--active);
  box-shadow: 0 0 0 3px rgba(71, 224, 138, 0.14);
}

.segmented {
  min-width: 0;
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  display: flex;
  gap: 2px;
  background: #e8ede8;
}

.segmented button {
  min-height: 30px;
  padding: 5px 9px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font-size: 10px;
  font-weight: 760;
  white-space: nowrap;
}

.segmented button.active {
  background: var(--raised);
  color: var(--ink);
  box-shadow: 0 2px 9px rgba(14, 28, 20, 0.09);
}

.context-bar {
  z-index: 18;
  min-width: 0;
  padding: 0 18px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  background: rgba(255, 255, 255, 0.76);
  backdrop-filter: blur(12px);
}

.breadcrumbs {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  overflow: hidden;
}

.breadcrumbs button,
.breadcrumbs span {
  border: 0;
  padding: 3px 2px;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}

.breadcrumbs button:hover { color: var(--ink); }
.breadcrumbs strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.breadcrumbs .crumb-separator { color: #aeb8b1; }

.context-tools { display: flex; align-items: center; gap: 7px; }

.context-button,
.zoom-control {
  min-height: 32px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--raised);
  color: var(--muted);
  font-size: 10px;
  font-weight: 730;
}

.context-button {
  padding: 6px 9px;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}

.context-button strong { color: var(--ink); }
.vector-trigger span { color: var(--muted-2); }

.zoom-control { padding: 2px; display: flex; align-items: center; }
.zoom-control button { width: 28px; height: 26px; border: 0; border-radius: 6px; background: transparent; cursor: pointer; }
.zoom-control button:hover { background: var(--app); }
.zoom-control output { min-width: 42px; text-align: center; }
.mobile-inspector-button { display: none; }

.status-strip {
  min-height: 34px;
  padding: 5px 18px 0;
  display: flex;
  align-items: center;
  gap: 7px;
  overflow: hidden;
}

.status-item {
  min-width: 0;
  min-height: 27px;
  padding: 4px 8px;
  border: 1px solid var(--line);
  border-radius: 999px;
  display: flex;
  align-items: center;
  gap: 6px;
  background: rgba(255, 255, 255, 0.70);
  color: var(--muted);
  font-size: 9px;
  white-space: nowrap;
}

.status-item::before { content: ""; width: 6px; height: 6px; border-radius: 999px; background: var(--muted-2); }
.status-item strong { color: var(--ink); font-size: inherit; text-transform: uppercase; }
.status-item span { max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
.status-item.status-ok::before { background: var(--active); box-shadow: 0 0 0 3px rgba(71,224,138,0.12); }
.status-item.status-failed::before { background: var(--danger); }
.status-item.status-pending::before { background: var(--positive); }

.workspace {
  min-width: 0;
  min-height: 0;
  padding: 10px 12px 12px;
  display: grid;
  grid-template-columns: 272px minmax(520px, 1fr) 352px;
  gap: 10px;
}

.panel {
  min-width: 0;
  min-height: 0;
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  background: var(--panel);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.panel-head,
.inspector-head,
.canvas-head {
  min-height: 56px;
  padding: 10px 13px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.panel-head strong,
.inspector-head strong,
.canvas-head strong { font-size: 13px; }
.panel-head > span,
.inspector-head > span { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; }

.layer-rail { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; }

.layer-tree {
  padding: 9px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  overflow: auto;
}

.layer-item {
  position: relative;
  width: 100%;
  min-height: 45px;
  padding: 7px 8px 7px 37px;
  border: 1px solid transparent;
  border-radius: 9px;
  display: block;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  text-align: left;
}

.layer-item::before {
  content: "";
  position: absolute;
  top: 50%;
  left: 13px;
  width: 10px;
  height: 10px;
  transform: translateY(-50%) rotate(45deg);
  border: 1px solid currentColor;
  border-radius: 2px;
}

.layer-item::after {
  content: "";
  position: absolute;
  top: -8px;
  bottom: 35px;
  left: 18px;
  width: 1px;
  background: var(--line);
}

.layer-item:first-child::after { display: none; }
.layer-item:hover { background: #edf2ed; color: var(--ink); }
.layer-item.active { border-color: #b7d8c2; background: #e7f4ea; color: #164e2d; }
.layer-item.kind-conv2d::before,
.layer-item.kind-linear::before { color: var(--positive); }
.layer-item.kind-relu::before { color: var(--teal); }
.layer-item.kind-argmax::before { color: var(--danger); }
.layer-item.kind-memory::before { color: var(--violet); }
.layer-item strong,
.layer-item span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.layer-item strong { color: currentColor; font-size: 11px; }
.layer-item span { margin-top: 2px; color: var(--muted-2); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; }

.rail-group-label {
  margin: 9px 7px 3px;
  color: var(--muted-2);
  font-size: 9px;
  font-weight: 760;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.rail-foot {
  padding: 11px 12px;
  border-top: 1px solid var(--line);
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  background: rgba(255,255,255,0.52);
}

.rail-stat { min-width: 0; }
.rail-stat strong,
.rail-stat span { display: block; }
.rail-stat strong { font-size: 12px; }
.rail-stat span { margin-top: 2px; color: var(--muted); font-size: 8px; text-transform: uppercase; }

.canvas-panel {
  border-color: #24382f;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  background: var(--board);
  box-shadow: var(--shadow-high);
}

.canvas-head {
  border-bottom-color: var(--board-line);
  background: rgba(255,255,255,0.025);
  color: var(--on-board);
}

.canvas-head .panel-kicker { color: var(--on-board-muted); }
.canvas-head-meta { display: flex; align-items: center; gap: 9px; color: var(--on-board-muted); font-size: 9px; }
.live-indicator { padding: 4px 7px; border: 1px solid var(--board-line); border-radius: 999px; }
.live-indicator i { display: inline-block; width: 6px; height: 6px; margin-right: 4px; border-radius: 999px; background: var(--active); box-shadow: 0 0 8px var(--active); }

.canvas-viewport {
  position: relative;
  min-height: 0;
  overflow: auto;
  overscroll-behavior: contain;
  background-color: var(--board);
  background-image:
    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
    linear-gradient(rgba(199,122,60,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(199,122,60,0.035) 1px, transparent 1px);
  background-size: 32px 32px, 32px 32px, 128px 128px, 128px 128px;
}

.visualization-canvas {
  position: relative;
  width: 100%;
  min-width: 760px;
  min-height: 100%;
  transform: scale(var(--canvas-zoom, 1));
  transform-origin: top left;
  transition: transform 160ms ease;
}

.visualization-canvas > svg {
  display: block;
  width: 100%;
  min-width: 760px;
  height: 100%;
  min-height: 500px;
  overflow: visible;
}

.canvas-hint {
  position: sticky;
  z-index: 5;
  bottom: 12px;
  left: 12px;
  width: fit-content;
  max-width: calc(100% - 24px);
  padding: 7px 9px;
  border: 1px solid var(--board-line);
  border-radius: 8px;
  background: rgba(11,21,17,0.86);
  color: var(--on-board-muted);
  font-size: 9px;
  backdrop-filter: blur(8px);
}

.minimap {
  position: absolute;
  right: 12px;
  bottom: 12px;
  width: 104px;
  height: 64px;
  border: 1px solid var(--board-line);
  border-radius: 8px;
  background:
    linear-gradient(90deg, transparent 8%, var(--copper) 8% 11%, transparent 11% 29%, var(--copper) 29% 32%, transparent 32% 51%, var(--copper) 51% 54%, transparent 54% 73%, var(--copper) 73% 76%, transparent 76%),
    var(--board-raised);
  opacity: 0.72;
  pointer-events: none;
}

.minimap::after {
  content: "";
  position: absolute;
  inset: 9px 12px;
  border: 1px solid var(--active);
  border-radius: 3px;
  box-shadow: 0 0 8px rgba(71,224,138,0.28);
}

/* Shared SVG schematic language */
.board-grid line { stroke: rgba(255,255,255,0.035); }
.schematic-wire { fill: none; stroke: var(--copper); stroke-width: 2; vector-effect: non-scaling-stroke; }
.schematic-wire.muted { stroke: #456155; opacity: 0.72; }
.schematic-wire.param { stroke: var(--violet); stroke-width: 1.4; stroke-dasharray: 5 5; }
.schematic-wire.active { stroke: var(--active); stroke-width: 2.7; stroke-dasharray: 10 8; animation: signal-flow 800ms linear infinite; filter: drop-shadow(0 0 6px rgba(71,224,138,0.56)); }
.schematic-wire.negative { stroke: var(--negative); }
.schematic-wire.positive { stroke: var(--positive); }
.schematic-node { cursor: pointer; }
.schematic-node rect,
.schematic-node circle,
.schematic-node polygon { fill: var(--board-raised); stroke: #466759; stroke-width: 1.3; vector-effect: non-scaling-stroke; }
.schematic-node:hover rect,
.schematic-node:hover circle,
.schematic-node:focus rect,
.schematic-node.selected rect { stroke: var(--active); stroke-width: 2; filter: drop-shadow(0 0 8px rgba(71,224,138,0.30)); }
.schematic-node.kind-input rect,
.schematic-node.kind-output rect { fill: #15335a; stroke: var(--blue); }
.schematic-node.kind-conv2d rect,
.schematic-node.kind-linear rect { fill: #3a2c13; stroke: var(--positive); }
.schematic-node.kind-relu rect { fill: #0e3b35; stroke: var(--teal); }
.schematic-node.kind-argmax rect { fill: #421c27; stroke: var(--danger); }
.schematic-node.kind-flatten rect { fill: #26352f; stroke: #789085; }
.schematic-node.kind-memory rect { fill: #2e2445; stroke: var(--violet); }
.schematic-node text { fill: var(--on-board); font-family: Inter, ui-sans-serif, sans-serif; font-size: 12px; font-weight: 720; pointer-events: none; }
.schematic-node text.subtext { fill: var(--on-board-muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; font-weight: 500; }
.schematic-node text.value-positive { fill: var(--positive); }
.schematic-node text.value-negative { fill: var(--negative); }
.schematic-port { fill: var(--active); stroke: var(--board); stroke-width: 2; }
.schematic-label { fill: var(--on-board-muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; }
.schematic-title { fill: var(--on-board); font-size: 15px; font-weight: 760; }
.schematic-overline { fill: var(--on-board-muted); font-size: 9px; font-weight: 720; letter-spacing: 1px; text-transform: uppercase; }
.operator-symbol { fill: var(--on-board); font-size: 20px; font-weight: 760; text-anchor: middle; dominant-baseline: central; }
.active-node rect,
.active-node circle { stroke: var(--active); filter: drop-shadow(0 0 9px rgba(71,224,138,0.40)); }
.done-node rect,
.done-node circle { stroke: #68a681; }

@keyframes signal-flow { to { stroke-dashoffset: -18; } }

.board-note {
  position: absolute;
  z-index: 2;
  top: 18px;
  left: 20px;
  max-width: 420px;
  padding: 9px 11px;
  border: 1px solid var(--board-line);
  border-left: 3px solid var(--copper);
  border-radius: 8px;
  background: rgba(16,32,26,0.88);
  color: var(--on-board-muted);
  font-size: 10px;
  line-height: 1.45;
  backdrop-filter: blur(7px);
}

.board-note strong { color: var(--on-board); }

/* Conv matrices */
.conv-layout,
.linear-layout,
.yosys-layout,
.mapping-layout {
  min-width: 820px;
  min-height: 520px;
  padding: 68px 32px 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 26px;
  color: var(--on-board);
}

.matrix-section { flex: 0 0 auto; }
.matrix-section h3 { margin: 0 0 5px; font-size: 13px; }
.matrix-section p { margin: 0 0 12px; color: var(--on-board-muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; }
.matrix-grid { display: grid; gap: 5px; }
.matrix-grid.cols-2 { grid-template-columns: repeat(2, 58px); }
.matrix-grid.cols-3 { grid-template-columns: repeat(3, 58px); }
.matrix-grid.cols-4 { grid-template-columns: repeat(4, 46px); }

.matrix-cell {
  position: relative;
  min-width: 46px;
  min-height: 48px;
  padding: 7px 5px;
  border: 1px solid var(--board-line);
  border-radius: 7px;
  display: grid;
  place-items: center;
  background: var(--board-raised);
  color: var(--on-board);
  cursor: default;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 12px;
}

button.matrix-cell { cursor: pointer; }
.matrix-cell small { display: block; color: var(--on-board-muted); font-size: 7px; }
.matrix-cell.positive { color: var(--positive); background: rgba(242,184,75,0.11); }
.matrix-cell.negative { color: var(--negative); background: rgba(92,168,255,0.10); }
.matrix-cell.receptive { border-color: var(--active); box-shadow: inset 0 0 0 1px rgba(71,224,138,0.28), 0 0 12px rgba(71,224,138,0.14); }
.matrix-cell.selected { border-color: var(--positive); box-shadow: 0 0 0 2px rgba(242,184,75,0.20), 0 0 16px rgba(242,184,75,0.18); }
.matrix-operator { color: var(--copper); font-size: 30px; font-weight: 300; }
.route-stack { display: grid; place-items: center; gap: 7px; color: var(--on-board-muted); font-size: 9px; }
.route-stack strong { color: var(--active); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 16px; }

.canvas-primary {
  min-height: 36px;
  padding: 8px 12px;
  border: 1px solid #64e99b;
  border-radius: 8px;
  background: var(--active);
  color: #062414;
  cursor: pointer;
  font-size: 10px;
  font-weight: 800;
  box-shadow: 0 9px 24px rgba(71,224,138,0.17);
}

.canvas-secondary {
  min-height: 34px;
  padding: 7px 10px;
  border: 1px solid var(--board-line);
  border-radius: 8px;
  background: var(--board-raised);
  color: var(--on-board);
  cursor: pointer;
  font-size: 9px;
  font-weight: 730;
}

.conv-action { margin-top: 17px; width: 100%; }

/* Linear and argmax */
.linear-layout { align-items: stretch; gap: 18px; }
.linear-card,
.yosys-cluster,
.mapping-card,
.empty-state-card {
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--board-line);
  border-radius: 12px;
  background: rgba(16,32,26,0.86);
}
.linear-card h3,
.yosys-cluster h3,
.mapping-card h3 { margin: 0 0 4px; font-size: 12px; }
.linear-card > p,
.yosys-cluster > p,
.mapping-card > p { margin: 0 0 13px; color: var(--on-board-muted); font-size: 9px; }
.activation-vector { display: flex; gap: 5px; }
.activation-pill { min-width: 38px; padding: 8px 6px; border: 1px solid var(--board-line); border-radius: 7px; text-align: center; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 10px; }
.mac-row-list { display: grid; gap: 6px; }
.mac-row-button { width: 100%; padding: 8px; border: 1px solid var(--board-line); border-radius: 8px; display: grid; grid-template-columns: 26px 1fr auto; align-items: center; gap: 7px; background: var(--board-raised); color: var(--on-board); cursor: pointer; text-align: left; }
.mac-row-button:hover,
.mac-row-button.winner { border-color: var(--positive); }
.mac-row-button code { color: var(--on-board-muted); font-size: 9px; }
.logit-bars { min-width: 170px; display: grid; gap: 8px; }
.logit-row { display: grid; grid-template-columns: 28px 1fr 30px; align-items: center; gap: 7px; color: var(--on-board-muted); font-size: 9px; }
.logit-track { height: 10px; border-radius: 999px; background: #1c3027; overflow: hidden; }
.logit-fill { height: 100%; min-width: 2px; border-radius: inherit; background: var(--blue); }
.logit-row.winner { color: var(--positive); }
.logit-row.winner .logit-fill { background: var(--positive); box-shadow: 0 0 10px rgba(242,184,75,0.42); }
.argmax-stack { display: grid; align-content: center; justify-items: center; gap: 9px; }
.compare-node { width: 54px; height: 54px; border: 1px solid var(--danger); border-radius: 50%; display: grid; place-items: center; background: #351923; color: var(--on-board); font-weight: 800; }
.winner-readout { padding: 11px 15px; border: 1px solid var(--positive); border-radius: 9px; background: rgba(242,184,75,0.10); color: var(--positive); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; font-weight: 760; }

/* Yosys and mapping */
.yosys-layout { align-items: stretch; justify-content: flex-start; gap: 14px; overflow-x: auto; }
.yosys-warning { position: absolute; top: 16px; right: 18px; left: 18px; padding: 8px 10px; border: 1px solid rgba(242,184,75,0.36); border-radius: 8px; background: rgba(242,184,75,0.08); color: #f6d98d; font-size: 9px; }
.yosys-cluster { flex: 1 0 180px; max-width: 280px; }
.cell-grid { display: grid; grid-template-columns: repeat(2, minmax(66px, 1fr)); gap: 7px; }
.yosys-cell { min-height: 48px; padding: 7px; border: 1px solid var(--board-line); border-radius: 7px; background: #162a21; color: var(--on-board); cursor: pointer; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; text-align: left; }
.yosys-cell strong,
.yosys-cell span { display: block; }
.yosys-cell span { margin-top: 3px; color: var(--on-board-muted); font-size: 7px; }
.yosys-cell:hover,
.yosys-cell.active { border-color: var(--active); box-shadow: 0 0 12px rgba(71,224,138,0.15); }
.empty-state-card { width: min(560px, 92%); margin: auto; text-align: center; }
.empty-state-icon { width: 58px; height: 58px; margin: 0 auto 12px; border: 1px dashed var(--copper); border-radius: 16px; display: grid; place-items: center; color: var(--copper); font-size: 24px; }
.empty-state-card h3 { margin: 0 0 7px; }
.empty-state-card p { max-width: 420px; margin: 0 auto 15px; color: var(--on-board-muted); font-size: 10px; line-height: 1.5; }
.mapping-layout { align-items: stretch; }
.mapping-card { flex: 1 1 0; max-width: 380px; }
.mapping-bridge { align-self: center; min-width: 110px; color: var(--active); text-align: center; }
.mapping-bridge::before { content: ""; display: block; height: 2px; margin-bottom: 7px; background: linear-gradient(90deg, var(--copper), var(--active)); box-shadow: 0 0 8px rgba(71,224,138,0.30); }
.mapping-bridge strong { font-size: 9px; text-transform: uppercase; }
.mapping-list { display: grid; gap: 7px; }
.mapping-row { padding: 8px; border: 1px solid var(--board-line); border-radius: 7px; display: flex; justify-content: space-between; gap: 12px; color: var(--on-board-muted); font-size: 9px; }
.mapping-row code { color: var(--on-board); text-align: right; }

.inspector { display: grid; grid-template-rows: auto minmax(0, 1fr); }
.inspector-head { position: relative; }
.close-inspector { display: none; width: 30px; height: 30px; border: 0; border-radius: 7px; background: var(--app); cursor: pointer; font-size: 18px; }
.inspector-body { padding: 14px; overflow: auto; }
.detail-title { margin: 0; font-size: 19px; line-height: 1.15; letter-spacing: -0.025em; }
.detail-kind { margin: 5px 0 15px; color: var(--muted); font-size: 10px; font-weight: 720; text-transform: uppercase; }
.inspector-section { margin-top: 15px; }
.inspector-section-title { margin: 0 0 8px; color: var(--muted); font-size: 9px; font-weight: 780; letter-spacing: 0.08em; text-transform: uppercase; }
.kv { margin: 0; display: grid; grid-template-columns: minmax(88px, 0.8fr) minmax(0, 1.2fr); gap: 0; }
.kv dt,
.kv dd { min-width: 0; margin: 0; padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 10px; }
.kv dt { color: var(--muted); }
.kv dd { color: var(--ink); font-weight: 680; text-align: right; overflow-wrap: anywhere; }
code,
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.kv code { padding: 2px 4px; border-radius: 4px; background: #edf1ed; font-size: 9px; }
.value-positive { color: #9a6500 !important; }
.value-negative { color: #246db5 !important; }
.inspector-callout { margin: 12px 0 0; padding: 10px; border: 1px solid #cfe0d3; border-left: 3px solid var(--active); border-radius: 8px; background: #edf7ef; color: #42604c; font-size: 10px; line-height: 1.45; }
.inspector-callout.warning { border-color: #ead7ad; border-left-color: var(--positive); background: #fff8e8; color: #745c28; }
.inspector-actions { margin-top: 13px; display: grid; gap: 7px; }
.primary-button,
.secondary-button { min-height: 38px; padding: 8px 11px; border-radius: 8px; cursor: pointer; font-size: 10px; font-weight: 790; }
.primary-button { border: 1px solid #42c878; background: var(--active); color: #082716; }
.secondary-button { border: 1px solid var(--line); background: var(--raised); color: var(--ink); }
.badge-row { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px; }
.status-badge.good { border-color: #b8dec3; background: #eaf7ed; color: #245b35; }
.status-badge.warn { border-color: #ecd9a9; background: #fff8e7; color: #765c1d; }
.equation { padding: 10px; border-radius: 8px; background: var(--board); color: var(--on-board); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; line-height: 1.7; overflow-x: auto; }
.equation strong { color: var(--positive); }
.logit-table { display: grid; gap: 4px; }
.logit-table-row { min-height: 34px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 7px; display: grid; grid-template-columns: 1fr auto auto; align-items: center; gap: 7px; background: var(--raised); font-size: 10px; }
.logit-table-row.winner { border-color: #dfbd6a; background: #fff8e8; }
.logit-table-row code { min-width: 34px; text-align: right; }
.artifact-mini { display: grid; gap: 4px; }
.artifact-mini a { min-width: 0; padding: 7px 8px; border: 1px solid var(--line); border-radius: 7px; display: flex; align-items: center; justify-content: space-between; gap: 8px; background: var(--raised); color: var(--ink); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; text-decoration: none; }
.artifact-mini a:hover { border-color: #93cba4; background: #edf7ef; }
.artifact-mini a span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.artifact-mini a small,
.empty-note { color: var(--muted); font-size: 8px; }

.transport {
  z-index: 19;
  min-width: 0;
  min-height: var(--transport-h);
  padding: 10px 16px;
  border-top: 1px solid var(--line);
  display: grid;
  grid-template-columns: auto minmax(260px, 1fr) auto;
  align-items: center;
  gap: 16px;
  background: rgba(247,250,246,0.96);
  box-shadow: 0 -12px 34px rgba(14,28,20,0.08);
  backdrop-filter: blur(18px);
}

.transport-controls { display: flex; align-items: center; gap: 7px; }
.transport-button { min-height: 38px; padding: 7px 10px; border-radius: 9px; cursor: pointer; font-size: 10px; font-weight: 790; }
.transport-button.secondary { border: 1px solid var(--line); background: var(--raised); }
.play-button { width: 42px; border: 1px solid #2bc96e; background: var(--active); color: #062414; font-size: 14px; }
.timeline-wrap { min-width: 0; }
.timeline-meta { margin-bottom: 7px; display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 9px; }
.timeline-meta strong { color: var(--ink); font-size: 10px; }
.timeline { position: relative; min-width: 0; display: grid; grid-template-columns: repeat(6, minmax(32px, 1fr)); gap: 5px; }
.timeline::before { content: ""; position: absolute; z-index: 0; top: 7px; right: 2%; left: 2%; height: 2px; background: var(--line); }
.timeline-step { position: relative; z-index: 1; min-width: 0; padding: 0; border: 0; display: grid; grid-template-rows: 16px auto; justify-items: center; gap: 3px; background: transparent; color: var(--muted); cursor: pointer; font-size: 8px; }
.timeline-step i { width: 15px; height: 15px; border: 3px solid var(--panel); border-radius: 999px; background: #bac4bd; box-shadow: 0 0 0 1px var(--line); }
.timeline-step.done i { background: #78b98d; }
.timeline-step.active i { background: var(--active); box-shadow: 0 0 0 2px rgba(71,224,138,0.22), 0 0 9px rgba(71,224,138,0.36); }
.timeline-step span { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.result-readout { min-width: 180px; display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
.logits-compact { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; }
.class-chip { min-height: 46px; min-width: 84px; padding: 7px 11px; border: 1px solid #d7af4d; border-radius: 10px; background: #fff8e7; color: #684c06; text-align: center; }
.class-chip span,
.class-chip strong { display: block; }
.class-chip span { font-size: 8px; text-transform: uppercase; }
.class-chip strong { margin-top: 2px; font-size: 14px; }

.vector-dialog {
  width: min(680px, calc(100vw - 28px));
  max-height: min(720px, calc(100vh - 28px));
  padding: 0;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--panel);
  box-shadow: var(--shadow-high);
}
.vector-dialog::backdrop { background: rgba(8,18,12,0.52); backdrop-filter: blur(5px); }
.dialog-shell { display: grid; grid-template-rows: auto minmax(0, 1fr); max-height: inherit; }
.dialog-head,
.drawer-head { padding: 15px 17px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.dialog-head h2,
.drawer-head h2 { margin: 0; font-size: 17px; }
.dialog-head > button,
.drawer-head > button { width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 8px; background: var(--raised); cursor: pointer; font-size: 18px; }
.vector-list { padding: 12px; display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; overflow: auto; }
.vector-option { min-width: 0; padding: 10px; border: 1px solid var(--line); border-radius: 10px; display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 9px; background: var(--raised); cursor: pointer; text-align: left; }
.vector-option:hover,
.vector-option.active { border-color: #88c99c; background: #edf7ef; }
.vector-index { width: 36px; height: 36px; border-radius: 9px; display: grid; place-items: center; background: var(--board); color: var(--active); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 10px; }
.vector-option strong,
.vector-option span { display: block; min-width: 0; }
.vector-option strong { overflow: hidden; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.vector-option span { margin-top: 3px; color: var(--muted); font-size: 8px; }
.vector-result { color: #8b6206; font-size: 9px; font-weight: 790; }

.drawer-backdrop { position: fixed; z-index: 90; inset: 0; background: rgba(8,18,12,0.42); backdrop-filter: blur(3px); }
.source-drawer { position: fixed; z-index: 100; top: 0; right: 0; bottom: 0; width: min(620px, 92vw); display: grid; grid-template-rows: auto auto minmax(0,1fr); transform: translateX(105%); background: var(--panel); box-shadow: -24px 0 80px rgba(6,17,11,0.26); transition: transform 220ms ease; }
.source-drawer.open { transform: translateX(0); }
.source-tabs { padding: 8px 10px; border-bottom: 1px solid var(--line); display: flex; gap: 6px; overflow-x: auto; }
.source-tab { min-height: 30px; padding: 5px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--raised); color: var(--muted); cursor: pointer; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; white-space: nowrap; }
.source-tab.active { border-color: #7fc494; background: #edf7ef; color: #235b35; }
.source-code { min-height: 0; margin: 0; padding: 17px; overflow: auto; background: var(--board); color: var(--on-board); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 10px; line-height: 1.6; tab-size: 2; }

.signal-tooltip { position: fixed; z-index: 120; max-width: 260px; padding: 8px 10px; border: 1px solid var(--board-line); border-radius: 8px; background: rgba(11,21,17,0.96); color: var(--on-board); box-shadow: var(--shadow-high); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; line-height: 1.45; pointer-events: none; }
.screen-reader-status { position: fixed; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap; }
.support-data { display: none !important; }

@media (max-width: 1320px) {
  .topbar { grid-template-columns: minmax(230px, 1fr) auto minmax(450px, 1.25fr); }
  .quant-readout .chip:nth-child(n+3) { display: none; }
  .workspace { grid-template-columns: 220px minmax(500px, 1fr); }
  .inspector {
    position: fixed;
    z-index: 60;
    top: calc(var(--header-h) + var(--context-h) + 8px);
    right: 10px;
    bottom: calc(var(--transport-h) + 10px);
    width: min(352px, calc(100vw - 20px));
    transform: translateX(calc(100% + 24px));
    transition: transform 200ms ease;
    box-shadow: var(--shadow-high);
  }
  body.inspector-open .inspector { transform: translateX(0); }
  .close-inspector,
  .mobile-inspector-button { display: block; }
}

@media (max-width: 1040px) {
  :root { --header-h: 92px; }
  .topbar { padding: 8px 13px; grid-template-columns: 1fr auto; grid-template-rows: auto auto; gap: 7px 12px; }
  .brand-lockup { grid-row: 1; }
  .build-identity { grid-row: 1; }
  .topbar-actions { grid-column: 1 / -1; grid-row: 2; justify-content: space-between; }
  .quant-readout { flex: 1; }
  .workspace { grid-template-columns: minmax(0,1fr); grid-template-rows: 92px minmax(0,1fr); }
  .layer-rail { grid-row: 1; display: grid; grid-template-columns: auto minmax(0,1fr); grid-template-rows: 1fr; }
  .layer-rail .panel-head,
  .rail-foot { display: none; }
  .layer-tree { flex-direction: row; align-items: stretch; overflow-x: auto; }
  .layer-item { flex: 0 0 148px; min-height: 64px; padding-left: 34px; }
  .layer-item::after { display: none; }
  .rail-group-label { display: none; }
  .canvas-panel { grid-row: 2; }
}

@media (max-width: 720px) {
  :root { --header-h: 122px; --context-h: 54px; --transport-h: 126px; }
  body { overflow: hidden; }
  .topbar { grid-template-columns: minmax(0,1fr); grid-template-rows: auto auto; }
  .build-identity { display: none; }
  .brand-mark { flex-basis: 32px; width: 32px; height: 32px; }
  .brand-mark::before { width: 42px; left: -5px; top: 15px; }
  .brand-mark::after { height: 42px; top: -5px; left: 15px; }
  h1 { font-size: 17px; }
  .topbar-actions { overflow-x: auto; justify-content: flex-start; }
  .quant-readout { display: none; }
  .segmented { flex: 0 0 auto; }
  .context-bar { padding: 5px 10px; }
  .breadcrumbs { display: none; }
  .context-tools { width: 100%; justify-content: space-between; }
  .status-strip { display: none; }
  .workspace { padding: 7px; grid-template-rows: 74px minmax(0,1fr); gap: 7px; }
  .panel { border-radius: 11px; }
  .layer-item { flex-basis: 130px; min-height: 54px; }
  .canvas-head { min-height: 50px; }
  .canvas-head-meta #graph-summary { display: none; }
  .visualization-canvas { min-width: 720px; }
  .transport { padding: 8px 10px; grid-template-columns: 1fr auto; grid-template-rows: auto auto; gap: 7px 10px; }
  .transport-controls { grid-column: 1; grid-row: 2; }
  .transport-button.secondary { min-width: 86px; }
  .timeline-wrap { grid-column: 1 / -1; grid-row: 1; }
  .result-readout { grid-column: 2; grid-row: 2; min-width: 0; }
  .logits-compact { display: none; }
  .class-chip { min-height: 38px; min-width: 70px; }
  .minimap { display: none; }
  .vector-list { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; }
}
"""
