STYLES = """
:root {
  --ink: #151713;
  --muted: #62685f;
  --paper: #f3f5f0;
  --panel: #ffffff;
  --board: #101915;
  --trace: #37d67a;
  --linear: #f59e0b;
  --relu: #14b8a6;
  --argmax: #e11d48;
  --memory: #8b5cf6;
  --io: #3b82f6;
  --wire: #afbab0;
  --line: #d8ded6;
  --shadow: 0 18px 50px rgba(23, 31, 27, 0.12);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  color: var(--ink);
  background:
    linear-gradient(135deg, rgba(20, 184, 166, 0.08), transparent 34%),
    linear-gradient(315deg, rgba(245, 158, 11, 0.10), transparent 28%),
    var(--paper);
  font-family: "Segoe UI", Inter, Arial, sans-serif;
  letter-spacing: 0;
}

.app-shell {
  min-height: 100vh;
  padding: 22px;
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto;
  gap: 14px;
}

.topbar {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-end;
}

.eyebrow {
  margin: 0 0 5px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.1;
  font-weight: 760;
}

.quant-readout {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.chip {
  min-height: 30px;
  padding: 7px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.72);
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.status-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.status-item {
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-left: 5px solid var(--muted);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: 0 8px 24px rgba(23, 31, 27, 0.06);
}

.status-item strong {
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  text-transform: uppercase;
}

.status-item span {
  color: var(--muted);
  font-size: 13px;
  overflow-wrap: anywhere;
}

.status-ok { border-left-color: #16a34a; }
.status-failed { border-left-color: #dc2626; }
.status-pending { border-left-color: #f59e0b; }

.workspace {
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 14px;
}

.panel,
.trace-panel {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.canvas-panel {
  position: relative;
  background: var(--board);
}

.canvas-head,
.inspector-head,
.trace-head {
  min-height: 48px;
  padding: 13px 16px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.canvas-head {
  color: #e7eee8;
  background: rgba(255, 255, 255, 0.03);
  border-bottom-color: rgba(216, 222, 214, 0.18);
}

.canvas-head strong,
.inspector-head strong,
.trace-head strong {
  font-size: 13px;
  text-transform: uppercase;
}

.canvas-head span,
.inspector-head span,
.trace-head span {
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}

.canvas-head span { color: #9fb3a7; }

.canvas-scroll {
  overflow: auto;
  min-height: 430px;
}

svg {
  display: block;
  width: 100%;
  min-width: 880px;
  height: 430px;
}

.board-grid line { stroke: rgba(255, 255, 255, 0.045); }
.wire { fill: none; stroke: var(--wire); stroke-width: 2.3; }
.wire.param { stroke: rgba(139, 92, 246, 0.72); stroke-width: 1.8; }
.wire.active {
  stroke: var(--trace);
  stroke-dasharray: 8 10;
  animation: flow 1s linear infinite;
}

@keyframes flow { to { stroke-dashoffset: -18; } }

.block rect {
  stroke: rgba(255, 255, 255, 0.38);
  stroke-width: 1.5;
  filter: drop-shadow(0 14px 18px rgba(0, 0, 0, 0.22));
}

.block text {
  fill: #f8faf8;
  font-size: 12px;
  font-weight: 760;
  pointer-events: none;
}

.block .subtext {
  fill: rgba(248, 250, 248, 0.72);
  font-size: 11px;
  font-weight: 620;
}

.block.selected rect,
.block:focus rect {
  stroke: #ffffff;
  stroke-width: 2.5;
}

.block:focus { outline: none; }
.kind-input rect, .kind-output rect { fill: var(--io); }
.kind-linear rect { fill: var(--linear); }
.kind-relu rect { fill: var(--relu); }
.kind-argmax rect { fill: var(--argmax); }
.kind-flatten rect { fill: #64748b; }
.kind-memory rect { fill: var(--memory); }

.signal-label {
  fill: #b8c7bd;
  font-size: 11px;
  font-weight: 650;
}

.inspector {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
}

.inspector-body {
  padding: 16px;
  overflow: auto;
}

.detail-title {
  margin: 0 0 6px;
  font-size: 20px;
  line-height: 1.2;
}

.detail-kind {
  margin: 0 0 16px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
}

.kv {
  display: grid;
  grid-template-columns: 112px minmax(0, 1fr);
  gap: 8px 10px;
  padding: 10px 0;
  border-top: 1px solid var(--line);
  font-size: 13px;
}

.kv dt { color: var(--muted); font-weight: 700; }
.kv dd { margin: 0; overflow-wrap: anywhere; }

a {
  color: #166534;
  font-weight: 720;
  text-decoration: none;
}

a:focus,
a:hover { text-decoration: underline; }

.trace-grid {
  padding: 14px;
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 12px;
}

.trace-summary {
  padding-right: 12px;
  border-right: 1px solid var(--line);
}

.class-readout {
  margin: 0 0 12px;
  font-size: 32px;
  line-height: 1;
  font-weight: 800;
}

.trace-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
}

.trace-step {
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfa;
  cursor: pointer;
  text-align: left;
}

.trace-step strong {
  display: block;
  margin-bottom: 7px;
  font-size: 12px;
  text-transform: uppercase;
}

.trace-step span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}

code {
  padding: 1px 5px;
  border-radius: 6px;
  background: rgba(21, 23, 19, 0.08);
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 12px;
}

@media (max-width: 980px) {
  .app-shell { padding: 12px; }
  .topbar { align-items: flex-start; flex-direction: column; }
  .quant-readout { justify-content: flex-start; }
  .status-strip { grid-template-columns: 1fr 1fr; }
  .workspace { grid-template-columns: 1fr; }
  .trace-grid { grid-template-columns: 1fr; }
  .trace-summary {
    padding-right: 0;
    padding-bottom: 10px;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
}
"""
