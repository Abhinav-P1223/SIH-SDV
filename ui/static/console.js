/* SIH26037 autonomy console — client.
   Read-only. Renders telemetry published by the frozen stack; computes no autonomy quantity.
   Every value drawn comes from a field in the compact frame (autonomy/telemetry/dashboard.py). */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const deg = (r) => (r === null || r === undefined ? "—" : (r * 180 / Math.PI).toFixed(1));

const COLOR = {
  PEDESTRIAN: "#f472b6", BICYCLE: "#c084fc", MOTORCYCLE: "#fb923c",
  AUTO_RICKSHAW: "#fbbf24", CAR: "#38bdf8", BUS: "#34d399",
  TRUCK: "#2dd4bf", PUSHCART: "#a3a3a3", CATTLE: "#f87171", UNKNOWN: "#94a3b8",
};
const SHORT = {
  PEDESTRIAN: "PED", BICYCLE: "BIKE", MOTORCYCLE: "MOTO", AUTO_RICKSHAW: "AUTO",
  CAR: "CAR", BUS: "BUS", TRUCK: "TRUCK", PUSHCART: "CART", CATTLE: "COW", UNKNOWN: "UNK",
};

const S = {
  frame: null, es: null, history: [], events: [],
  lastState: null, lastRisk: null, lastSel: null, lastBrake: false, lastRev: false,
  seenTracks: new Set(), why: null, demo: null, results: null, running: false,
};

/* ══════════════ tabs ══════════════ */
document.querySelectorAll(".tab").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("is-on", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("is-on", p.id === "tab-" + b.dataset.tab));
    $("ctl").style.visibility = b.dataset.tab === "live" ? "visible" : "hidden";
    if (b.dataset.tab !== "live") loadResults();
    if (b.dataset.tab === "live") sizeCanvas();
  };
});

function toast(msg, err) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast on" + (err ? " err" : "");
  clearTimeout(t._t);
  t._t = setTimeout(() => (t.className = "toast"), 3600);
}

/* ══════════════ controls ══════════════ */
async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  return r.json();
}

async function loadScenarios() {
  const d = await api("/api/scenarios");
  const sel = $("sel-scenario");
  sel.innerHTML = "";
  d.scenarios.forEach((n) => {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())
      + (d.demo.includes(n) ? "  ·  demo" : "");
    sel.appendChild(o);
  });
  S.demoList = d.demo;
}

$("btn-run").onclick = () => startRun($("sel-scenario").value);
$("btn-pause").onclick = async () => {
  const d = await api("/api/pause", {});
  $("btn-pause").textContent = d.status === "paused" ? "Resume" : "Pause";
};
$("btn-reset").onclick = async () => { S.demo = null; await doReset(); };
$("btn-demo").onclick = () => { S.demo = { i: 0 }; runDemoStep(); };

async function doReset() {
  await api("/api/reset", {});
  S.frame = null; S.history = []; S.events = []; S.seenTracks.clear();
  S.lastState = S.lastRisk = S.lastSel = null; S.lastBrake = S.lastRev = false;
  S.why = null; S.running = false;
  $("events").innerHTML = ""; $("runsum").hidden = true; $("scene-empty").hidden = false;
  $("why-body").hidden = true; $("why-empty").hidden = false;
  $("btn-run").disabled = false; $("btn-pause").disabled = true; $("btn-pause").textContent = "Pause";
  document.body.dataset.risk = "NONE";
  clearPanels(); draw();
}

async function startRun(name) {
  await api("/api/reset", {});
  S.frame = null; S.history = []; S.events = []; S.seenTracks.clear();
  S.lastState = S.lastRisk = S.lastSel = null; S.lastBrake = S.lastRev = false; S.why = null;
  $("events").innerHTML = ""; $("runsum").hidden = true;
  const d = await api("/api/run", {
    scenario: name, perception: $("sel-mode").value, speed: parseFloat($("sel-speed").value),
  });
  if (!d.ok) { toast(d.message, true); return; }
  S.running = true;
  $("scene-empty").hidden = true;
  $("btn-run").disabled = true; $("btn-pause").disabled = false;
  pushEvent(0, "scenario " + name + " · " + $("sel-mode").value, "k-state");
}

function runDemoStep() {
  if (!S.demo || S.demo.i >= S.demoList.length) { S.demo = null; toast("Demo complete"); return; }
  const name = S.demoList[S.demo.i];
  $("sel-scenario").value = name;
  toast(`Demo ${S.demo.i + 1}/${S.demoList.length} — ${name.replace(/_/g, " ")}`);
  startRun(name);
}

/* ══════════════ SSE ══════════════ */
function connect() {
  if (S.es) S.es.close();
  const es = new EventSource("/api/stream");
  S.es = es;
  es.onopen = () => $("conn").className = "conn on";
  es.onerror = () => { $("conn").className = "conn"; $("conn").querySelector("span").textContent = "reconnecting"; };
  es.addEventListener("frame", (e) => {
    let f; try { f = JSON.parse(e.data); } catch { return; }
    $("conn").className = "conn on"; $("conn").querySelector("span").textContent = "live";
    if (f._event) return handleEvent(f);
    onFrame(f);
  });
}

function handleEvent(f) {
  if (f._event === "done") {
    S.running = false;
    $("btn-run").disabled = false; $("btn-pause").disabled = true;
    showSummary(f.summary);
    pushEvent(f.summary && f.summary.duration_s, "run finished — " + (f.summary ? f.summary.termination : ""), "k-goal");
    if (S.demo) { S.demo.i += 1; setTimeout(runDemoStep, 2200); }
  } else if (f._event === "error") {
    S.running = false; $("btn-run").disabled = false; $("btn-pause").disabled = true;
    toast(f.message, true); pushEvent(null, "ERROR " + f.message, "k-brake");
  } else if (f._event === "cancelled" || f._event === "reset") {
    S.running = false; $("btn-run").disabled = false; $("btn-pause").disabled = true;
  }
}

/* ══════════════ frame handling ══════════════ */
function onFrame(f) {
  S.frame = f;
  $("scene-empty").hidden = true;
  const e = f.ego || {};
  if (S.history.length === 0 || Math.hypot(e.x - S.history[S.history.length - 1][0],
                                           e.y - S.history[S.history.length - 1][1]) > 0.25) {
    S.history.push([e.x, e.y]);
    if (S.history.length > 2600) S.history.shift();
  }
  detectEvents(f);
  updatePanels(f);
  draw();
}

function detectEvents(f) {
  const t = f.t, b = f.behavior, r = f.risk || {}, s = f.safety || {}, p = f.plan;

  (f.objects || []).forEach((o) => {
    if (!S.seenTracks.has(o.id)) {
      S.seenTracks.add(o.id);
      pushEvent(t, `track ${o.id} · ${SHORT[o.type] || o.type}`, "");
    }
  });
  if (b && b.state !== S.lastState) {
    pushEvent(t, `${S.lastState || "—"} → ${b.state}`, b.state === "EMERGENCY_BRAKE" ? "k-brake"
      : b.state === "REVERSING" ? "k-rev" : "k-state");
    if (S.lastState !== null) captureWhy(f);
    S.lastState = b.state;
  }
  if (r.max_level && r.max_level !== S.lastRisk) {
    if (S.lastRisk !== null) pushEvent(t, `risk ${S.lastRisk} → ${r.max_level}`, "k-risk");
    S.lastRisk = r.max_level;
  }
  if (s.override_active && !S.lastBrake) pushEvent(t, "EMERGENCY BRAKE — " + (s.reason || ""), "k-brake");
  S.lastBrake = !!s.override_active;
  const rev = (f.ego && f.ego.longitudinal_velocity < -0.05);
  if (rev && !S.lastRev) pushEvent(t, "reverse manoeuvre started", "k-rev");
  S.lastRev = rev;
  if (p && p.selected_id && p.selected_id !== S.lastSel) {
    if (S.lastSel !== null) pushEvent(t, `replan → ${p.selected_id}`, "");
    S.lastSel = p.selected_id;
  }
}

/* "Why did we replan?" — every field below is a telemetry value, not a model.
   The cause string is the FSM's own `reason`; the object is the highest-risk track. */
function captureWhy(f) {
  const b = f.behavior, r = f.risk || {}, p = f.plan, c = f.control || {};
  if (!b || !p) return;
  const worst = (f.objects || []).find((o) => o.id === r.worst_object_id);
  S.why = {
    cause: b.reason || "—",
    ttc: r.min_ttc_current_speed != null ? fmt(r.min_ttc_current_speed, 2) + " s"
       : (r.min_ttc != null ? fmt(r.min_ttc, 2) + " s (route)" : "—"),
    feas: `${p.feasible_count} / ${p.candidate_count}`,
    dec: b.state + (b.force_stop ? " · STOP" : ` · ${fmt(b.target_speed, 1)} m/s`),
    resp: fmt((c.acceleration || 0) - (c.brake || 0), 2) + " m/s²",
    clear: p.selected && p.selected.min_clearance != null ? fmt(p.selected.min_clearance, 2) + " m" : "—",
    sel: p.selected_id || "—",
    obj: worst ? `${SHORT[worst.type] || worst.type} ${worst.id}` : null,
  };
  $("why-empty").hidden = true; $("why-body").hidden = false;
  $("why-cause").textContent = (S.why.obj ? S.why.obj + " — " : "") + S.why.cause;
  $("why-ttc").textContent = S.why.ttc;
  $("why-feas").textContent = S.why.feas;
  $("why-dec").textContent = S.why.dec;
  $("why-resp").textContent = S.why.resp;
  $("why-clear").textContent = S.why.clear;
  $("why-sel").textContent = S.why.sel;
}

function pushEvent(t, text, cls) {
  const box = $("events");
  const d = document.createElement("div");
  d.className = "ev " + (cls || "");
  d.innerHTML = `<time>${t == null ? "—" : Number(t).toFixed(1)}</time><span></span>`;
  d.querySelector("span").textContent = text;
  box.appendChild(d);
  while (box.children.length > 220) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

function clearPanels() {
  ["m-ttc", "m-clear", "m-speed", "m-steer", "m-acc", "m-brake", "k-tracks", "k-agents", "k-worst",
   "k-lead", "k-cand", "k-feas", "k-sel", "k-cost", "k-replans", "k-lat", "k-score", "k-step"]
    .forEach((i) => ($(i).textContent = "—"));
  $("beh-state").textContent = "—";
  $("beh-reason").textContent = "Waiting for telemetry.";
  $("risk-chip").textContent = "NONE";
  $("sim-clock").textContent = "0.00";
  $("scene-mode").textContent = "—";
  $("scene-alert").className = "scene-alert";
  $("rej-list").innerHTML = "<em>none</em>";
  ["f-brake", "f-coll", "f-rev"].forEach((i) => ($(i).className = "flag"));
  $("f-brake").querySelector("b").textContent = "INACTIVE";
  $("f-coll").querySelector("b").textContent = "NONE";
  $("f-rev").querySelector("b").textContent = "NO";
}

function updatePanels(f) {
  const e = f.ego || {}, c = f.control || {}, r = f.risk || {}, b = f.behavior,
        p = f.plan, s = f.safety || {}, m = f.metrics || {};

  $("sim-clock").textContent = fmt(f.t, 2);
  $("scene-mode").textContent = (f.perception_mode || "").replace("_", " ");

  document.body.dataset.risk = r.max_level || "NONE";
  $("risk-chip").textContent = r.max_level || "NONE";
  if (b) {
    $("beh-state").textContent = b.state.replace(/_/g, " ");
    $("beh-reason").textContent = b.reason || "";
  }

  $("m-ttc").textContent = r.min_ttc_current_speed != null ? fmt(r.min_ttc_current_speed, 2) : "∞";
  $("m-clear").textContent = m.minimum_obstacle_clearance != null ? fmt(m.minimum_obstacle_clearance, 2) : "—";
  $("m-speed").textContent = fmt(e.longitudinal_velocity, 2);
  $("m-steer").textContent = deg(e.steering_angle);
  $("m-acc").textContent = fmt(c.acceleration, 2);
  $("m-brake").textContent = fmt(c.brake, 2);

  const eb = !!s.override_active;
  $("f-brake").className = "flag" + (eb ? " hot" : "");
  $("f-brake").querySelector("b").textContent = eb ? "ACTIVE" : "INACTIVE";
  $("scene-alert").className = "scene-alert" + (eb ? " on" : "");
  const coll = (m.collision_count || 0) > 0;
  $("f-coll").className = "flag" + (coll ? " hot" : "");
  $("f-coll").querySelector("b").textContent = coll ? String(m.collision_count) : "NONE";
  const rev = e.longitudinal_velocity < -0.05;
  $("f-rev").className = "flag" + (rev ? " act" : "");
  $("f-rev").querySelector("b").textContent = rev ? "ENGAGED" : "NO";

  $("k-tracks").textContent = (f.objects || []).length;
  $("k-agents").textContent = (f.agents || []).length;
  $("k-worst").textContent = r.worst_object_id || "none";
  $("k-lead").textContent = r.lead_object_id || "none";
  $("k-score").textContent = fmt(r.max_score, 3);
  $("k-step").textContent = f.step != null ? f.step : "—";

  if (p) {
    $("k-cand").textContent = p.candidate_count;
    $("k-feas").textContent = p.feasible_count;
    $("k-sel").textContent = p.selected_id;
    $("k-cost").textContent = p.selected && p.selected.total_cost != null ? fmt(p.selected.total_cost, 3) : "—";
    $("k-lat").textContent = fmt(p.latency_ms, 1) + " ms";
    const hist = p.rejection_histogram || {};
    const keys = Object.keys(hist);
    $("rej-list").innerHTML = keys.length
      ? keys.sort((a, x) => hist[x] - hist[a]).map((k) =>
          `<div class="rej"><span>${k.replace(/_/g, " ").toLowerCase()}</span><b>${hist[k]}</b></div>`).join("")
      : "<em>none</em>";
  }
  $("k-replans").textContent = m.replanning_count != null ? m.replanning_count : "—";
}

function showSummary(s) {
  if (!s) return;
  const cards = [
    ["Outcome", s.termination || "—", s.completed ? "ok" : "bad"],
    ["Collisions", s.collisions, s.collisions === 0 ? "ok" : "bad"],
    ["Min clearance", fmt(s.min_clearance_m, 2) + " m", ""],
    ["Replans", s.replans, ""],
    ["Plan changes", s.plan_changes, ""],
    ["Emergency brakes", s.emergency_brakes, s.emergency_brakes ? "bad" : "ok"],
    ["Reverse", fmt(s.reverse_distance_m, 1) + " m", ""],
    ["Mean latency", fmt(s.latency_mean_ms, 1) + " ms", ""],
    ["Duration", fmt(s.duration_s, 1) + " s", ""],
    ["Path", fmt(s.path_length_m, 1) + " m", ""],
  ];
  $("runsum-body").innerHTML = cards.map(([k, v, cl]) =>
    `<div class="rs ${cl}"><span>${k}</span><b>${v}</b></div>`).join("");
  $("runsum").hidden = false;
}

/* ══════════════ scene renderer ══════════════ */
const cv = $("scene"), cx = cv.getContext("2d");
let VIEW = { s: 8, ox: 0, oy: 0, w: 0, h: 0 };

function sizeCanvas() {
  const r = cv.parentElement.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.max(1, Math.floor(r.width * dpr));
  cv.height = Math.max(1, Math.floor(r.height * dpr));
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  VIEW.w = r.width; VIEW.h = r.height;
  draw();
}
window.addEventListener("resize", sizeCanvas);

const X = (x) => VIEW.ox + (x - VIEW.cx) * VIEW.s;
const Y = (y) => VIEW.oy - (y - VIEW.cy) * VIEW.s;

function poly(pts, close) {
  cx.beginPath();
  pts.forEach((p, i) => (i ? cx.lineTo(X(p[0]), Y(p[1])) : cx.moveTo(X(p[0]), Y(p[1]))));
  if (close) cx.closePath();
}
function line(xs, ys) {
  cx.beginPath();
  for (let i = 0; i < xs.length; i++) (i ? cx.lineTo(X(xs[i]), Y(ys[i])) : cx.moveTo(X(xs[i]), Y(ys[i])));
}
function box(x, y, yaw, L, W) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hl = L / 2, hw = W / 2;
  return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([a, b]) => [x + a * c - b * s, y + a * s + b * c]);
}

function draw() {
  const f = S.frame;
  cx.clearRect(0, 0, VIEW.w, VIEW.h);
  if (!f) return;

  const e = f.ego, v = f.vehicle || { length: 4.2, width: 1.8, footprint_center_offset: 0 };
  // Follow camera: ego sits left-of-centre so the road ahead gets the space.
  const span = 62;
  VIEW.s = Math.min(VIEW.w / span, VIEW.h / 26);
  VIEW.cx = e.x; VIEW.cy = e.y;
  VIEW.ox = VIEW.w * 0.30; VIEW.oy = VIEW.h * 0.52;

  drawGrid();
  drawRoad(f.road);
  drawHistory();
  if (f.plan) drawCandidates(f.plan);
  drawPredictions(f.predictions || []);
  drawObjects(f.objects || [], f.risk || {});
  if (f.plan && f.plan.selected) drawSelected(f.plan.selected);
  drawEgo(e, v, f.safety && f.safety.override_active);
  drawScale();
}

function drawGrid() {
  cx.save();
  cx.strokeStyle = "rgba(60,84,112,.16)"; cx.lineWidth = 1;
  const step = 10, x0 = Math.floor((VIEW.cx - 60) / step) * step;
  for (let x = x0; x < VIEW.cx + 120; x += step) {
    cx.beginPath(); cx.moveTo(X(x), 0); cx.lineTo(X(x), VIEW.h); cx.stroke();
  }
  const y0 = Math.floor((VIEW.cy - 30) / step) * step;
  for (let y = y0; y < VIEW.cy + 30; y += step) {
    cx.beginPath(); cx.moveTo(0, Y(y)); cx.lineTo(VIEW.w, Y(y)); cx.stroke();
  }
  cx.restore();
}

function drawRoad(road) {
  if (!road || !road.left_boundary) return;
  const L = road.left_boundary, R = road.right_boundary;
  cx.save();
  poly(L.concat([...R].reverse()), true);
  cx.fillStyle = "rgba(24,36,52,.72)"; cx.fill();
  cx.lineWidth = 2.5; cx.strokeStyle = "#3d5570";
  poly(L); cx.stroke();
  poly(R); cx.stroke();
  if (road.reference) {                      // direction of travel, NOT a lane line
    cx.setLineDash([9, 11]); cx.lineWidth = 1.4; cx.strokeStyle = "rgba(120,150,185,.34)";
    poly(road.reference); cx.stroke(); cx.setLineDash([]);
  }
  cx.restore();
}

function drawHistory() {
  if (S.history.length < 2) return;
  cx.save();
  cx.strokeStyle = "rgba(167,139,250,.55)"; cx.lineWidth = 2; cx.lineJoin = "round";
  cx.beginPath();
  S.history.forEach((p, i) => (i ? cx.lineTo(X(p[0]), Y(p[1])) : cx.moveTo(X(p[0]), Y(p[1]))));
  cx.stroke(); cx.restore();
}

function drawCandidates(p) {
  cx.save(); cx.lineJoin = "round";
  (p.candidates || []).forEach((c) => {
    if (!c.x || c.x.length < 2) return;
    cx.strokeStyle = c.feasible ? "rgba(96,132,170,.42)" : "rgba(180,60,60,.30)";
    cx.lineWidth = c.feasible ? 1.5 : 1.1;
    line(c.x, c.y); cx.stroke();
  });
  cx.restore();
}

function drawSelected(sel) {
  if (!sel.x || sel.x.length < 2) return;
  const bad = sel.fallback || sel.degraded;
  const col = bad ? "#ef4444" : "#22d3ee";
  cx.save(); cx.lineJoin = "round"; cx.lineCap = "round";
  cx.shadowColor = col; cx.shadowBlur = 16;
  cx.strokeStyle = col; cx.lineWidth = 3.4;
  line(sel.x, sel.y); cx.stroke();
  cx.shadowBlur = 0;
  // speed dots along the plan
  for (let i = 0; i < sel.x.length; i += 4) {
    const v = sel.v ? sel.v[i] : 0;
    cx.beginPath(); cx.arc(X(sel.x[i]), Y(sel.y[i]), 2.1, 0, 7);
    cx.fillStyle = v < 0 ? "#a78bfa" : col; cx.fill();
  }
  const n = sel.x.length - 1;
  cx.beginPath(); cx.arc(X(sel.x[n]), Y(sel.y[n]), 4.4, 0, 7);
  cx.fillStyle = "rgba(8,11,16,.9)"; cx.fill();
  cx.strokeStyle = col; cx.lineWidth = 2; cx.stroke();
  cx.restore();
}

function drawPredictions(preds) {
  cx.save(); cx.lineJoin = "round";
  preds.forEach((p) => {
    if (!p.x || p.x.length < 2) return;
    cx.strokeStyle = "rgba(245,158,11,.62)"; cx.lineWidth = 1.9;
    cx.setLineDash([5, 4]);
    line(p.x, p.y); cx.stroke(); cx.setLineDash([]);
    const n = p.x.length - 1;                 // horizon end marker
    cx.beginPath(); cx.arc(X(p.x[n]), Y(p.y[n]), 3, 0, 7);
    cx.fillStyle = "rgba(245,158,11,.85)"; cx.fill();
  });
  cx.restore();
}

function drawObjects(objs, risk) {
  cx.save();
  objs.forEach((o) => {
    const col = COLOR[o.type] || COLOR.UNKNOWN;
    const worst = o.id === risk.worst_object_id;
    poly(box(o.x, o.y, o.heading || 0, o.length || 1, o.width || 1), true);
    cx.fillStyle = col + "3d"; cx.fill();
    cx.strokeStyle = col; cx.lineWidth = worst ? 2.4 : 1.5;
    if (worst) { cx.shadowColor = col; cx.shadowBlur = 12; }
    cx.stroke(); cx.shadowBlur = 0;

    const spd = Math.hypot(o.vx || 0, o.vy || 0);
    if (spd > 0.25) {                          // velocity vector, 1 s of travel
      cx.beginPath();
      cx.moveTo(X(o.x), Y(o.y));
      cx.lineTo(X(o.x + o.vx), Y(o.y + o.vy));
      cx.strokeStyle = col; cx.lineWidth = 1.8; cx.stroke();
    }
    // label
    const lx = X(o.x) + 9, ly = Y(o.y) - (o.width || 1) * VIEW.s / 2 - 8;
    const l1 = `${SHORT[o.type] || o.type} ${o.id}`;
    const l2 = `v ${spd.toFixed(1)} m/s`;
    cx.font = "600 10px ui-monospace,Menlo,monospace";
    const w = Math.max(cx.measureText(l1).width, cx.measureText(l2).width) + 10;
    cx.fillStyle = "rgba(8,11,16,.82)";
    cx.fillRect(lx - 4, ly - 20, w, 26);
    cx.fillStyle = col; cx.fillText(l1, lx, ly - 9);
    cx.fillStyle = "#93a4b8"; cx.fillText(l2, lx, ly + 2);
  });
  cx.restore();
}

function drawEgo(e, v, braking) {
  const off = v.footprint_center_offset || 0;
  const bx = e.x + off * Math.cos(e.yaw), by = e.y + off * Math.sin(e.yaw);
  cx.save();
  poly(box(bx, by, e.yaw, v.length, v.width), true);
  cx.fillStyle = braking ? "rgba(239,68,68,.42)" : "rgba(34,211,238,.30)";
  cx.fill();
  cx.strokeStyle = braking ? "#ef4444" : "#22d3ee";
  cx.lineWidth = 2.4; cx.shadowColor = cx.strokeStyle; cx.shadowBlur = 15;
  cx.stroke(); cx.shadowBlur = 0;
  // heading nose
  cx.beginPath();
  cx.moveTo(X(bx + Math.cos(e.yaw) * v.length * .5), Y(by + Math.sin(e.yaw) * v.length * .5));
  cx.lineTo(X(bx + Math.cos(e.yaw) * (v.length * .5 + 1.6)), Y(by + Math.sin(e.yaw) * (v.length * .5 + 1.6)));
  cx.strokeStyle = cx.strokeStyle; cx.lineWidth = 2; cx.stroke();
  cx.restore();
}

function drawScale() {
  const m = 10, px = m * VIEW.s;
  cx.save();
  cx.strokeStyle = "rgba(150,175,200,.5)"; cx.lineWidth = 1.5;
  const x0 = VIEW.w - px - 22, y0 = VIEW.h - 22;
  cx.beginPath();
  cx.moveTo(x0, y0 - 4); cx.lineTo(x0, y0); cx.lineTo(x0 + px, y0); cx.lineTo(x0 + px, y0 - 4);
  cx.stroke();
  cx.fillStyle = "rgba(150,175,200,.75)"; cx.font = "600 10px ui-monospace,Menlo,monospace";
  cx.fillText("10 m", x0 + px / 2 - 13, y0 - 7);
  cx.restore();
}

/* ══════════════ report tabs ══════════════ */
async function loadResults() {
  if (S.results) return;
  const d = await api("/api/results");
  if (d.error) { toast(d.error, true); return; }
  S.results = d;
  renderValidation(d);
  renderPerception(d);
}

function renderValidation(d) {
  const h = d.headline, runs = d.scenarios || [];
  $("v-cards").innerHTML = [
    ["ok", h.completion, "runs completed"],
    ["ok", h.collisions, "collisions"],
    ["", h.scenarios, "scenarios"],
    ["", h.runs, "total runs"],
    ["warn", fmt(h.worst_min_clearance_m, 2) + " m", "worst clearance"],
    ["warn", fmt(h.worst_p95_replan_latency_ms, 1) + " ms", "worst p95 re-plan"],
    ["", fmt(h.worst_single_replan_latency_ms, 1) + " ms", "worst single re-plan"],
    ["", h.emergency_brakes_total, "emergency brakes"],
    ["ok", h.tests.passed, "tests passed"],
    ["", h.tests.failed, "tests failed"],
    ["", h.tests.skipped, "tests skipped"],
    ["", h.replans_total.toLocaleString(), "re-planning cycles"],
  ].map(([c, v, l]) => `<div class="hc ${c}"><b>${v}</b><span>${l}</span></div>`).join("");

  const cl = [...runs].sort((a, b) => a.min_clearance_m - b.min_clearance_m);
  $("chart-clear").innerHTML = cl.map((r) => {
    const pct = Math.min(100, r.min_clearance_m / 3.0 * 100);
    const col = r.min_clearance_m < 0.7 ? "#f59e0b" : r.min_clearance_m < 1.0 ? "#38bdf8" : "#22c55e";
    return `<div class="crow"><span>${r.scenario.replace(/_/g, " ").toLowerCase()} · ${r.mode === "sensors" ? "S" : "GT"}</span>
      <div class="ctrack"><div class="cfill" style="width:${pct}%;background:${col}"></div></div>
      <b>${fmt(r.min_clearance_m, 2)}</b></div>`;
  }).join("");

  const lat = [...runs].sort((a, b) => b.replan_latency_ms.p95 - a.replan_latency_ms.p95);
  $("chart-lat").innerHTML = lat.map((r) => {
    const p95 = r.replan_latency_ms.p95, pct = Math.min(100, p95 / 120 * 100);
    const col = p95 > 100 ? "#ef4444" : p95 > 55 ? "#f59e0b" : "#22d3ee";
    return `<div class="crow"><span>${r.scenario.replace(/_/g, " ").toLowerCase()} · ${r.mode === "sensors" ? "S" : "GT"}</span>
      <div class="ctrack"><div class="cfill" style="width:${pct}%;background:${col}"></div>
        <div class="cbudget" style="left:${100 / 120 * 100}%"></div></div>
      <b>${fmt(p95, 1)}</b></div>`;
  }).join("");

  $("v-table").innerHTML =
    `<thead><tr><th>Scenario</th><th>Mode</th><th>Result</th><th class="n">Coll.</th>
      <th class="n">Min clear</th><th class="n">Replans</th><th class="n">p95 ms</th>
      <th class="n">Max ms</th><th class="n">EB</th><th class="n">Rev m</th><th class="n">Dur s</th></tr></thead><tbody>` +
    runs.map((r) => `<tr>
      <td>${r.scenario.replace(/_/g, " ")}</td>
      <td><span class="mode-t ${r.mode}">${r.mode === "sensors" ? "SENSORS" : "GROUND TRUTH"}</span></td>
      <td class="${r.completed ? "pass" : "fail"}">${r.termination}</td>
      <td class="n ${r.collisions ? "fail" : "pass"}">${r.collisions}</td>
      <td class="n">${fmt(r.min_clearance_m, 2)}</td>
      <td class="n">${r.replans}</td>
      <td class="n">${fmt(r.replan_latency_ms.p95, 1)}</td>
      <td class="n">${fmt(r.replan_latency_ms.max, 1)}</td>
      <td class="n">${r.emergency_brakes}</td>
      <td class="n">${fmt(r.reverse_distance_m, 1)}</td>
      <td class="n">${fmt(r.duration_s, 1)}</td></tr>`).join("") + "</tbody>";

  $("v-check-list").innerHTML = Object.entries(d.safety_checks || {})
    .map(([k, v]) => `<div class="chk"><i>${v ? "✓" : "✕"}</i><span>${k}</span></div>`).join("");
}

function renderPerception(d) {
  const ab = d.detector_ab, A = ab.arms.A_coco_baseline, B = ab.arms.B_uvh26_finetuned,
        C = ab.arms.C_uvh26_oversampled;
  $("p-cards").innerHTML = [
    ["", fmt(A.mAP, 3), "COCO baseline mAP"],
    ["ok", fmt(B.mAP, 3), "fine-tuned mAP · shipped"],
    ["", fmt(C.mAP, 3), "oversampled mAP · rejected"],
    ["ok", "+" + fmt(B.recall - A.recall, 3), "recall gain"],
    ["warn", fmt(B.latency_ms, 0) + " ms", "inference latency"],
    ["", ab.test_images, "held-out test images"],
  ].map(([c, v, l]) => `<div class="hc ${c}"><b>${v}</b><span>${l}</span></div>`).join("");

  const classes = ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"];
  const MAXR = 0.6;
  $("p-bars").innerHTML = classes.map((k) => {
    const a = A.per_class[k], b = B.per_class[k];
    const ra = a.recall || 0, rb = b.recall || 0, dlt = rb - ra;
    const col = dlt > 0.001 ? "#22d3ee" : "#ef4444";
    const tag = k === "AUTO_RICKSHAW" && ra === 0 ? "newly detectable"
      : dlt > 0 ? "improved" : rb === 0 ? "collapsed" : "regressed";
    return `<div class="pb-row">
      <div class="pb-hd"><b>${k.replace("_", "-").toLowerCase()}</b>
        <em>${a.gt} boxes · ${dlt >= 0 ? "+" : ""}${dlt.toFixed(3)} · ${tag}</em></div>
      <div class="pb-t"><div class="pb-f" style="width:${ra / MAXR * 100}%;background:#475569"></div></div>
      <div class="pb-t"><div class="pb-f" style="width:${rb / MAXR * 100}%;background:${col}"></div></div>
    </div>`;
  }).join("") + `<div class="pb-legend">
      <span><i style="background:#475569"></i>COCO baseline</span>
      <span><i style="background:#22d3ee"></i>Fine-tuned — improved</span>
      <span><i style="background:#ef4444"></i>Fine-tuned — regressed</span></div>`;

  const row = (label, f, dp) => `<tr><td>${label}</td>
    <td class="n">${fmt(f(A), dp)}</td><td class="n">${fmt(f(B), dp)}</td><td class="n">${fmt(f(C), dp)}</td></tr>`;
  const cls = (k) => `<tr><td>${k.replace("_", "-").toLowerCase()} recall
      <span class="mode-t">${A.per_class[k].gt} gt</span></td>
    <td class="n">${fmt(A.per_class[k].recall, 3)}</td>
    <td class="n ${B.per_class[k].recall >= A.per_class[k].recall ? "pass" : "fail"}">${fmt(B.per_class[k].recall, 3)}</td>
    <td class="n">${fmt(C.per_class[k].recall, 3)}</td></tr>`;
  $("p-table").innerHTML =
    `<thead><tr><th>Metric</th><th class="n">A · COCO</th><th class="n">B · fine-tuned</th>
      <th class="n">C · oversampled</th></tr></thead><tbody>` +
    row("Precision", (x) => x.precision, 3) + row("Recall", (x) => x.recall, 3) +
    row("mAP @ 0.5", (x) => x.mAP, 3) + row("Latency (ms)", (x) => x.latency_ms, 0) +
    classes.map(cls).join("") + "</tbody>";

  $("p-limits").innerHTML = [
    "<b>UVH-26 is elevated CCTV imagery, not dashcam.</b> The held-out test split is CCTV too, so CCTV-to-dashcam transfer is entirely unvalidated.",
    "<b>Bicycle detection collapsed</b> to 0.000 recall on 32 test boxes against 106 training boxes. Class-aware oversampling was tested (arm C) and did not fix it.",
    "<b>Bus and truck regressed.</b> Fine-tuning on a small dataset shifts the head toward what that dataset contains. This is not a uniform improvement.",
    "<b>Pedestrians and animals are untouched</b> — UVH-26 contains neither class.",
    `<b>Latency ${fmt(B.latency_ms, 0)} ms per image</b> is far outside a 100 ms planning budget. This is an offline evidence pipeline, not a real-time front end.`,
  ].map((t) => `<li>${t}</li>`).join("");
}

/* ══════════════ boot ══════════════ */
(async function boot() {
  await loadScenarios();
  connect();
  sizeCanvas();
  clearPanels();
  const st = await api("/api/state");
  if (st.status === "running" || st.status === "paused") {
    S.running = true; $("btn-run").disabled = true; $("btn-pause").disabled = false;
  }
})();
