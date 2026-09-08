/* Telemetry panels, "why did we replan?", event timeline, storytelling beats.
   Every value shown is read from a recorded telemetry field. Where a field is genuinely absent the
   panel shows "NOT AVAILABLE" rather than a placeholder. */
"use strict";

import { SHORT } from "./replay.js";

export const $ = (id) => document.getElementById(id);
export const NA = "NOT AVAILABLE";
export const fmt = (v, d = 2) =>
  (v === null || v === undefined || Number.isNaN(v)) ? NA : Number(v).toFixed(d);
export const deg = (r) => (r === null || r === undefined ? NA : (r * 180 / Math.PI).toFixed(1));

export function toast(msg, err) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast on" + (err ? " err" : "");
  clearTimeout(t._t);
  t._t = setTimeout(() => (t.className = "toast"), 3400);
}

/* ── main telemetry panels ── */
export function updatePanels(f, planFrame, replay) {
  const e = f.ego || {}, c = f.control || {}, r = f.risk || {},
        b = f.behavior, s = f.safety || {}, m = f.metrics || {};
  const plan = planFrame && planFrame.plan;

  $("clock").textContent = f.t.toFixed(3);

  document.body.dataset.risk = r.max_level || "NONE";
  $("risk-chip").textContent = r.max_level || "NONE";
  if (b) {
    $("beh-state").textContent = b.state.replace(/_/g, " ");
    $("beh-why").textContent = b.reason || "";
  }

  // TTC is infinite when nothing is closing — that is a real value, not a gap.
  $("m-ttc").textContent = r.min_ttc_current_speed == null ? "∞" : fmt(r.min_ttc_current_speed, 2);
  $("m-score").textContent = fmt(r.max_score, 3);
  $("m-clear").textContent = fmt(m.minimum_obstacle_clearance, 2);
  $("m-worst").textContent = r.worst_object_id || "none";

  $("m-speed").textContent = fmt(e.longitudinal_velocity, 2);
  $("m-steer").textContent = deg(e.steering_angle);
  $("m-acc").textContent = fmt(c.acceleration, 2);
  $("m-brake").textContent = fmt(c.brake, 2);

  // both counts on one row: published tracks vs how many objects the simulator actually holds
  $("m-tracks").textContent = `${(f.objects || []).length} / ${(f.agents || []).length}`;
  $("m-cand").textContent = plan ? plan.candidate_count : NA;
  $("m-feas").textContent = plan ? plan.feasible_count : NA;
  // compact_frame does not carry replanning_count; count recorded planning cycles instead.
  $("m-replans").textContent = replay ? replay.plansUpTo(f.t) : NA;
  $("m-lat").textContent = plan ? fmt(plan.latency_ms, 1) : NA;

  const eb = !!s.override_active;
  flag("s-eb", eb, eb ? "ACTIVE" : "INACTIVE", "hot");
  $("eb").className = "eb" + (eb ? " on" : "");
  const coll = (m.collision_count || 0) > 0;
  flag("s-coll", coll, coll ? String(m.collision_count) : "NONE", "hot");
  const rev = e.longitudinal_velocity < -0.05;
  flag("s-rev", rev, rev ? "ENGAGED" : "NO", "act");
}

function flag(id, on, text, cls) {
  const el = $(id);
  el.className = "s" + (on ? " " + cls : "");
  el.querySelector("b").textContent = text;
}

/* ── why did we replan? ──────────────────────────────────────────────────
   Fires on a behaviour-state change. The cause line is the FSM's own `reason`
   string prefixed with the highest-risk track. Nothing causal is invented. */
export function findWhy(replay, t) {
  return replay.transitionAt(t);
}

let _whyKey = null;

export function renderWhy(hit, replay, nowFrame) {
  // Before the first decision change the panel would sit empty and tall. Show the CURRENT planner
  // state instead, plainly labelled — an empty box reads as broken, and this is all measured.
  if (!hit) {
    if (_whyKey === "none") return;
    _whyKey = "none";
    $("why-cause").textContent =
      "No decision change yet — the vehicle is still in its opening behaviour state. " +
      "Showing the current planner state.";
    $("why-time").textContent = nowFrame ? `t = ${nowFrame.t.toFixed(2)} s` : "";
    $("why-row").innerHTML = nowFrame ? cells(nowFrame, replay, null) : "";
    return;
  }
  const key = hit.i;
  if (key === _whyKey) return;          // the panel only changes on a transition, not every frame
  _whyKey = key;
  const f = hit.f;
  const b = f.behavior, r = f.risk || {};
  const worst = (f.objects || []).find((o) => o.id === r.worst_object_id);

  $("why-time").textContent = `t = ${f.t.toFixed(2)} s`;
  $("why-cause").textContent =
    (worst ? `${SHORT[worst.type] || worst.type} ${worst.id} — ` : "") + (b.reason || "—");

  $("why-row").innerHTML = cells(f, replay, hit.from);
}

/** The measured row shared by both states of the panel. `from` is null when no transition yet. */
function cells(f, replay, from) {
  const b = f.behavior || {}, r = f.risk || {}, c = f.control || {};
  let plan = f.plan;
  if (!plan && replay) { const pf = replay.planFrameAt(f.t); plan = pf ? pf.plan : null; }
  return [
    ["TTC physical", r.min_ttc_current_speed == null ? "∞" : fmt(r.min_ttc_current_speed, 2) + " s"],
    ["Feasible paths", plan ? `${plan.feasible_count} / ${plan.candidate_count}` : NA],
    [from ? "Transition" : "Behaviour", from ? `${from} → ${b.state}` : (b.state || NA)],
    ["Target speed", b.force_stop ? "STOP" : fmt(b.target_speed, 1) + " m/s"],
    ["Response", fmt((c.acceleration || 0) - (c.brake || 0), 2) + " m/s²"],
    ["Clearance", plan && plan.selected && plan.selected.min_clearance != null
      ? fmt(plan.selected.min_clearance, 2) + " m" : NA],
    ["Selected", plan ? plan.selected_id : NA],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
}

/* ── event timeline ── */
export function renderEvents(replay, onSeek) {
  _whyKey = null;                        // new replay: force the why panel to repaint
  _lastEvIdx = -2;
  const box = $("events");
  box.innerHTML = "";
  replay.events.forEach((e, i) => {
    const d = document.createElement("div");
    d.className = "ev k-" + e.kind;
    d.dataset.t = e.t;
    d.innerHTML = `<time>${e.t.toFixed(2)}s</time><span></span>`;
    d.querySelector("span").textContent = e.text;
    d.onclick = () => onSeek(e.t);
    box.appendChild(d);
  });
}

let _lastEvIdx = -2;

export function highlightEvent(replay, t) {
  const box = $("events");
  let idx = -1;
  for (let i = 0; i < replay.events.length; i++) {
    if (replay.events[i].t > t + 1e-6) break;      // events are sorted: stop at the first future one
    idx = i;
  }
  if (idx === _lastEvIdx) return;                  // nothing to repaint
  if (_lastEvIdx >= 0 && box.children[_lastEvIdx]) box.children[_lastEvIdx].classList.remove("now");
  _lastEvIdx = idx;
  if (idx >= 0 && box.children[idx]) box.children[idx].classList.add("now");
  if (idx >= 0) {
    // keep the current event centred; offsetTop is meaningful because .ev-wrap is positioned
    const el = box.children[idx];
    const want = el.offsetTop - box.clientHeight / 2 + el.clientHeight / 2;
    if (Math.abs(box.scrollTop - want) > 12) box.scrollTop = Math.max(0, want);
  }
}

export function renderMarks(replay, onSeek) {
  const box = $("tp-marks");
  box.innerHTML = "";
  const col = { eb: "#f04438", rev: "#a78bfa", state: "#2ad4ee", risk: "#f5a524", goal: "#2fce7f" };
  replay.events.forEach((e) => {
    if (!col[e.kind]) return;
    const i = document.createElement("i");
    i.style.left = ((e.t - replay.t0) / replay.duration * 100) + "%";
    i.style.background = col[e.kind];
    i.title = `${e.t.toFixed(2)}s — ${e.text}`;
    i.onclick = () => onSeek(e.t - replay.t0);
    box.appendChild(i);
  });
}

/* ── object inspector ── */
export function showInspector(o, f) {
  if (!o) { $("inspect").hidden = true; return; }
  const r = f.risk || {};
  const spd = Math.hypot(o.vx || 0, o.vy || 0);
  const pred = (f.predictions || []).find((p) => p.id === o.id);
  const worst = o.id === r.worst_object_id;

  $("inspect").hidden = false;
  $("ins-title").textContent = `${(SHORT[o.type] || o.type)} · ${o.id}`;
  const chip = $("ins-risk");
  chip.textContent = worst ? (r.max_level || "") : "";
  chip.style.background = worst ? "rgba(240,68,56,.18)" : "transparent";
  chip.style.color = worst ? "#f04438" : "transparent";

  const cells = [
    ["Position", `${o.x.toFixed(1)}, ${o.y.toFixed(1)}`],
    ["Speed", spd.toFixed(2) + " m/s"],
    ["Heading", deg(o.heading) + "°"],
    ["Size", `${(o.length || 0).toFixed(1)}×${(o.width || 0).toFixed(1)} m`],
    ["Confidence", o.confidence != null ? o.confidence.toFixed(2) : NA],
    ["Prediction", pred ? `${pred.x.length} pts` : NA],
  ];
  if (worst) {
    cells.push(["TTC physical", r.min_ttc_current_speed == null ? "∞"
      : fmt(r.min_ttc_current_speed, 2) + " s"]);
    cells.push(["Risk score", fmt(r.max_score, 3)]);
  }
  $("ins-grid").innerHTML = cells.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
}
