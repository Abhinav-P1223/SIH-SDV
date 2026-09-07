/* Validation, Perception and Scenario screens.
   Every figure is read from docs/FINAL_SYSTEM_RESULTS.json, served verbatim by the server.
   Nothing on these screens is computed from anything else. */
"use strict";

import { $, fmt, NA } from "./panels.js";

const pretty = (s) => s.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

export function renderValidation(d) {
  const h = d.headline, runs = d.scenarios || [];
  $("v-hero").innerHTML = [
    ["", h.scenarios, "scenarios"],
    ["", h.runs, "runs"],
    ["ok", h.completion, "completed"],
    ["ok", h.collisions, "collisions"],
    ["warn", fmt(h.worst_min_clearance_m, 2) + " m", "worst min clearance"],
    ["warn", fmt(h.worst_p95_replan_latency_ms, 1) + " ms", "worst p95 re-plan"],
    ["", fmt(h.worst_single_replan_latency_ms, 1) + " ms", "worst single re-plan"],
    ["", h.emergency_brakes_total, "emergency brakes"],
    ["ok", h.tests.passed, "tests passed"],
    ["", h.tests.failed, "tests failed"],
    ["", h.tests.skipped, "tests skipped"],
    ["", h.replans_total.toLocaleString(), "re-plan cycles"],
  ].map(([c, v, l]) => `<div class="hc ${c}"><b>${v}</b><span>${l}</span></div>`).join("");

  const label = (r) => `${pretty(r.scenario)} · ${r.mode === "sensors" ? "SENSORS" : "GT"}`;

  $("ch-clear").innerHTML = [...runs].sort((a, b) => a.min_clearance_m - b.min_clearance_m)
    .map((r) => {
      const pct = Math.min(100, r.min_clearance_m / 3 * 100);
      const col = r.min_clearance_m < 0.7 ? "#f5a524" : r.min_clearance_m < 1 ? "#4b8ef8" : "#2fce7f";
      return `<div class="crow"><span>${label(r)}</span>
        <div class="ctrack"><div class="cfill" style="width:${pct}%;background:${col}"></div></div>
        <b>${fmt(r.min_clearance_m, 2)}</b></div>`;
    }).join("");

  $("ch-lat").innerHTML = [...runs].sort((a, b) => b.replan_latency_ms.p95 - a.replan_latency_ms.p95)
    .map((r) => {
      const p95 = r.replan_latency_ms.p95, pct = Math.min(100, p95 / 120 * 100);
      const col = p95 > 100 ? "#f04438" : p95 > 55 ? "#f5a524" : "#2ad4ee";
      return `<div class="crow"><span>${label(r)}</span>
        <div class="ctrack"><div class="cfill" style="width:${pct}%;background:${col}"></div>
          <div class="cbud" style="left:83.33%"></div></div>
        <b>${fmt(p95, 1)}</b></div>`;
    }).join("");

  // Ground truth vs sensors — aggregated from the same 22 rows, nothing else.
  const agg = (mode) => {
    const rs = runs.filter((r) => r.mode === mode);
    return {
      n: rs.length,
      completed: rs.filter((r) => r.completed).length,
      collisions: rs.reduce((a, r) => a + r.collisions, 0),
      eb: rs.reduce((a, r) => a + r.emergency_brakes, 0),
      clear: Math.min(...rs.map((r) => r.min_clearance_m)),
      replans: rs.reduce((a, r) => a + r.replans, 0),
      p95: Math.max(...rs.map((r) => r.replan_latency_ms.p95)),
    };
  };
  const g = agg("ground_truth"), s = agg("sensors");
  const cmp = (k, f2) => `<tr><td>${k}</td><td class="n">${f2(g)}</td><td class="n">${f2(s)}</td></tr>`;
  $("v-modes").innerHTML =
    `<thead><tr><th>Metric</th><th class="n">Ground truth</th><th class="n">Sensor mode</th></tr></thead><tbody>` +
    cmp("Runs", (x) => x.n) +
    cmp("Completed", (x) => `${x.completed} / ${x.n}`) +
    cmp("Collisions", (x) => x.collisions) +
    cmp("Emergency brakes", (x) => x.eb) +
    cmp("Worst min clearance (m)", (x) => fmt(x.clear, 3)) +
    cmp("Total re-plans", (x) => x.replans.toLocaleString()) +
    cmp("Worst p95 latency (ms)", (x) => fmt(x.p95, 2)) + "</tbody>";

  $("v-table").innerHTML =
    `<thead><tr><th>Scenario</th><th>Mode</th><th>Result</th><th class="n">Coll.</th>
      <th class="n">Min clear</th><th class="n">Replans</th><th class="n">p95 ms</th>
      <th class="n">Max ms</th><th class="n">EB</th><th class="n">Rev m</th>
      <th class="n">Dur s</th></tr></thead><tbody>` +
    runs.map((r) => `<tr>
      <td>${pretty(r.scenario)}</td>
      <td><span class="tag ${r.mode}">${r.mode === "sensors" ? "SENSORS" : "GROUND TRUTH"}</span></td>
      <td class="${r.completed ? "pass" : "fail"}">${r.termination}</td>
      <td class="n ${r.collisions ? "fail" : "pass"}">${r.collisions}</td>
      <td class="n">${fmt(r.min_clearance_m, 2)}</td>
      <td class="n">${r.replans}</td>
      <td class="n">${fmt(r.replan_latency_ms.p95, 1)}</td>
      <td class="n">${fmt(r.replan_latency_ms.max, 1)}</td>
      <td class="n">${r.emergency_brakes}</td>
      <td class="n">${fmt(r.reverse_distance_m, 1)}</td>
      <td class="n">${fmt(r.duration_s, 1)}</td></tr>`).join("") + "</tbody>";

  $("v-checks").innerHTML = Object.entries(d.safety_checks || {})
    .map(([k, v]) => `<div class="chk"><i>${v ? "✓" : "✕"}</i><span>${k}</span></div>`).join("");
}

export function renderPerception(d) {
  const ab = d.detector_ab;
  const A = ab.arms.A_coco_baseline, B = ab.arms.B_uvh26_finetuned, C = ab.arms.C_uvh26_oversampled;

  $("p-hero").innerHTML = [
    ["", fmt(A.mAP, 3), "COCO baseline mAP"],
    ["ok", fmt(B.mAP, 3), "fine-tuned mAP · shipped"],
    ["", fmt(C.mAP, 3), "oversampled mAP · rejected"],
    ["ok", "+" + fmt(B.recall - A.recall, 3), "overall recall gain"],
    ["warn", fmt(B.latency_ms, 0) + " ms", "inference latency"],
    ["", ab.test_images, "held-out test images"],
  ].map(([c, v, l]) => `<div class="hc ${c}"><b>${v}</b><span>${l}</span></div>`).join("");

  const classes = ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"];
  const MAX = 0.6;
  $("p-bars").innerHTML = classes.map((k) => {
    const a = A.per_class[k], b = B.per_class[k];
    const ra = a.recall || 0, rb = b.recall || 0, dl = rb - ra;
    const col = dl > 0.001 ? "#2ad4ee" : "#f04438";
    const tag = (k === "AUTO_RICKSHAW" && ra === 0) ? "newly detectable"
      : dl > 0 ? "improved" : rb === 0 ? "collapsed" : "regressed";
    return `<div><div class="pb-hd"><b>${k.replace("_", "-").toLowerCase()}</b>
        <em>${a.gt} gt boxes · ${dl >= 0 ? "+" : ""}${dl.toFixed(3)} · ${tag}</em></div>
      <div class="pb-t"><div class="pb-f" style="width:${ra / MAX * 100}%;background:#475569"></div></div>
      <div class="pb-t"><div class="pb-f" style="width:${rb / MAX * 100}%;background:${col}"></div></div></div>`;
  }).join("") + `<div class="pb-lg">
      <span><i style="background:#475569"></i>COCO baseline</span>
      <span><i style="background:#2ad4ee"></i>Fine-tuned — improved</span>
      <span><i style="background:#f04438"></i>Fine-tuned — regressed</span></div>`;

  const row = (label, get, dp) => `<tr><td>${label}</td>
    <td class="n">${fmt(get(A), dp)}</td><td class="n">${fmt(get(B), dp)}</td>
    <td class="n">${fmt(get(C), dp)}</td></tr>`;
  const cls = (k) => {
    const a = A.per_class[k], b = B.per_class[k], c = C.per_class[k];
    return `<tr><td>${k.replace("_", "-").toLowerCase()} recall <span class="tag">${a.gt} gt</span></td>
      <td class="n">${fmt(a.recall, 3)}</td>
      <td class="n ${b.recall >= a.recall ? "pass" : "fail"}">${fmt(b.recall, 3)}</td>
      <td class="n">${fmt(c.recall, 3)}</td></tr>`;
  };
  $("p-table").innerHTML =
    `<thead><tr><th>Metric</th><th class="n">A · COCO baseline</th>
      <th class="n">B · UVH-26 fine-tuned</th><th class="n">C · oversampled</th></tr></thead><tbody>` +
    row("Precision", (x) => x.precision, 3) +
    row("Recall", (x) => x.recall, 3) +
    row("mAP @ 0.5", (x) => x.mAP, 3) +
    row("Latency (ms)", (x) => x.latency_ms, 0) +
    classes.map(cls).join("") + "</tbody>";

  $("p-limits").innerHTML = [
    "<b>UVH-26 is elevated CCTV imagery, not dashcam.</b> The held-out test split is CCTV too, so CCTV-to-dashcam transfer is entirely unvalidated.",
    "<b>Bicycle detection collapsed</b> to 0.000 recall on 32 test boxes against 106 training boxes. Class-aware oversampling was tested (arm C) and did not fix it.",
    "<b>Bus and truck regressed.</b> Fine-tuning on a small dataset shifts the head toward what that dataset contains. This is not a uniform improvement.",
    "<b>Pedestrians and animals are untouched</b> — UVH-26 contains neither class.",
    `<b>Latency ${fmt(B.latency_ms, 0)} ms per image</b> is far outside a 100 ms planning budget. This is an offline evidence pipeline, not a real-time front end.`,
    "<b>The detector consumed no camera pixels in any closed-loop run.</b> The Phase 8 simulator renders no image frames.",
  ].map((t) => `<li>${t}</li>`).join("");
}

/* ── scenario cards ── */
export function renderScenarios(results, replays, onOpen) {
  const runs = results.scenarios || [];
  const byName = {};
  runs.forEach((r) => { (byName[r.scenario] ||= []).push(r); });
  const haveReplay = {};
  replays.forEach((r) => { haveReplay[r.scenario] = r; });

  const order = replays.map((r) => r.scenario)
    .concat(Object.keys(byName).filter((n) => !haveReplay[n]));
  const seen = new Set();

  $("scn-grid").innerHTML = order.filter((n) => !seen.has(n) && seen.add(n)).map((name) => {
    const rep = haveReplay[name];
    const rows = byName[name] || [];
    const sens = rows.find((r) => r.mode === "sensors") || rows[0];
    if (!sens) return "";
    const desc = rep ? rep.description : "";
    return `<article class="scn ${rep ? "" : "noreplay"}" data-scn="${name}">
      <h4>${pretty(name)}</h4>
      <p>${desc || "Validated scenario. No replay recorded — run <code>python -m ui.capture --all</code>."}</p>
      <div class="scn-m">
        <div><span>Result</span><b class="${sens.completed ? "ok" : ""}">${sens.completed ? "PASS" : "FAIL"}</b></div>
        <div><span>Collisions</span><b class="${sens.collisions ? "" : "ok"}">${sens.collisions}</b></div>
        <div><span>Min clear</span><b>${fmt(sens.min_clearance_m, 2)}</b></div>
        <div><span>Replans</span><b>${sens.replans}</b></div>
        <div><span>p95 ms</span><b>${fmt(sens.replan_latency_ms.p95, 1)}</b></div>
        <div><span>Brakes</span><b>${sens.emergency_brakes}</b></div>
      </div>
      <div class="scn-go">${rep ? "▶ Open replay" : "No replay recorded"}</div>
    </article>`;
  }).join("");

  $("scn-grid").querySelectorAll(".scn:not(.noreplay)").forEach((el) => {
    el.onclick = () => onOpen(haveReplay[el.dataset.scn]);
  });
}
