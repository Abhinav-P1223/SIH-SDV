/* Application bootstrap: wiring, tabs, transport, demo mode. */
"use strict";

import { Scene3D } from "./scene.js";
import { Clock, Replay, loadReplay } from "./replay.js";
import {
  $, toast, updatePanels, findWhy, renderWhy, renderEvents, highlightEvent,
  renderMarks, updateBeat, showInspector,
} from "./panels.js";
import { renderValidation, renderPerception, renderScenarios } from "./reports.js";
import { buildStory, renderStory, highlightChapter, showCaption } from "./story.js";

const A = {
  scene: null, clock: null, replay: null, index: [], results: null,
  demo: false, demoAt: 0, selected: null, trail: [], lastIdx: -1,
  story: [], storyMode: true, tests: null, chapterIdx: -1,
};

/* ── boot ─────────────────────────────────────────────────────────────── */
async function boot() {
  A.scene = new Scene3D($("gl"));
  A.scene.onPick = (o) => { A.selected = o; showInspector(o, A.replay ? current() : {}); };
  window.addEventListener("resize", () => A.scene.resize());

  A.clock = new Clock(onTick);
  A.clock.onEnd = () => { $("tp-play").textContent = "▶"; if (A.demo) setTimeout(nextDemo, 1800); };

  wireUI();
  loop();

  const [idx, res, tst] = await Promise.all([
    fetch("/api/replays").then((r) => r.json()),
    fetch("/api/results").then((r) => r.json()),
    fetch("/api/tests").then((r) => r.json()).catch(() => ({ by_scenario: {} })),
  ]);
  A.index = idx.replays || [];
  A.results = res;
  A.tests = tst;

  if (!A.index.length) {
    toast("No replays recorded. Run:  python -m ui.capture", true);
    $("intro").classList.add("gone");
    return;
  }

  const sel = $("sel-replay");
  sel.innerHTML = A.index.map((r, i) =>
    `<option value="${i}">${r.scenario.replace(/_/g, " ")} · ${r.mode}</option>`).join("");
  sel.onchange = () => open(A.index[+sel.value]);

  renderValidation(res);
  renderPerception(res);
  renderScenarios(res, A.index, (rep) => { showTab("drive"); open(rep, true); });

  await open(A.index[0]);
}

function current() {
  return A.replay.frameAt(A.replay.t0 + A.clock.t);
}

/* ── replay loading ── */
async function open(meta, autoplay) {
  if (!meta) return;
  A.clock.pause();
  $("boot").classList.add("on");
  $("boot-msg").textContent = `Loading ${meta.scenario.replace(/_/g, " ")}…`;
  try {
    A.replay = await loadReplay(meta.file);
  } catch (e) {
    $("boot").classList.remove("on");
    toast("Replay failed: " + e.message, true);
    return;
  }
  const r = A.replay;
  $("sel-replay").value = String(A.index.findIndex((x) => x.file === meta.file));

  A.scene.setRoad(r.road);
  A.trail = []; A.lastIdx = -1; A.selected = null;
  showInspector(null);

  $("ov-name").textContent = r.scenario.replace(/_/g, " ");
  $("ov-desc").textContent = r.doc.description || "";
  $("ov-mode").textContent = r.mode === "sensors" ? "SENSOR MODE" : "GROUND TRUTH";
  $("ov-frames").textContent = `${r.doc.frame_count} recorded frames`;
  $("tp-dur").textContent = r.duration.toFixed(2);

  A.clock.duration = r.duration;
  A.clock.t = 0;
  $("tp-slider").max = "1000";
  renderEvents(r, (t) => seekAbs(t));
  renderMarks(r, (t) => A.clock.seek(t));

  A.story = buildStory(r);
  A.chapterIdx = -1;
  renderStory(A.story, (rel) => A.clock.seek(rel));
  renderTests(r.scenario);
  onTick(0);

  $("boot").classList.remove("on");
  if (autoplay !== false) { A.clock.play(); $("tp-play").textContent = "❚❚"; }
}

/* ── per-tick ── */
function onTick(t) {
  const r = A.replay;
  if (!r) return;
  const abs = r.t0 + t;
  const f = r.frameAt(abs);
  const pf = r.planFrameAt(abs);

  // trail: rebuild on a backwards seek, otherwise append
  if (r.index < A.lastIdx) A.trail = [];
  for (let i = Math.max(A.lastIdx + 1, 0); i <= r.index; i++) {
    const e = r.frames[i].ego;
    const last = A.trail[A.trail.length - 1];
    if (!last || Math.hypot(e.x - last[0], e.y - last[1]) > 0.25) A.trail.push([e.x, e.y]);
  }
  A.lastIdx = r.index;
  A.scene.setTrail(A.trail);
  A.scene.update(f, pf);

  updatePanels(f, pf, r);
  renderWhy(findWhy(r, abs), r);
  highlightEvent(r, abs);
  updateBeat(r, abs);
  if (A.selected) {
    const live = (f.objects || []).find((o) => o.id === A.selected.id);
    if (live) { A.selected = live; showInspector(live, f); }
  }

  if (A.story.length) {
    const i = highlightChapter(A.story, t);
    A.chapterIdx = i;
    showCaption(A.storyMode && i >= 0 ? A.story[i] : null);
  }

  $("tp-now").textContent = t.toFixed(2);
  const sl = $("tp-slider");
  if (document.activeElement !== sl) sl.value = String((t / r.duration) * 1000);
}

function seekAbs(absT) { A.clock.seek(absT - A.replay.t0); }

/* ── UI wiring ── */
function wireUI() {
  document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  document.querySelectorAll(".cam").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".cam").forEach((x) => x.classList.toggle("on", x === b));
    A.scene.setMode(b.dataset.cam);
  }));

  $("tp-play").onclick = () => {
    A.clock.toggle();
    $("tp-play").textContent = A.clock.playing ? "❚❚" : "▶";
  };
  $("tp-back").onclick = () => A.clock.nudge(-2);
  $("tp-fwd").onclick = () => A.clock.nudge(2);
  $("tp-reset").onclick = () => { A.clock.pause(); A.clock.seek(0); $("tp-play").textContent = "▶"; };
  $("tp-speed").onchange = (e) => (A.clock.speed = parseFloat(e.target.value));
  $("tp-slider").oninput = (e) => {
    if (A.replay) A.clock.seek((+e.target.value / 1000) * A.replay.duration);
  };
  $("tp-demo").onclick = () => startDemo();
  $("tp-story").onclick = () => {
    A.storyMode = !A.storyMode;
    $("tp-story").classList.toggle("demo", A.storyMode);
    toast(A.storyMode ? "Story captions on" : "Story captions off");
    if (!A.storyMode) showCaption(null);
  };
  document.querySelectorAll(".sub").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".sub").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".sub-pane").forEach((p) =>
      p.classList.toggle("on", p.id === "pane-" + b.dataset.sub));
    $("strip-note").textContent = { story: "click a chapter to jump there",
      events: "click an event to seek", tests: "real pytest outcomes for this scenario"
    }[b.dataset.sub];
  }));
  $("tp-next").onclick = () => nextDemo();
  $("tp-prev").onclick = () => { A.demoAt = Math.max(0, A.demoAt - 2); nextDemo(); };
  $("inspect-x").onclick = () => { A.selected = null; showInspector(null); };

  $("intro-demo").onclick = () => { $("intro").classList.add("gone"); startDemo(); };
  $("intro-skip").onclick = () => $("intro").classList.add("gone");

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); $("tp-play").click(); }
    if (e.key === "ArrowLeft") A.clock.nudge(-2);
    if (e.key === "ArrowRight") A.clock.nudge(2);
    if (e.key >= "1" && e.key <= "4") document.querySelectorAll(".cam")[+e.key - 1].click();
  });
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x.dataset.tab === name));
  document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("on", s.id === "s-" + name));
  $("top-ctl").style.visibility = name === "drive" ? "visible" : "hidden";
  if (name === "drive") setTimeout(() => A.scene.resize(), 30);
}

/* ── per-scenario tests ── */
function renderTests(scenario) {
  const box = $("test-list"), badge = $("test-badge");
  const rows = (A.tests && A.tests.by_scenario && A.tests.by_scenario[scenario]) || [];
  if (!rows.length) {
    badge.textContent = "";
    box.innerHTML = `<div class="test-empty">No tests are mapped to ${scenario.replace(/_/g, " ")}.
      Run <code>python -m ui.collect_tests</code> to record them.</div>`;
    return;
  }
  const pass = rows.filter((r) => r.outcome === "PASSED").length;
  badge.textContent = `${pass}/${rows.length}`;
  const secs = rows.reduce((a, r) => a + (r.seconds || 0), 0);
  box.innerHTML =
    `<div class="test-head"><b>${pass} of ${rows.length} passed</b>
       <span>real pytest run · ${secs.toFixed(0)} s of execution</span></div>` +
    rows.map((r) => {
      const cls = r.outcome === "PASSED" ? "pass" : r.outcome === "SKIPPED" ? "skip" : "fail";
      const mark = r.outcome === "PASSED" ? "✓" : r.outcome === "SKIPPED" ? "–" : "✕";
      const sub = r.doc || `<code>${r.file.split("/").pop()}</code>`;
      return `<div class="tst ${cls}"><i>${mark}</i><div class="tst-b">
        <b>${escapeHtml(r.title)}</b><span>${sub}</span></div></div>`;
    }).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* ── demo mode ── */
function startDemo() {
  A.demo = true; A.demoAt = 0;
  showTab("drive");
  toast("Demo mode — five scenarios, in order");
  nextDemo();
}

async function nextDemo() {
  const demo = A.index.filter((r) => r.demo_rank < 90);
  if (!demo.length) { A.demo = false; return; }
  if (A.demoAt >= demo.length) { A.demo = false; toast("Demo complete"); return; }
  const m = demo[A.demoAt++];
  toast(`${A.demoAt}/${demo.length} — ${m.scenario.replace(/_/g, " ")}`);
  await open(m, true);
}

/* ── render loop ── */
function loop() {
  A.scene.render();
  requestAnimationFrame(loop);
}

boot().catch((e) => {
  console.error(e);
  $("boot").classList.remove("on");
  toast("Startup failed: " + e.message, true);
});
