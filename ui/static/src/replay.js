/* Replay data model.
   Loads a recorded run and exposes frame lookup by simulation time plus a playback clock.
   Every frame here was published by the frozen autonomy stack during a real run — see ui/capture.py.
   Nothing is interpolated, smoothed or synthesised. */
"use strict";

export const SHORT = {
  PEDESTRIAN: "PED", BICYCLE: "BIKE", MOTORCYCLE: "MOTO", AUTO_RICKSHAW: "AUTO",
  CAR: "CAR", BUS: "BUS", TRUCK: "TRUCK", PUSHCART: "CART", CATTLE: "COW", UNKNOWN: "UNK",
};
export const CLASS_COLOR = {
  PEDESTRIAN: 0xf472b6, BICYCLE: 0xc084fc, MOTORCYCLE: 0xfb8b3c, AUTO_RICKSHAW: 0xf5a524,
  CAR: 0x4b8ef8, BUS: 0x2fce7f, TRUCK: 0x2dd4bf, PUSHCART: 0x9aa5b4,
  CATTLE: 0xf04438, UNKNOWN: 0x7d8fa6,
};

export class Replay {
  constructor(doc) {
    this.doc = doc;
    this.frames = doc.frames;
    this.t0 = this.frames[0].t;
    this.duration = doc.duration_s - this.t0;
    this.times = this.frames.map((f) => f.t);
    this.events = buildEvents(this.frames);
    this.transitions = buildTransitions(this.frames);   // precomputed: findWhy was O(frames)/tick
    this.index = 0;
  }

  get scenario() { return this.doc.scenario; }
  get mode() { return this.doc.mode; }
  get summary() { return this.doc.summary; }
  get road() { return this.doc.road; }
  get vehicle() { return this.doc.vehicle; }

  /** Index of the last frame at or before absolute time t. Binary search — replays run to 1300 frames. */
  indexAt(t) {
    const a = this.times;
    let lo = 0, hi = a.length - 1;
    if (t <= a[0]) return 0;
    if (t >= a[hi]) return hi;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (a[mid] <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  frameAt(t) { this.index = this.indexAt(t); return this.frames[this.index]; }

  /** Planning cycles recorded at or before t — a real count of re-plans so far. */
  plansUpTo(t) {
    if (!this._cum) {
      this._cum = new Int32Array(this.frames.length);
      let n = 0;
      this.frames.forEach((f, i) => { if (f.planning_cycle) n++; this._cum[i] = n; });
    }
    return this._cum[this.indexAt(t)];
  }

  /** The last behaviour-state change at or before t, or null. */
  transitionAt(t) {
    const a = this.transitions;
    let lo = 0, hi = a.length - 1, r = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (a[mid].f.t <= t) { r = a[mid]; lo = mid + 1; } else hi = mid - 1;
    }
    return r;
  }

  /** Most recent frame that carried a plan — between planning cycles the fan is not re-published. */
  planFrameAt(t) {
    for (let i = this.indexAt(t); i >= 0; i--) if (this.frames[i].plan) return this.frames[i];
    return null;
  }
}

/* ── event extraction ──────────────────────────────────────────────────────
   Events are derived from measured state changes only. No causal claim is made
   beyond "this value changed at this timestamp". */
function buildEvents(frames) {
  const ev = [];
  const push = (t, kind, text) => ev.push({ t, kind, text });
  let state = null, risk = null, eb = false, rev = false, sel = null;
  const seen = new Set();
  let lastTrackEv = -99;

  frames.forEach((f) => {
    const t = f.t;
    (f.objects || []).forEach((o) => {
      if (!seen.has(o.id)) {
        seen.add(o.id);
        if (t - lastTrackEv > 0.25 || seen.size < 4) {  // avoid a burst of identical rows at t=0
          push(t, "track", `track confirmed · ${SHORT[o.type] || o.type} ${o.id}`);
          lastTrackEv = t;
        }
      }
    });
    const b = f.behavior;
    if (b && b.state !== state) {
      if (state !== null) {
        const kind = b.state === "EMERGENCY_BRAKE" ? "eb" : b.state === "REVERSING" ? "rev" : "state";
        push(t, kind, `behaviour ${state} → ${b.state}`);
      }
      state = b.state;
    }
    const r = f.risk || {};
    if (r.max_level && r.max_level !== risk) {
      if (risk !== null) push(t, "risk", `risk ${risk} → ${r.max_level}`);
      risk = r.max_level;
    }
    const on = !!(f.safety && f.safety.override_active);
    if (on && !eb) push(t, "eb", `safety override · ${(f.safety.reason || "").slice(0, 74)}`);
    eb = on;
    const backing = f.ego && f.ego.longitudinal_velocity < -0.05;
    if (backing && !rev) push(t, "rev", "reverse manoeuvre engaged");
    rev = backing;
    // The selected id changes almost every cycle because the target speed is baked into it.
    // Only the LATERAL decision is a story beat, so key on the offset part of the id.
    if (f.plan && f.plan.selected_id) {
      const lat = String(f.plan.selected_id).split("_v")[0];
      if (sel !== null && lat !== sel) push(t, "plan", `re-plan · lateral target ${lat}`);
      sel = lat;
    }
  });
  const last = frames[frames.length - 1];
  push(last.t, "goal", `${last.metrics ? last.metrics.termination_reason || "run complete" : "run complete"}`);
  return ev;
}

function buildTransitions(frames) {
  const out = [];
  let prev = null;
  frames.forEach((f, i) => {
    if (!f.behavior) return;
    if (prev !== null && f.behavior.state !== prev) out.push({ f, i, from: prev });
    prev = f.behavior.state;
  });
  return out;
}

/* ── playback clock ─────────────────────────────────────────────────────── */
export class Clock {
  constructor(onTick) {
    this.t = 0; this.speed = 1; this.playing = false;
    this.duration = 0; this.onTick = onTick;
    this._last = 0; this._raf = null;
  }
  play() {
    if (this.playing) return;
    if (this.t >= this.duration - 1e-6) this.t = 0;
    this.playing = true; this._last = performance.now();
    const step = (now) => {
      if (!this.playing) return;
      const dt = Math.min((now - this._last) / 1000, 0.25);
      this._last = now;
      this.t = Math.min(this.t + dt * this.speed, this.duration);
      this.onTick(this.t);
      if (this.t >= this.duration - 1e-6) { this.playing = false; this.onEnd && this.onEnd(); return; }
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }
  pause() { this.playing = false; if (this._raf) cancelAnimationFrame(this._raf); }
  toggle() { this.playing ? this.pause() : this.play(); }
  seek(t) {
    this.t = Math.max(0, Math.min(t, this.duration));
    this._last = performance.now();
    this.onTick(this.t);
  }
  nudge(d) { this.seek(this.t + d); }
}

export async function loadReplay(file, onProgress) {
  onProgress && onProgress(`Loading ${file}…`);
  const r = await fetch(`/api/replay/${file}`);
  if (!r.ok) throw new Error((await r.json()).error || `HTTP ${r.status}`);
  return new Replay(await r.json());
}
