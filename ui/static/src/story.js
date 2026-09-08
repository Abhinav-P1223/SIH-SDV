/* Story mode — turns a recorded run into a narrated, chaptered sequence.

   Every chapter is anchored to a real recorded event at a real timestamp. The caption names what
   the telemetry actually did; the numbers in it are read from the frame at that instant. Nothing
   is scripted per scenario, so a re-record produces a new story automatically. */
"use strict";

import { SHORT } from "./replay.js";

const f2 = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

/* Chapter templates, keyed by the event kind the replay already derives.
   `title` is the beat; `line(ev, f, replay)` builds the sentence from measured values. */
const BEATS = {
  start: {
    title: "NORMAL DRIVING",
    tone: "ok",
    line: (ev, f) =>
      `Cruising the drivable corridor at ${f2(f.ego.longitudinal_velocity, 1)} m/s. ` +
      `No lane markings are used — the planner works from the road edges and a direction of travel.`,
  },
  track: {
    title: "OBJECT DETECTED",
    tone: "info",
    line: (ev, f) => {
      const o = (f.objects || [])[f.objects.length - 1];
      if (!o) return "A new object entered sensor range and was confirmed as a track.";
      const spd = Math.hypot(o.vx || 0, o.vy || 0);
      return `${SHORT[o.type] || o.type} confirmed as track ${o.id}, ` +
             `moving at ${f2(spd, 1)} m/s. Fusion needed 3 consistent detections before publishing it.`;
    },
  },
  risk: {
    title: "RISK RISING",
    tone: "warn",
    line: (ev, f) => {
      const r = f.risk || {};
      const ttc = r.min_ttc_current_speed == null ? "no closing threat" : `${f2(r.min_ttc_current_speed)} s`;
      return `Risk is now ${r.max_level} (score ${f2(r.max_score, 3)}), driven by ${r.worst_object_id || "the scene"}. ` +
             `Physical time-to-collision: ${ttc}.`;
    },
  },
  state: {
    title: "DECISION",
    tone: "warn",
    line: (ev, f, rp) => {
      const b = f.behavior || {};
      const plan = planAt(rp, f.t);
      const feas = plan
        ? ` ${plan.feasible_count} of ${plan.candidate_count} candidate paths are feasible; ` +
          `it chose ${plan.selected_id}.`
        : "";
      return `${b.reason || ""}${feas}`.trim();
    },
  },
  plan: {
    title: "RE-PLANNING",
    tone: "info",
    line: (ev, f, rp) => {
      const plan = planAt(rp, f.t);
      if (!plan) return "The planner selected a different trajectory.";
      return `New trajectory ${plan.selected_id} selected from ${plan.candidate_count} candidates ` +
             `(${plan.feasible_count} feasible). Clearance ${f2(plan.selected && plan.selected.min_clearance)} m.`;
    },
  },
  eb: {
    title: "EMERGENCY BRAKE",
    tone: "bad",
    line: (ev, f) => {
      const s = f.safety || {}, c = f.control || {};
      return `Safety supervisor overrode the planner: ${s.reason || ""} ` +
             `Braking at ${f2(c.brake)} m/s² from ${f2(f.ego.longitudinal_velocity, 1)} m/s.`;
    },
  },
  rev: {
    title: "REVERSE RECOVERY",
    tone: "special",
    line: (ev, f, rp) => {
      const plan = planAt(rp, f.t);
      return `No forward trajectory exists — every candidate was rejected. The planner generated a ` +
             `curved reverse leg${plan ? ` (${plan.selected_id})` : ""} and the vehicle is backing out.`;
    },
  },
  goal: {
    title: "SAFE PASSAGE",
    tone: "ok",
    line: (ev, f, rp) => {
      const s = rp.summary;
      return `Goal reached with ${s.collisions} collisions. Closest approach across the whole run: ` +
             `${f2(s.min_clearance_m)} m, over ${s.replans} re-planning cycles.`;
    },
  },
};

function planAt(replay, t) {
  const pf = replay.planFrameAt(t);
  return pf ? pf.plan : null;
}

/* Build the chapter list.
   Not every recorded event earns a chapter — a 40-event run would be unwatchable. We keep the
   first detection, each risk ESCALATION, each behaviour change, every brake and reverse, and the
   ending, then drop anything within `minGap` of the chapter before it. */
export function buildStory(replay) {
  const RANK = { NONE: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 };
  // Severity of a behaviour state, so a chapter can say whether the vehicle escalated or relaxed.
  const SEV = { CRUISE: 0, FOLLOW: 1, CAUTION: 2, AVOID: 3, STOPPED: 4, REVERSING: 4, EMERGENCY_BRAKE: 5 };
  // Chapter importance. When two chapters land close together the higher score survives — without
  // this, a trivial NONE->LOW risk tick could evict the CAUTION->AVOID decision 0.6 s later, which
  // is the single most important beat in the run.
  const PRIO = { start: 100, goal: 100, eb: 90, rev: 90, escalate: 70, risk: 40, track: 30, release: 20 };

  const cand = [];
  const first = replay.frames[0];
  cand.push({ t: first.t, kind: "start", prio: PRIO.start });

  let seenTrack = false, lastRisk = "NONE", lastState = null;
  replay.events.forEach((ev) => {
    if (ev.kind === "track" && !seenTrack) {
      cand.push({ t: ev.t, kind: "track", prio: PRIO.track });
      seenTrack = true;
    } else if (ev.kind === "risk") {
      const to = ev.text.split("→").pop().trim();
      // only escalations, and only to MEDIUM or above: LOW chatter is not a story beat
      if ((RANK[to] ?? 0) > (RANK[lastRisk] ?? 0) && (RANK[to] ?? 0) >= 2) {
        cand.push({ t: ev.t, kind: "risk", prio: PRIO.risk + (RANK[to] ?? 0) });
      }
      lastRisk = to;
    } else if (ev.kind === "state") {
      const to = ev.text.split("→").pop().trim();
      const up = lastState === null || (SEV[to] ?? 0) > (SEV[lastState] ?? 0);
      cand.push({ t: ev.t, kind: "state", up,
                  prio: up ? PRIO.escalate + (SEV[to] ?? 0) : PRIO.release });
      lastState = to;
    } else if (ev.kind === "eb") {
      cand.push({ t: ev.t, kind: "eb", prio: PRIO.eb });
      lastState = "EMERGENCY_BRAKE";
    } else if (ev.kind === "rev") {
      cand.push({ t: ev.t, kind: "rev", prio: PRIO.rev });
      lastState = "REVERSING";
    }
  });
  cand.push({ t: replay.frames[replay.frames.length - 1].t, kind: "goal", prio: PRIO.goal });

  // Thin by importance, not arrival order: walk in time, and when a chapter falls inside the
  // minimum gap keep whichever of the two matters more.
  const minGap = 1.5;
  const kept = [];
  cand.sort((a, b) => a.t - b.t).forEach((c) => {
    const prev = kept[kept.length - 1];
    // The opening chapter sits at t=0 and must not consume the slot of the first detection, which
    // in a short scenario arrives within a second. It never competes for spacing.
    if (prev && prev.kind === "start") { kept.push(c); return; }
    if (prev && c.t - prev.t < minGap) {
      if (c.prio > prev.prio) kept[kept.length - 1] = c;   // the newcomer wins the slot
      return;
    }
    kept.push(c);
  });

  return kept.map((c, i) => {
    const f = replay.frameAt(c.t);
    const b = BEATS[c.kind];
    return {
      index: i, t: c.t, rel: c.t - replay.t0, kind: c.kind, tone: b.tone,
      title: c.kind === "state" ? (c.up ? "DECISION" : "THREAT CLEARED") : b.title,
      text: b.line(c, f, replay),
    };
  });
}

/* ── rendering ── */
export function renderStory(chapters, onSeek) {
  const box = document.getElementById("story-list");
  box.innerHTML = chapters.map((c) => `
    <button class="ch t-${c.tone}" data-t="${c.rel}" data-i="${c.index}">
      <span class="ch-t">${c.rel.toFixed(1)}s</span>
      <span class="ch-b"><b>${c.title}</b><em>${escapeHtml(c.text)}</em></span>
    </button>`).join("");
  [...box.children].forEach((el) => {
    el.onclick = () => onSeek(parseFloat(el.dataset.t));
  });
}

export function highlightChapter(chapters, relT) {
  const box = document.getElementById("story-list");
  let idx = -1;
  for (let i = 0; i < chapters.length; i++) {
    if (chapters[i].rel > relT + 1e-6) break;
    idx = i;
  }
  if (box._last === idx) return idx;
  if (box._last >= 0 && box.children[box._last]) box.children[box._last].classList.remove("now");
  box._last = idx;
  if (idx >= 0 && box.children[idx]) {
    const el = box.children[idx];
    el.classList.add("now");
    const want = el.offsetTop - box.clientHeight / 2 + el.clientHeight / 2;
    box.scrollTop = Math.max(0, want);
  }
  return idx;
}

/** The big caption over the scene during story playback. */
export function showCaption(ch) {
  const el = document.getElementById("caption");
  if (!ch) { el.classList.remove("on"); return; }
  if (el._i === ch.index) { el.classList.add("on"); return; }
  el._i = ch.index;
  el.className = `caption on t-${ch.tone}`;
  el.querySelector(".cap-title").textContent = ch.title;
  el.querySelector(".cap-text").textContent = ch.text;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
