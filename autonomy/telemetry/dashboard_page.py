"""The dashboard web page served at `/` by `DashboardSink`.

Plain HTML + CSS + JavaScript + canvas, no external libraries. It subscribes to
`/stream` (Server-Sent Events) and draws the compact frames produced by
`dashboard.compact_frame`. Kept as a Python string so the package needs no
build step and no static-file lookup.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SIH26037 live dashboard</title>
<style>
  :root { --bg:#0f1318; --panel:#171d25; --line:#2a3340; --fg:#d8dee8; --dim:#8a96a8; --acc:#4fc3f7; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui, Segoe UI, sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:8px 14px; border-bottom:1px solid var(--line); }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  #status { color:var(--dim); }
  #status.live { color:#7ee787; } #status.done { color:#f0b429; } #status.lost { color:#ff7b72; }
  main { display:grid; grid-template-columns: 1fr 360px; grid-template-rows: 1fr 150px; gap:8px; padding:8px;
         height: calc(100vh - 42px - 16px); }
  #view { grid-row: 1 / 2; background:#0b0e12; border:1px solid var(--line); border-radius:6px; width:100%; height:100%; }
  aside { grid-row: 1 / 3; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px; overflow:auto; }
  #strips { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
  #strips canvas { width:100%; height:100%; background:#0b0e12; border:1px solid var(--line); border-radius:6px; }
  h2 { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); margin:14px 0 6px; }
  h2:first-child { margin-top:0; }
  .badge { display:inline-block; padding:3px 10px; border-radius:4px; font-weight:700; font-size:14px; color:#0b0e12; background:#7ee787; }
  .row { display:flex; justify-content:space-between; gap:8px; padding:2px 0; }
  .row span:last-child { font-variant-numeric: tabular-nums; color:#fff; }
  #reason { color:var(--dim); font-size:12px; margin-top:6px; min-height:2.8em; }
  .bar { display:flex; align-items:center; gap:6px; font-size:11px; }
  .bar i { display:block; height:8px; background:#ff7b72; border-radius:2px; min-width:2px; }
  .flag { padding:4px 8px; border-radius:4px; background:#232b36; color:var(--dim); font-weight:600; }
  .flag.on { background:#ff7b72; color:#0b0e12; }
  .legend { color:var(--dim); font-size:11px; margin-top:10px; }
  .legend b { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:4px; }
</style>
</head>
<body>
<header>
  <h1>SIH26037 &mdash; closed-loop autonomy, live telemetry</h1>
  <span id="status">connecting&hellip;</span>
  <span id="clock" style="margin-left:auto;color:var(--dim)"></span>
</header>
<main>
  <canvas id="view"></canvas>
  <aside>
    <h2>Behaviour</h2>
    <span id="state" class="badge">&mdash;</span>
    <div id="reason"></div>
    <div class="row"><span>previous</span><span id="prev">&mdash;</span></div>
    <div class="row"><span>time in state</span><span id="tis">&mdash;</span></div>
    <div class="row"><span>policy target speed</span><span id="tspeed">&mdash;</span></div>

    <h2>Ego</h2>
    <div class="row"><span>speed</span><span id="speed">&mdash;</span></div>
    <div class="row"><span>steering (actual)</span><span id="steer">&mdash;</span></div>
    <div class="row"><span>accel / brake cmd</span><span id="cmd">&mdash;</span></div>
    <div class="row"><span>command source</span><span id="src">&mdash;</span></div>

    <h2>Risk</h2>
    <div class="row"><span>level / score</span><span id="risk">&mdash;</span></div>
    <div class="row"><span>TTC route / physical</span><span id="ttc">&mdash;</span></div>
    <div class="row"><span>min predicted distance</span><span id="mpd">&mdash;</span></div>
    <div class="row"><span>worst / lead object</span><span id="worst">&mdash;</span></div>

    <h2>Planner</h2>
    <div class="row"><span>feasible / rejected / total</span><span id="feas">&mdash;</span></div>
    <div class="row"><span>selected</span><span id="sel">&mdash;</span></div>
    <div class="row"><span>cost / min clearance</span><span id="cost">&mdash;</span></div>
    <div class="row"><span>latency</span><span id="lat">&mdash;</span></div>
    <div id="hist"></div>

    <h2>Safety supervisor</h2>
    <div id="override" class="flag">override inactive</div>
    <div class="row"><span>activations</span><span id="acts">&mdash;</span></div>
    <div id="sreason" style="color:var(--dim);font-size:12px"></div>

    <h2>Metrics (running)</h2>
    <div class="row"><span>min obstacle clearance</span><span id="mclr">&mdash;</span></div>
    <div class="row"><span>collisions</span><span id="mcol">&mdash;</span></div>
    <div class="row"><span>emergency brakes</span><span id="meb">&mdash;</span></div>
    <div class="row"><span>perception</span><span id="mode">&mdash;</span></div>

    <div class="legend">
      <div><b style="background:#4fc3f7"></b>ego footprint &nbsp; <b style="background:#7ee787"></b>selected trajectory</div>
      <div><b style="background:#6b7a8f"></b>feasible candidate &nbsp; <b style="background:#8a3b3b"></b>rejected candidate</div>
      <div><b style="background:#f0b429"></b>predicted path &nbsp; <b style="background:#556"></b>road boundary / reference</div>
    </div>
  </aside>
  <div id="strips">
    <canvas id="strip_speed"></canvas>
    <canvas id="strip_steer"></canvas>
  </div>
</main>
<script>
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const view = $('view'), stripSpeed = $('strip_speed'), stripSteer = $('strip_steer');
  const STATE_COLORS = { CRUISE:'#7ee787', FOLLOW:'#79c0ff', CAUTION:'#f0b429', AVOID:'#ffa657',
                         EMERGENCY_BRAKE:'#ff7b72', STOPPED:'#d2a8ff', REVERSING:'#ff9bce' };
  const TYPE_COLORS = { PEDESTRIAN:'#ff7b72', BICYCLE:'#ffa657', MOTORCYCLE:'#ffa657', AUTO_RICKSHAW:'#f0b429',
                        CAR:'#79c0ff', BUS:'#79c0ff', TRUCK:'#79c0ff', PUSHCART:'#d2a8ff', CATTLE:'#f2cc60', UNKNOWN:'#8a96a8' };
  const HISTORY = 600;                       // samples kept for the time-series strips (60 s at 10 Hz)
  const hist = { t: [], speed: [], steer: [] };
  let frame = null, dirty = false;

  const fmt = (v, d = 2, unit = '') => (v === null || v === undefined) ? 'inf' : v.toFixed(d) + unit;
  const deg = r => r * 180 / Math.PI;

  // ---- text panel -------------------------------------------------------
  function updatePanel(f) {
    $('clock').textContent = 't = ' + f.t.toFixed(2) + ' s   step ' + f.step;
    const b = f.behavior;
    if (b) {
      $('state').textContent = b.state;
      $('state').style.background = STATE_COLORS[b.state] || '#8a96a8';
      $('reason').textContent = b.reason;
      $('prev').textContent = b.previous_state;
      $('tis').textContent = fmt(b.time_in_state, 1, ' s');
      $('tspeed').textContent = fmt(b.target_speed, 1, ' m/s') + (b.force_stop ? ' (stop)' : '') +
                                (b.allow_lateral_avoidance ? '' : ' (no lateral)');
    }
    const e = f.ego, c = f.control;
    $('speed').textContent = fmt(e.longitudinal_velocity, 2, ' m/s') + ' (' + fmt(e.longitudinal_velocity * 3.6, 0, ' km/h') + ')';
    $('steer').textContent = fmt(deg(e.steering_angle), 1, ' deg');
    $('cmd').textContent = fmt(c.acceleration, 2) + ' / ' + fmt(c.brake, 2, ' m/s²');
    $('src').textContent = c.source;
    const r = f.risk;
    $('risk').textContent = r.max_level + ' / ' + fmt(r.max_score, 2);
    $('ttc').textContent = fmt(r.min_ttc, 2, ' s') + ' / ' + fmt(r.min_ttc_current_speed, 2, ' s');
    $('mpd').textContent = fmt(r.min_predicted_distance, 2, ' m');
    $('worst').textContent = (r.worst_object_id || '—') + ' / ' + (r.lead_object_id || '—');
    const p = f.plan;
    if (p) {
      $('feas').textContent = p.feasible_count + ' / ' + p.rejected_count + ' / ' + p.candidate_count;
      $('sel').textContent = p.selected_id + (p.selected.fallback ? ' FALLBACK' : '') + (p.selected.degraded ? ' degraded' : '');
      $('cost').textContent = fmt(p.selected.total_cost, 3) + ' / ' + fmt(p.selected.min_clearance, 2, ' m');
      $('lat').textContent = fmt(p.latency_ms, 1, ' ms');
      const total = Math.max(1, p.candidate_count);
      $('hist').innerHTML = Object.entries(p.rejection_histogram)
        .filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1])
        .map(([k, n]) => '<div class="bar"><i style="width:' + (100 * n / total).toFixed(0) + 'px"></i>' + n + ' ' + k + '</div>')
        .join('') || '<div class="bar" style="color:var(--dim)">no rejections</div>';
    }
    const s = f.safety;
    $('override').textContent = s.override_active ? 'SAFETY OVERRIDE ACTIVE' : 'override inactive';
    $('override').className = 'flag' + (s.override_active ? ' on' : '');
    $('acts').textContent = s.activation_count;
    $('sreason').textContent = s.override_active ? s.reason : '';
    const m = f.metrics;
    $('mclr').textContent = fmt(m.minimum_obstacle_clearance, 2, ' m');
    $('mcol').textContent = m.collision_count;
    $('meb').textContent = m.emergency_brake_activations;
    $('mode').textContent = f.perception_mode;
  }

  // ---- top-down canvas ----------------------------------------------------
  function fitCanvas(cv) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
      cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    }
    return { w: cv.width, h: cv.height, dpr };
  }

  function drawView(f) {
    const { w, h } = fitCanvas(view);
    const g = view.getContext('2d');
    g.clearRect(0, 0, w, h);
    const e = f.ego;
    const window_m = 90;                                   // metres of world shown across the canvas width
    const scale = w / window_m;
    // ego kept at 35% from the left along its heading, so most of the view is ahead of it
    const cx = w * 0.35, cy = h * 0.5, cos = Math.cos(-e.yaw), sin = Math.sin(-e.yaw);
    const X = (x, y) => { const dx = x - e.x, dy = y - e.y; return [cx + scale * (dx * cos - dy * sin), cy - scale * (dx * sin + dy * cos)]; };
    const poly = (pts, style, width, dash) => {
      if (!pts || pts.length < 2) return;
      g.beginPath(); g.setLineDash(dash || []); g.strokeStyle = style; g.lineWidth = width;
      pts.forEach((p, i) => { const q = X(p[0], p[1]); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
      g.stroke(); g.setLineDash([]);
    };
    const xy = (xs, ys) => xs.map((x, i) => [x, ys[i]]);
    const box = (x, y, yaw, L, W, fill, stroke) => {
      const c = Math.cos(yaw), s = Math.sin(yaw), hl = L / 2, hw = W / 2;
      const corners = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([a, b]) => X(x + a * c - b * s, y + a * s + b * c));
      g.beginPath(); corners.forEach((q, i) => i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])); g.closePath();
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.5; g.stroke(); }
    };

    // road
    poly(f.road.left_boundary, '#556', 2); poly(f.road.right_boundary, '#556', 2);
    poly(f.road.reference, '#3a4250', 1, [6, 8]);

    // candidates (faded), then the selected trajectory
    if (f.plan) {
      for (const c of f.plan.candidates) poly(xy(c.x, c.y), c.feasible ? 'rgba(107,122,143,0.35)' : 'rgba(138,59,59,0.35)', 1);
      poly(xy(f.plan.selected.x, f.plan.selected.y), '#7ee787', 2.5);
    }

    // predicted paths
    for (const p of f.predictions) poly(xy(p.x, p.y), '#f0b429', 1.2, [3, 4]);

    // ground-truth agents (sensors mode only: they are what the tracker should have found)
    if (f.perception_mode !== 'ground_truth') {
      g.strokeStyle = 'rgba(200,200,200,0.35)';
      for (const a of f.agents) { const q = X(a.x, a.y); g.beginPath(); g.moveTo(q[0] - 5, q[1]); g.lineTo(q[0] + 5, q[1]); g.moveTo(q[0], q[1] - 5); g.lineTo(q[0], q[1] + 5); g.stroke(); }
    }

    // tracked objects with type labels and velocity vectors
    g.font = (11 * (window.devicePixelRatio || 1)) + 'px system-ui, sans-serif';
    for (const o of f.objects) {
      const col = TYPE_COLORS[o.type] || TYPE_COLORS.UNKNOWN;
      box(o.x, o.y, o.heading, o.length, o.width, col + '99', col);
      const q0 = X(o.x, o.y), q1 = X(o.x + o.vx, o.y + o.vy);
      g.beginPath(); g.moveTo(q0[0], q0[1]); g.lineTo(q1[0], q1[1]); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
      g.fillStyle = '#fff';
      g.fillText(o.id + ' ' + o.type + (o.confidence < 0.999 ? ' ' + (100 * o.confidence).toFixed(0) + '%' : ''), q0[0] + 8, q0[1] - 8);
    }

    // ego footprint (centre offset from the CG along the heading) and heading tick
    const v = f.vehicle, off = v.footprint_center_offset;
    box(e.x + off * Math.cos(e.yaw), e.y + off * Math.sin(e.yaw), e.yaw, v.length, v.width,
        f.safety.override_active ? 'rgba(255,123,114,0.9)' : 'rgba(79,195,247,0.9)', '#fff');
    const h0 = X(e.x, e.y), h1 = X(e.x + 3 * Math.cos(e.yaw), e.y + 3 * Math.sin(e.yaw));
    g.beginPath(); g.moveTo(h0[0], h0[1]); g.lineTo(h1[0], h1[1]); g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();

    // scale bar (10 m)
    g.strokeStyle = '#8a96a8'; g.lineWidth = 2; g.beginPath(); g.moveTo(16, h - 16); g.lineTo(16 + 10 * scale, h - 16); g.stroke();
    g.fillStyle = '#8a96a8'; g.fillText('10 m', 16, h - 22);
  }

  // ---- rolling time-series strips ----------------------------------------
  function pushHistory(f) {
    hist.t.push(f.t); hist.speed.push(f.ego.longitudinal_velocity); hist.steer.push(deg(f.ego.steering_angle));
    for (const k of Object.keys(hist)) if (hist[k].length > HISTORY) hist[k].shift();
  }

  function drawStrip(cv, ys, label, unit, color, symmetric) {
    const { w, h, dpr } = fitCanvas(cv);
    const g = cv.getContext('2d');
    g.clearRect(0, 0, w, h);
    if (ys.length < 2) return;
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-3); lo = -m; hi = m; }
    if (hi - lo < 1e-6) { hi = lo + 1; }
    const pad = 22 * dpr, span = hi - lo;
    const Y = y => h - pad / 2 - (h - pad) * (y - lo) / span;
    if (symmetric) { g.strokeStyle = '#2a3340'; g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(w, Y(0)); g.stroke(); }
    g.beginPath(); g.strokeStyle = color; g.lineWidth = 1.5 * dpr;
    ys.forEach((y, i) => { const x = w * i / (HISTORY - 1); i ? g.lineTo(x, Y(y)) : g.moveTo(x, Y(y)); });
    g.stroke();
    g.fillStyle = '#8a96a8'; g.font = (11 * dpr) + 'px system-ui, sans-serif';
    g.fillText(label + '  ' + ys[ys.length - 1].toFixed(2) + ' ' + unit + '   [' + lo.toFixed(1) + ', ' + hi.toFixed(1) + ']', 8 * dpr, 14 * dpr);
  }

  function render() {
    dirty = false;
    if (!frame) return;
    updatePanel(frame);
    drawView(frame);
    drawStrip(stripSpeed, hist.speed, 'speed', 'm/s', '#4fc3f7', false);
    drawStrip(stripSteer, hist.steer, 'steering', 'deg', '#ffa657', true);
  }
  function schedule() { if (!dirty) { dirty = true; requestAnimationFrame(render); } }
  window.addEventListener('resize', schedule);

  // ---- data feed ----------------------------------------------------------
  const status = $('status');
  const es = new EventSource('/stream');
  es.addEventListener('frame', ev => {
    frame = JSON.parse(ev.data);
    pushHistory(frame);
    status.textContent = 'live'; status.className = 'live';
    schedule();
  });
  es.addEventListener('done', () => {
    status.textContent = 'run finished — showing final frame'; status.className = 'done';
  });
  es.onerror = () => {
    if (status.className !== 'done') { status.textContent = 'stream lost — retrying'; status.className = 'lost'; }
  };
})();
</script>
</body>
</html>
"""
