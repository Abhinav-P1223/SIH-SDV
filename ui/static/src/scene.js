/* Three.js scene.
   World frame is x-forward / y-left (SI, from the simulator). Three is y-up, so the mapping is
   world (x, y) -> three (x, 0, -y). Every position drawn comes from a telemetry field. */
"use strict";

import * as THREE from "../vendor/three.module.min.js";
import { OrbitControls } from "../vendor/OrbitControls.js";
import { CLASS_COLOR, SHORT } from "./replay.js";

const W2T = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

export class Scene3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setClearColor(0x05070b, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x05070b, 55, 175);

    this.camera = new THREE.PerspectiveCamera(52, 1, 0.5, 700);
    this.camera.position.set(-14, 11, 9);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.487;
    this.controls.enabled = false;

    this.mode = "follow";
    this.camTarget = new THREE.Vector3();
    this.camPos = new THREE.Vector3(-14, 11, 9);
    this.pickables = [];
    this.onPick = null;

    this._lights();
    this._ground();

    this.roadGroup = new THREE.Group(); this.scene.add(this.roadGroup);
    this.objGroup = new THREE.Group(); this.scene.add(this.objGroup);
    this.pathGroup = new THREE.Group(); this.scene.add(this.pathGroup);
    this.ego = this._buildEgo(); this.scene.add(this.ego);

    this.trail = this._line(0xa78bfa, 3, 0.85);
    this.trail.frustumCulled = false;
    this.scene.add(this.trail);
    this.trailPts = [];

    this.ray = new THREE.Raycaster();
    canvas.addEventListener("pointerdown", (e) => this._pick(e));
    this.resize();
  }

  /* ── setup ── */
  _lights() {
    this.scene.add(new THREE.HemisphereLight(0x9fc4e8, 0x0a1018, 0.62));
    const key = new THREE.DirectionalLight(0xdcecff, 1.45);
    key.position.set(38, 54, 26);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    const d = 46;
    Object.assign(key.shadow.camera, { left: -d, right: d, top: d, bottom: -d, near: 1, far: 190 });
    key.shadow.bias = -0.0012;
    this.scene.add(key);
    this.key = key;
    const rim = new THREE.DirectionalLight(0x2ad4ee, 0.32);
    rim.position.set(-30, 16, -22);
    this.scene.add(rim);
  }

  _ground() {
    const g = new THREE.Mesh(
      new THREE.PlaneGeometry(1400, 1400),
      new THREE.MeshStandardMaterial({ color: 0x0a0f16, roughness: 0.97, metalness: 0 })
    );
    g.rotation.x = -Math.PI / 2;
    g.position.y = -0.045;
    this.scene.add(g);
    this.grid = new THREE.GridHelper(1400, 280, 0x1d2f42, 0x141f2c);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.5;
    this.grid.position.y = -0.03;
    this.scene.add(this.grid);
  }

  _line(color, width = 2, opacity = 1) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
    const mat = new THREE.LineBasicMaterial({ color, linewidth: width, transparent: true, opacity });
    return new THREE.Line(geo, mat);
  }

  _setLine(line, pts) {
    if (!pts.length) { line.visible = false; return; }
    line.visible = true;
    const attr = line.geometry.getAttribute("position");
    if (attr && attr.count === pts.length) {          // same length: write in place, no allocation
      const a = attr.array;
      for (let i = 0; i < pts.length; i++) {
        a[i * 3] = pts[i].x; a[i * 3 + 1] = pts[i].y; a[i * 3 + 2] = pts[i].z;
      }
      attr.needsUpdate = true;
      line.geometry.computeBoundingSphere();
      return;
    }
    const arr = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => { arr[i * 3] = p.x; arr[i * 3 + 1] = p.y; arr[i * 3 + 2] = p.z; });
    line.geometry.dispose();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    line.geometry = g;
  }

  /* ── ego: primitives only, no imported model ── */
  _buildEgo() {
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(4.2, 0.86, 1.8),
      new THREE.MeshStandardMaterial({ color: 0x1d5a72, roughness: 0.42, metalness: 0.55,
        emissive: 0x0d3644, emissiveIntensity: 0.55 })
    );
    body.position.y = 0.62; body.castShadow = true; g.add(body);

    const cabin = new THREE.Mesh(
      new THREE.BoxGeometry(2.05, 0.66, 1.58),
      new THREE.MeshStandardMaterial({ color: 0x0e2b38, roughness: 0.18, metalness: 0.85,
        transparent: true, opacity: 0.94 })
    );
    cabin.position.set(-0.16, 1.36, 0); cabin.castShadow = true; g.add(cabin);

    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(4.22, 0.88, 1.82)),
      new THREE.LineBasicMaterial({ color: 0x2ad4ee, transparent: true, opacity: 0.92 })
    );
    edge.position.y = 0.62; g.add(edge);
    this.egoEdge = edge;

    const wheelGeo = new THREE.CylinderGeometry(0.33, 0.33, 0.24, 18);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x10161e, roughness: 0.85 });
    this.wheels = [];
    [[1.3, 0.86], [1.3, -0.86], [-1.3, 0.86], [-1.3, -0.86]].forEach(([x, z]) => {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.x = Math.PI / 2;
      w.position.set(x, 0.33, z);
      w.castShadow = true;
      g.add(w); this.wheels.push(w);
    });

    const head = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.15, 0.42),
      new THREE.MeshBasicMaterial({ color: 0xdff4ff }));
    head.position.set(2.12, 0.68, 0.58); g.add(head);
    const head2 = head.clone(); head2.position.z = -0.58; g.add(head2);

    const tailMat = new THREE.MeshBasicMaterial({ color: 0x5c1512 });
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.14, 0.34), tailMat);
    tail.position.set(-2.12, 0.68, 0.6); g.add(tail);
    const tail2 = tail.clone(); tail2.position.z = -0.6; g.add(tail2);
    this.tailMat = tailMat;

    return g;
  }

  /* ── road ── */
  setRoad(road) {
    this._lastPlanFrame = null;                        // new replay: force the fan to rebuild
    while (this.roadGroup.children.length) {
      const c = this.roadGroup.children.pop();
      c.geometry && c.geometry.dispose();
      this.roadGroup.remove(c);
    }
    if (!road || !road.left_boundary) return;
    const L = road.left_boundary, R = road.right_boundary;
    const n = Math.max(L.length, R.length);
    const at = (arr, i) => arr[Math.min(i, arr.length - 1)];

    // surface: two triangles per station between the boundaries
    const pos = [], idx = [];
    for (let i = 0; i < n; i++) {
      const l = at(L, i), r = at(R, i);
      pos.push(l[0], 0, -l[1], r[0], 0, -r[1]);
    }
    for (let i = 0; i < n - 1; i++) {
      const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
      idx.push(a, c, b, b, c, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    const surf = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0x1e2836, roughness: 0.9, metalness: 0.04, side: THREE.DoubleSide }));
    surf.receiveShadow = true;
    this.roadGroup.add(surf);

    // boundaries as low kerbs so the corridor reads in 3D
    [L, R].forEach((B) => {
      const pts = B.map((p) => W2T(p[0], p[1], 0.09));
      const kerb = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x4d7ea8 }));
      this.roadGroup.add(kerb);
      const glow = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
        B.map((p) => W2T(p[0], p[1], 0.015))),
        new THREE.LineBasicMaterial({ color: 0x2ad4ee, transparent: true, opacity: 0.28 }));
      this.roadGroup.add(glow);
    });

    // direction reference — dashed, and NOT a lane line
    if (road.reference) {
      const pts = road.reference.map((p) => W2T(p[0], p[1], 0.02));
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineDashedMaterial({ color: 0x51708f, dashSize: 1.6, gapSize: 2.2,
          transparent: true, opacity: 0.55 }));
      l.computeLineDistances();
      this.roadGroup.add(l);
    }
  }

  /* ── per-frame update ── */
  update(f, planFrame) {
    const e = f.ego;
    this.ego.position.copy(W2T(e.x, e.y, 0));
    this.ego.rotation.y = -e.yaw;

    const braking = !!(f.safety && f.safety.override_active);
    this.egoEdge.material.color.setHex(braking ? 0xf04438 : 0x2ad4ee);
    this.tailMat.color.setHex((f.control && f.control.brake > 0.2) || braking ? 0xff3b30 : 0x5c1512);
    const steer = e.steering_angle || 0;
    this.wheels[0].rotation.y = steer; this.wheels[1].rotation.y = steer;

    this._objects(f);
    this._paths(f, planFrame);
    this.lastEgo = e;
    this._camera(e);
  }

  setTrail(pts) {
    this.trailPts = pts;
    this._setLine(this.trail, pts.map((p) => W2T(p[0], p[1], 0.05)));
  }

  _objects(f) {
    const objs = f.objects || [];
    const risk = f.risk || {};
    // pool by index so we are not allocating meshes every frame
    while (this.objGroup.children.length < objs.length) this.objGroup.add(this._objShell());
    this.objGroup.children.forEach((o, i) => {
      const on = i < objs.length;
      o.visible = on;
      // label and velocity line hang off the scene root, not the shell, so they must be hidden too
      if (o.userData.label) o.userData.label.visible = on;
      if (o.userData.vel) o.userData.vel.visible = on;
    });
    this.pickables = [];

    objs.forEach((o, i) => {
      const shell = this.objGroup.children[i];
      const col = CLASS_COLOR[o.type] ?? CLASS_COLOR.UNKNOWN;
      const L = Math.max(o.length || 1, 0.4), W = Math.max(o.width || 1, 0.4);
      const H = o.type === "PEDESTRIAN" ? 1.7 : o.type === "BUS" || o.type === "TRUCK" ? 3.0
        : o.type === "CATTLE" ? 1.4 : o.type === "MOTORCYCLE" || o.type === "BICYCLE" ? 1.5 : 1.5;

      shell.scale.set(L, H, W);
      shell.position.copy(W2T(o.x, o.y, H / 2));
      shell.rotation.y = -(o.heading || 0);
      const worst = o.id === risk.worst_object_id;
      shell.children[0].material.color.setHex(col);
      shell.children[0].material.opacity = worst ? 0.5 : 0.3;
      shell.children[1].material.color.setHex(col);
      shell.children[1].material.opacity = worst ? 1 : 0.7;
      shell.userData.obj = o;
      shell.userData.worst = worst;
      this.pickables.push(shell);

      // velocity vector — one second of travel, from vx/vy
      const spd = Math.hypot(o.vx || 0, o.vy || 0);
      const v = shell.userData.vel;
      if (spd > 0.25) {
        v.visible = true;
        this._setLine(v, [W2T(o.x, o.y, 0.3), W2T(o.x + o.vx, o.y + o.vy, 0.3)]);
        v.material.color.setHex(col);
      } else v.visible = false;

      // label
      const lab = shell.userData.label;
      const txt = `${SHORT[o.type] || o.type} ${o.id}   ${spd.toFixed(1)} m/s`;
      if (lab.userData.txt !== txt) { drawLabel(lab, txt, col, worst); lab.userData.txt = txt; }
      lab.position.copy(W2T(o.x, o.y, H + 0.95));
      // beyond ~55 m a label is unreadable clutter; the box and its colour still show the object
      lab.visible = lab.position.distanceTo(this.camera.position) < 55;
    });

    // predictions
    const preds = f.predictions || [];
    while (this.pathGroup.children.filter((c) => c.userData.pred).length < preds.length) {
      const l = this._line(0xf5a524, 2, 0.95); l.userData.pred = true;
      l.frustumCulled = false; this.pathGroup.add(l);
    }
    const predLines = this.pathGroup.children.filter((c) => c.userData.pred);
    predLines.forEach((l, i) => {
      if (i >= preds.length) { l.visible = false; return; }
      const p = preds[i];
      const pts = p.x.map((x, k) => W2T(x, p.y[k], 0.12));
      this._setLine(l, pts);
    });
  }

  _objShell() {
    const g = new THREE.Group();
    const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.3,
        roughness: 0.6, metalness: 0.15 }));
    m.castShadow = true;
    g.add(m);
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 })));
    const vel = this._line(0xffffff, 2, 0.9); vel.frustumCulled = false;
    this.scene.add(vel); g.userData.vel = vel;
    const label = makeLabel(); this.scene.add(label); g.userData.label = label;
    return g;
  }

  _paths(f, planFrame) {
    const plan = planFrame && planFrame.plan;
    if (planFrame === this._lastPlanFrame) return;      // unchanged since the last planning cycle
    this._lastPlanFrame = planFrame;
    const keep = this.pathGroup.children.filter((c) => c.userData.pred);
    this.pathGroup.children.filter((c) => c.userData.cand).forEach((c) => (c.visible = false));

    if (!plan) return;
    const cands = plan.candidates || [];
    let pool = this.pathGroup.children.filter((c) => c.userData.cand);
    while (pool.length < cands.length) {
      const l = this._line(0x4c6a8c, 1, 0.6); l.userData.cand = true;
      l.frustumCulled = false; this.pathGroup.add(l); pool.push(l);
    }
    pool = this.pathGroup.children.filter((c) => c.userData.cand);
    cands.forEach((c, i) => {
      const l = pool[i];
      if (!c.x || c.x.length < 2) { l.visible = false; return; }
      this._setLine(l, c.x.map((x, k) => W2T(x, c.y[k], 0.08)));
      l.material.color.setHex(c.feasible ? 0x6f9cc9 : 0xa04a4a);
      l.material.opacity = c.feasible ? 0.62 : 0.4;
    });

    // selected — visually dominant: a thick tube, not a line
    const sel = plan.selected;
    if (this.selTube) { this.scene.remove(this.selTube); this.selTube.geometry.dispose(); this.selTube = null; }
    if (sel && sel.x && sel.x.length > 1) {
      const pts = sel.x.map((x, k) => W2T(x, sel.y[k], 0.16));
      const curve = new THREE.CatmullRomCurve3(pts);
      const bad = sel.fallback || sel.degraded;
      const geo = new THREE.TubeGeometry(curve, Math.min(pts.length * 3, 120), 0.17, 7, false);
      const mat = new THREE.MeshBasicMaterial({
        color: bad ? 0xf04438 : 0x2ad4ee, transparent: true, opacity: 0.92 });
      this.selTube = new THREE.Mesh(geo, mat);
      this.scene.add(this.selTube);
    }
  }

  /* ── camera ── */
  setMode(m) {
    this.mode = m;
    this.controls.enabled = (m === "orbit");
    this.snapNext = true;                       // reframe immediately rather than drifting over
    if (m === "orbit") { this.controls.target.copy(this.camTarget); this.controls.update(); }
  }

  _camera(e) {
    // aim a little ahead of the vehicle: the interesting half of the scene is the road it is
    // driving into, not the road behind it
    const p = W2T(e.x + Math.cos(e.yaw) * 4.5, e.y + Math.sin(e.yaw) * 4.5, 0.8);
    // A scrub can move the ego tens of metres in one tick. Smoothing across that leaves the camera
    // stranded (the lerp only advances on a telemetry tick, and a paused seek delivers exactly one),
    // so snap on a big jump and smooth only during continuous playback.
    const jump = this.snapNext || this.camTarget.distanceTo(p) > 12;
    this.snapNext = false;
    if (jump) this.camTarget.copy(p); else this.camTarget.lerp(p, 0.16);
    if (this.mode === "orbit") { this.controls.target.copy(this.camTarget); return; }

    const yaw = e.yaw;
    let want;
    if (this.mode === "follow") {
      want = new THREE.Vector3(e.x - Math.cos(yaw) * 8.4 - Math.sin(yaw) * 1.8, 4.3,
        -(e.y - Math.sin(yaw) * 8.4 + Math.cos(yaw) * 1.8));
    } else if (this.mode === "chase") {
      want = new THREE.Vector3(e.x - Math.cos(yaw) * 5.6, 2.5, -(e.y - Math.sin(yaw) * 5.6));
    } else { // top
      want = new THREE.Vector3(e.x + 3, 34, -e.y);
    }
    if (jump) this.camPos.copy(want); else this.camPos.lerp(want, this.mode === "top" ? 0.1 : 0.13);
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camTarget);
    this.key.position.set(e.x + 34, 52, -e.y + 24);
    this.key.target.position.copy(p);
    this.key.target.updateMatrixWorld();
  }

  _pick(ev) {
    if (!this.onPick) return;
    const r = this.canvas.getBoundingClientRect();
    const m = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
      -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(m, this.camera);
    const hits = this.ray.intersectObjects(this.pickables, true);
    if (hits.length) {
      let o = hits[0].object;
      while (o && !o.userData.obj) o = o.parent;
      if (o) this.onPick(o.userData.obj);
    } else this.onPick(null);
  }

  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(r.width, r.height, false);
    this.camera.aspect = Math.max(r.width / Math.max(r.height, 1), 0.1);
    this.camera.updateProjectionMatrix();
  }

  render() {
    // The camera must keep converging even when playback is paused — a paused seek or a camera
    // mode change delivers no telemetry tick, so driving it from update() alone strands it.
    if (this.lastEgo) this._camera(this.lastEgo);
    if (this.mode === "orbit") this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/* ── canvas-texture labels ── */
function makeLabel() {
  const cv = document.createElement("canvas");
  cv.width = 320; cv.height = 64;
  const tex = new THREE.CanvasTexture(cv);
  tex.anisotropy = 4;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true,
    depthTest: false, sizeAttenuation: false }));
  spr.scale.set(0.15, 0.03, 1);   // screen-space: constant size at any range
  spr.renderOrder = 10;
  spr.userData.cv = cv; spr.userData.tex = tex;
  return spr;
}

function drawLabel(spr, text, color, strong) {
  const cv = spr.userData.cv, c = cv.getContext("2d");
  c.clearRect(0, 0, cv.width, cv.height);
  const hex = "#" + color.toString(16).padStart(6, "0");
  c.font = "600 26px ui-monospace,Menlo,Consolas,monospace";
  const w = c.measureText(text).width + 26;
  c.fillStyle = strong ? "rgba(10,14,20,.94)" : "rgba(10,14,20,.72)";
  roundRect(c, (cv.width - w) / 2, 12, w, 40, 8);
  c.fill();
  c.strokeStyle = hex; c.lineWidth = strong ? 2.5 : 1.4; c.stroke();
  c.fillStyle = strong ? "#ffffff" : hex;
  c.textAlign = "center"; c.textBaseline = "middle";
  c.fillText(text, cv.width / 2, 33);
  spr.userData.tex.needsUpdate = true;
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y); c.lineTo(x + w - r, y); c.quadraticCurveTo(x + w, y, x + w, y + r);
  c.lineTo(x + w, y + h - r); c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  c.lineTo(x + r, y + h); c.quadraticCurveTo(x, y + h, x, y + h - r);
  c.lineTo(x, y + r); c.quadraticCurveTo(x, y, x + r, y); c.closePath();
}
