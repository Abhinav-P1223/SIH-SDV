"""UI smoke tests.

Deliberately OUTSIDE `tests/` so the frozen 348-test autonomy suite keeps its exact count.
Run with:  python -m pytest ui/tests -q
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.server import ConsoleServer, DEMO_ORDER, Handler, scenario_names  # noqa: E402

PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"


def get(path: str, timeout: float = 10.0):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read()


def post(path: str, payload: dict | None = None, timeout: float = 10.0):
    body = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def app():
    from http.server import ThreadingHTTPServer
    import threading
    a = ConsoleServer(PORT)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    httpd.daemon_threads = True
    httpd.app = a
    a.httpd = httpd
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    for _ in range(50):                       # wait for the socket to accept
        try:
            get("/api/state", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    yield a
    a.reset()
    httpd.shutdown()


# --------------------------------------------------------------------------- assets
def test_required_local_assets_exist():
    static = ROOT / "ui" / "static"
    for name in ("console.html", "styles.css", "src/main.js", "src/scene.js",
                 "src/replay.js", "src/panels.js", "src/reports.js",
                 "vendor/three.module.min.js", "vendor/OrbitControls.js"):
        f = static / name
        assert f.exists(), f"missing {name}"
        assert f.stat().st_size > 500


def test_page_references_only_local_assets():
    html = (ROOT / "ui" / "static" / "console.html").read_text(encoding="utf-8")
    assert 'href="/static/styles.css"' in html
    assert 'src="/static/src/main.js"' in html
    assert "http://" not in html and "https://" not in html, "no external asset may be required"


def test_scenario_names_come_from_yaml_files():
    names = scenario_names()
    on_disk = {p.stem.upper() for p in (ROOT / "simulation" / "scenarios").glob("*.yaml")}
    assert set(names) == on_disk
    assert len(names) == 11
    assert names[:len(DEMO_ORDER)] == DEMO_ORDER      # demo scenarios listed first


def test_demo_scenarios_all_exist():
    assert set(DEMO_ORDER) <= set(scenario_names())


# --------------------------------------------------------------------------- server
def test_app_starts_and_serves_the_page(app):
    code, body = get("/")
    assert code == 200 and b"<title>SIH26037" in body


def test_static_files_serve(app):
    for path, token in (("/static/styles.css", b"--acc"), ("/static/src/main.js", b"Scene3D")):
        code, body = get(path)
        assert code == 200 and token in body


def test_static_path_traversal_is_refused(app):
    code, _ = post("/api/nope")
    assert code == 404
    with pytest.raises(urllib.error.HTTPError):
        get("/static/../server.py")


def test_scenario_list_loads(app):
    code, body = get("/api/scenarios")
    d = json.loads(body)
    assert code == 200 and len(d["scenarios"]) == 11
    assert d["modes"] == ["ground_truth", "sensors"]


def _nan_to_none(o):
    if isinstance(o, float):
        return None if o != o else o          # NaN is the only float that is not equal to itself
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return o


def test_validation_results_load_and_match_the_json(app):
    """Served results must equal the file, value for value.

    Not byte-for-byte: the file carries 4 bare `NaN` literals that JSON.parse rejects, so the
    server re-serialises them as null. Nothing else may differ.
    """
    code, body = get("/api/results")
    served = json.loads(body)
    on_disk = json.loads((ROOT / "docs" / "FINAL_SYSTEM_RESULTS.json").read_text(encoding="utf-8"))
    assert code == 200
    assert served == _nan_to_none(on_disk), "results must be served as-is, never recomputed"


def test_served_results_are_strict_json(app):
    """A bare NaN would break the browser at startup — this is what caught it."""
    _, body = get("/api/results")
    assert b"NaN" not in body and b"Infinity" not in body
    json.loads(body)                                     # strict parse, no special constants
    parsed = json.loads(body, parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(f"non-strict JSON constant: {c}")))
    assert parsed["headline"]["completion"] == "22/22"


def test_perception_results_present_in_payload(app):
    _, body = get("/api/results")
    ab = json.loads(body)["detector_ab"]
    assert {"A_coco_baseline", "B_uvh26_finetuned", "C_uvh26_oversampled"} <= set(ab["arms"])
    for cls in ("MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"):
        assert cls in ab["arms"]["B_uvh26_finetuned"]["per_class"]


def test_bad_run_requests_are_rejected(app):
    code, d = post("/api/run", {"scenario": "NOT_A_SCENARIO", "perception": "sensors"})
    assert code == 400 and not d["ok"]
    code, d = post("/api/run", {"scenario": scenario_names()[0], "perception": "telepathy"})
    assert code == 400 and not d["ok"]


# --------------------------------------------------------------------------- live data
def test_live_telemetry_renders_real_frames(app):
    """Start a run and confirm the frames carry the fields the scene draws."""
    code, d = post("/api/run", {"scenario": "SUDDEN_CATTLE_CROSSING",
                                "perception": "ground_truth", "speed": 0})
    assert code == 200 and d["ok"], d

    frame = None
    for _ in range(200):
        if app.latest is not None and "_event" not in app.latest:
            frame = app.latest
            break
        time.sleep(0.1)
    assert frame is not None, "no telemetry frame arrived"

    for key in ("t", "ego", "objects", "road", "risk", "vehicle", "perception_mode"):
        assert key in frame, f"frame missing {key}"
    assert {"x", "y", "yaw", "longitudinal_velocity"} <= set(frame["ego"])
    assert {"reference", "left_boundary", "right_boundary"} <= set(frame["road"])
    if frame.get("plan"):
        assert "candidates" in frame["plan"] and "selected" in frame["plan"]
        assert "x" in frame["plan"]["selected"], "selected trajectory geometry must be present"

    for _ in range(600):                       # let it finish (speed 0 == as fast as possible)
        if app.status in ("finished", "error"):
            break
        time.sleep(0.1)
    assert app.status == "finished", app.error
    assert app.summary["collisions"] == 0
    assert app.summary["termination"] == "GOAL_REACHED"


def test_reset_clears_state(app):
    app.reset()
    st = app.state()
    assert st["status"] == "idle" and st["summary"] is None and not st["has_frame"]


def test_ui_imports_no_autonomy_mutation():
    """The UI package must not import anything that writes to the simulation."""
    src = (ROOT / "ui" / "server.py").read_text(encoding="utf-8")
    for banned in ("vehicle.step", "planner.plan", "risk_engine.evaluate", "sim.step("):
        assert banned not in src, f"UI must not call {banned}"


# --------------------------------------------------------------------------- replay
def test_replay_index_lists_recorded_runs(app):
    code, body = get("/api/replays")
    d = json.loads(body)
    assert code == 200
    assert d["replays"], "no replays recorded; run  python -m ui.capture"
    for r in d["replays"]:
        assert (ROOT / "ui" / "replay" / r["file"]).exists()
        assert r["frame_count"] > 0 and r["duration_s"] > 0
        assert r["summary"]["termination"] in ("GOAL_REACHED", "TIMEOUT", "COLLISION")


def test_replay_frames_carry_every_field_the_scene_draws(app):
    import gzip
    idx = json.loads(get("/api/replays")[1])["replays"]
    meta = idx[0]
    raw = urllib.request.urlopen(BASE + "/api/replay/" + meta["file"], timeout=30).read()
    doc = json.loads(gzip.decompress(raw))

    assert doc["scenario"] and doc["mode"] in ("sensors", "ground_truth")
    for k in ("reference", "left_boundary", "right_boundary"):
        assert k in doc["road"], f"road missing {k}"
    for k in ("length", "width", "footprint_center_offset"):
        assert k in doc["vehicle"]

    planned = [f for f in doc["frames"] if f.get("plan")]
    assert planned, "no planning cycle recorded"
    f = planned[len(planned) // 2]
    assert {"x", "y", "yaw", "longitudinal_velocity", "steering_angle"} <= set(f["ego"])
    assert {"candidates", "selected", "feasible_count", "candidate_count"} <= set(f["plan"])
    assert "x" in f["plan"]["selected"], "selected trajectory geometry must be present"
    if f["plan"]["candidates"]:
        assert {"x", "y", "feasible"} <= set(f["plan"]["candidates"][0])
    if f["objects"]:
        assert {"id", "type", "x", "y", "vx", "vy", "heading"} <= set(f["objects"][0])


def test_replay_summary_matches_the_validation_results(app):
    """A replay is a real run of the frozen stack, so its outcome must match the evidence file."""
    idx = json.loads(get("/api/replays")[1])["replays"]
    results = json.loads(get("/api/results")[1])["scenarios"]
    for r in idx:
        row = next((x for x in results
                    if x["scenario"] == r["scenario"] and x["mode"] == r["mode"]), None)
        if row is None:
            continue
        assert r["summary"]["termination"] == row["termination"], r["scenario"]
        assert r["summary"]["collisions"] == row["collisions"], r["scenario"]


def test_unknown_replay_is_refused(app):
    code, d = post("/api/nope")
    assert code == 404
    with pytest.raises(urllib.error.HTTPError):
        get("/api/replay/does_not_exist")
    with pytest.raises(urllib.error.HTTPError):
        get("/api/replay/..%2F..%2Fserver.py")


def test_frontend_modules_and_vendored_three_are_present(app):
    for path, token in (
        ("/static/src/main.js", b"Scene3D"),
        ("/static/src/scene.js", b"TubeGeometry"),
        ("/static/src/replay.js", b"plansUpTo"),
        ("/static/src/panels.js", b"NOT AVAILABLE"),
        ("/static/src/reports.js", b"renderValidation"),
        ("/static/vendor/three.module.min.js", b"THREE"),
        ("/static/vendor/OrbitControls.js", b"OrbitControls"),
    ):
        code, body = get(path)
        assert code == 200 and token in body, path


def test_no_external_assets_are_required(app):
    """The demo must work with no network. Nothing may point at a CDN."""
    static = ROOT / "ui" / "static"
    for f in list(static.glob("*.html")) + list(static.glob("*.css")) + list(static.glob("src/*.js")):
        text = f.read_text(encoding="utf-8")
        for bad in ("http://", "https://", "cdn.", "unpkg", "jsdelivr"):
            assert bad not in text, f"{f.name} references {bad}"


# --------------------------------------------------------------------------- story & tests
def test_tests_endpoint_reports_real_pytest_outcomes(app):
    code, body = get("/api/tests")
    d = json.loads(body)
    assert code == 200
    assert d.get("by_scenario"), "no test results; run  python -m ui.collect_tests"
    assert d["exit_code"] == 0, "the recorded scenario-test run did not pass"

    total = sum(len(v) for v in d["by_scenario"].values()) + len(d.get("ungrouped", []))
    assert total == d["total"]
    for scenario, rows in d["by_scenario"].items():
        assert scenario in scenario_names(), f"{scenario} is not a real scenario"
        for r in rows:
            assert r["outcome"] in ("PASSED", "FAILED", "SKIPPED")
            assert (ROOT / r["file"]).exists(), r["file"]
            assert r["title"] and not r["title"].startswith("test_")


def test_every_recorded_replay_has_mapped_tests(app):
    """A judge opening a scenario should see the tests that cover it."""
    replays = json.loads(get("/api/replays")[1])["replays"]
    tests = json.loads(get("/api/tests")[1])["by_scenario"]
    missing = [r["scenario"] for r in replays if r["scenario"] not in tests]
    assert not missing, f"no tests mapped to: {missing}"


def test_story_module_is_served_and_self_contained(app):
    code, body = get("/static/src/story.js")
    assert code == 200
    assert b"buildStory" in body and b"showCaption" in body
    text = body.decode("utf-8")
    for bad in ("http://", "https://"):
        assert bad not in text


def test_story_beats_cover_the_events_the_replays_contain(app):
    """Every event kind a replay can produce must have a caption template."""
    story = (ROOT / "ui" / "static" / "src" / "story.js").read_text(encoding="utf-8")
    replay = (ROOT / "ui" / "static" / "src" / "replay.js").read_text(encoding="utf-8")
    kinds = set(re.findall(r'push\(t,\s*"(\w+)"', replay))
    kinds |= {"start"}                      # synthesised opening chapter
    kinds -= {"track"} - kinds              # no-op, keeps the set explicit
    for k in kinds:
        assert f"  {k}: {{" in story, f"story.js has no beat for event kind {k!r}"
