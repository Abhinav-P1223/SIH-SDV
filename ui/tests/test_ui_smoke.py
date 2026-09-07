"""UI smoke tests.

Deliberately OUTSIDE `tests/` so the frozen 348-test autonomy suite keeps its exact count.
Run with:  python -m pytest ui/tests -q
"""
from __future__ import annotations

import json
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
    for name in ("console.html", "console.css", "console.js"):
        assert (static / name).exists(), f"missing {name}"
        assert (static / name).stat().st_size > 500


def test_page_references_only_local_assets():
    html = (ROOT / "ui" / "static" / "console.html").read_text(encoding="utf-8")
    assert 'href="/static/console.css"' in html
    assert 'src="/static/console.js"' in html
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
    for path, token in (("/static/console.css", b"--accent"), ("/static/console.js", b"drawSelected")):
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


def test_validation_results_load_and_match_the_json(app):
    code, body = get("/api/results")
    served = json.loads(body)
    on_disk = json.loads((ROOT / "docs" / "FINAL_SYSTEM_RESULTS.json").read_text(encoding="utf-8"))
    assert code == 200 and served == on_disk, "results must be served verbatim, never recomputed"


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
