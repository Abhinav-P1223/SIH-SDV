"""DashboardSink serves real frames over HTTP: page, /latest, /metrics, /stream."""
import json
import time
import urllib.error
import urllib.request

import pytest

from autonomy.telemetry.dashboard import DashboardSink, RealtimePacingSink, compact_frame
from autonomy.core.config import load_vehicle_parameters
from simulation.runner import run_scenario


@pytest.fixture(scope="module")
def frames():
    result = run_scenario("SUDDEN_CATTLE_CROSSING", log_dir=None, console=False, keep_frames=True,
                          perception="ground_truth")
    return result.frames


@pytest.fixture
def sink():
    s = DashboardSink(port=0)
    yield s
    s.shutdown()


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_endpoints_serve_real_frames(sink, frames):
    assert sink.port > 0
    status, _ = _get(sink.url + "latest")
    assert status == 204                                  # nothing published yet

    for f in frames[::20]:
        sink.write(f)
    sink.close()                                          # end of run: flush, keep serving

    status, body = _get(sink.url + "latest")
    latest = json.loads(body)
    assert status == 200
    assert {"t", "ego", "control", "objects", "predictions", "risk", "behavior", "plan", "safety",
            "metrics", "road", "vehicle"} <= set(latest)
    assert {"x", "y", "yaw", "longitudinal_velocity", "steering_angle"} <= set(latest["ego"])
    assert {"max_level", "max_score", "min_ttc"} <= set(latest["risk"])
    assert {"state", "reason", "target_speed"} <= set(latest["behavior"])
    assert {"feasible_count", "rejected_count", "rejection_histogram", "selected", "candidates"} <= set(latest["plan"])
    assert len(latest["plan"]["selected"]["x"]) == len(latest["plan"]["selected"]["y"]) > 1
    assert {"left_boundary", "right_boundary", "reference"} <= set(latest["road"])

    _, body = _get(sink.url + "metrics")
    metrics = json.loads(body)
    assert {"minimum_obstacle_clearance", "collision_count", "scenario_completed", "termination_reason"} <= set(metrics)
    assert metrics["collision_count"] == 0

    _, page = _get(sink.url)
    assert b"<canvas" in page and b"/stream" in page

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(sink.url + "nope")
    assert exc.value.code == 404


def test_stream_delivers_latest_frame_to_late_client(sink, frames):
    sink.write(frames[-1])
    req = urllib.request.urlopen(sink.url + "stream", timeout=5)
    first = req.readline().decode()                        # "event: frame"
    payload = req.readline().decode()                      # "data: {...}"
    req.close()
    assert first.strip() == "event: frame"
    frame = json.loads(payload[len("data: "):])
    assert frame["step"] == frames[-1].step


def test_compact_frame_is_json_clean_and_strided(frames):
    f = next(fr for fr in frames if fr.planning_cycle and fr.plan is not None)
    d = compact_frame(f, load_vehicle_parameters(), candidate_stride=4)
    json.dumps(d)                                          # no numpy types
    full = len(f.plan.selected.trajectory)
    assert len(d["plan"]["selected"]["x"]) == len(range(0, full, 4))
    assert d["plan"]["candidate_count"] == len(f.plan.candidates)
    assert len(d["plan"]["candidates"]) == len(f.plan.candidates) - 1    # selected drawn separately


def test_realtime_pacing_sleeps_for_sim_time(frames):
    pacer = RealtimePacingSink(speed=20.0)
    t0 = time.monotonic()
    for f in frames[:101]:                                 # 2.0 s of simulation at 20x -> ~0.1 s wall
        pacer.write(f)
    assert 0.08 <= time.monotonic() - t0 < 1.0
