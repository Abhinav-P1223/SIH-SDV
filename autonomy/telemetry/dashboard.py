"""Live dashboard telemetry sink: a stdlib HTTP server streaming compact frames.

`DashboardSink` is an ordinary `TelemetrySink`. `write()` keeps the newest
frame and, at most `rate_hz` times per wall-clock second, fans a compact JSON
view of it out to every connected Server-Sent-Events client. Endpoints:

    /          self-contained HTML/JS/canvas page (dashboard_page.PAGE_HTML)
    /stream    text/event-stream of compact frames (event `frame`, then `done`)
    /latest    compact JSON of the newest frame
    /metrics   SimulationMetrics.to_dict() of the newest frame

`close()` (called by the publisher when the run ends) pushes the final frame
and a `done` event but keeps serving so the page can still be inspected;
`shutdown()` stops the server. `RealtimePacingSink` sleeps in `write()` so a
run advances at wall-clock speed; place it last in the publisher.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from autonomy.core.config import load_vehicle_parameters
from autonomy.core.types import VehicleParameters
from autonomy.telemetry.dashboard_page import PAGE_HTML
from autonomy.telemetry.telemetry import TelemetryFrame, TelemetrySink

_QUEUE_DEPTH = 32          # frames buffered per SSE client before the oldest is dropped
_KEEPALIVE_S = 1.0         # SSE comment interval while no frame arrives


def _json_safe(v: Any) -> Any:
    """Replace non-finite floats with None so JavaScript's JSON.parse accepts the payload."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return v


def compact_frame(frame: TelemetryFrame, vehicle: VehicleParameters, candidate_stride: int = 3) -> dict[str, Any]:
    """Reduce a TelemetryFrame to the fields the dashboard page draws.

    Everything is taken from the dataclasses' own `to_dict()` methods; candidate
    trajectories are strided and stripped to their x/y polylines.
    """
    ego = frame.ego.to_dict()
    control = frame.control.to_dict()
    risk = frame.risk.to_dict()
    safety = frame.safety.to_dict()
    metrics = frame.metrics.to_dict()
    behavior = frame.decision.to_dict() if frame.decision else None
    plan = None
    if frame.plan is not None:
        p = frame.plan.to_dict(candidate_stride)
        plan = {
            "selected_id": p["selected_id"],
            "latency_ms": p["latency_ms"],
            "feasible_count": p["feasible_count"],
            "rejected_count": p["rejected_count"],
            "candidate_count": len(p["candidates"]),
            "rejection_histogram": p["rejection_histogram"],
            "selected": {
                "x": p["selected"]["trajectory"]["x"], "y": p["selected"]["trajectory"]["y"],
                "v": p["selected"]["trajectory"]["v"],
                "total_cost": p["selected"]["total_cost"],
                "min_clearance": p["selected"]["min_clearance"],
                "fallback": p["selected"]["fallback"], "degraded": p["selected"]["degraded"],
            },
            "candidates": [
                {"x": c["trajectory"]["x"], "y": c["trajectory"]["y"], "feasible": c["feasible"]}
                for c in p["candidates"] if c["id"] != p["selected_id"]
            ],
        }
    return {
        "t": frame.timestamp,
        "step": frame.step,
        "planning_cycle": frame.planning_cycle,
        "perception_mode": frame.perception_mode,
        "vehicle": {"length": vehicle.length, "width": vehicle.width,
                    "footprint_center_offset": vehicle.footprint_center_offset},
        "ego": {k: ego[k] for k in ("x", "y", "yaw", "longitudinal_velocity",
                                    "longitudinal_acceleration", "steering_angle")},
        "control": {k: control[k] for k in ("steering_angle", "acceleration", "brake", "source")},
        "objects": [{k: o[k] for k in ("id", "type", "x", "y", "vx", "vy", "heading", "length", "width", "confidence")}
                    for o in (o.to_dict() for o in frame.objects)],
        "predictions": [{"id": p["object_id"], "x": p["x"][::candidate_stride], "y": p["y"][::candidate_stride]}
                        for p in (p.to_dict() for p in frame.predictions)],
        "agents": [{k: a[k] for k in ("id", "type", "x", "y", "heading")} for a in frame.agents],
        "risk": {k: risk[k] for k in ("max_level", "max_score", "min_ttc", "min_ttc_current_speed",
                                      "min_predicted_distance", "worst_object_id", "lead_object_id")},
        "behavior": None if behavior is None else {
            k: behavior[k] for k in ("state", "previous_state", "reason", "target_speed", "time_in_state",
                                     "allow_lateral_avoidance", "force_stop")},
        "plan": plan,
        "safety": {k: safety[k] for k in ("override_active", "reason", "activation_count")},
        "metrics": {k: metrics[k] for k in ("minimum_obstacle_clearance", "collision_count",
                                            "emergency_brake_activations", "termination_reason")},
        "road": {k: frame.road[k] for k in ("reference", "left_boundary", "right_boundary")},
    }


class _Handler(BaseHTTPRequestHandler):
    """Routes GET requests; the owning DashboardSink is `self.server.sink`."""

    server_version = "SIHDashboard/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence per-request stderr lines
        return

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        sink: DashboardSink = self.server.sink  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE_HTML.encode("utf-8"))
        elif path == "/latest":
            self._send_json(sink.latest_json())
        elif path == "/metrics":
            self._send_json(sink.metrics_json())
        elif path == "/stream":
            self._stream(sink)
        elif path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _send_json(self, body: Optional[str]) -> None:
        if body is None:
            self._send(204, "application/json", b"")
        else:
            self._send(200, "application/json", body.encode("utf-8"))

    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, sink: "DashboardSink") -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        q = sink.subscribe()
        try:
            while not sink.stopping:
                try:
                    event, body = q.get(timeout=_KEEPALIVE_S)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            sink.unsubscribe(q)


class DashboardSink(TelemetrySink):
    """Serve the live dashboard; see the module docstring for the endpoints."""

    def __init__(self, port: int = 8765, host: str = "127.0.0.1", rate_hz: float = 10.0,
                 candidate_stride: int = 3, vehicle: Optional[VehicleParameters] = None):
        self.vehicle = vehicle or load_vehicle_parameters()
        self.candidate_stride = candidate_stride
        self._min_interval = 1.0 / rate_hz
        self._last_sent_wall = -math.inf
        self._lock = threading.Lock()
        self._latest: Optional[TelemetryFrame] = None
        self._clients: set[queue.Queue] = set()
        self.stopping = False
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.daemon_threads = True
        self._server.sink = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, name="dashboard-http", daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        host = self._server.server_address[0]
        return f"http://{host}:{self.port}/"

    # -- sink interface ---------------------------------------------------- #
    def write(self, frame: TelemetryFrame) -> None:
        with self._lock:
            self._latest = frame
        now = time.monotonic()
        if now - self._last_sent_wall >= self._min_interval:
            self._last_sent_wall = now
            self._broadcast("frame", self._encode(frame))

    def close(self) -> None:
        """Flush the final frame and tell clients the run is over; the server keeps serving."""
        with self._lock:
            frame = self._latest
        if frame is not None:
            self._broadcast("frame", self._encode(frame))
        self._broadcast("done", self.metrics_json() or "{}")

    def shutdown(self) -> None:
        """Stop the HTTP server and release the port."""
        self.stopping = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    # -- data for the handler ---------------------------------------------- #
    def latest_json(self) -> Optional[str]:
        with self._lock:
            frame = self._latest
        return None if frame is None else self._encode(frame)

    def metrics_json(self) -> Optional[str]:
        with self._lock:
            frame = self._latest
        return None if frame is None else json.dumps(frame.metrics.to_dict())

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        with self._lock:
            self._clients.add(q)
            frame = self._latest
        if frame is not None:                       # late joiners see the current state at once
            q.put(("frame", self._encode(frame)))
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    # -- internals --------------------------------------------------------- #
    def _encode(self, frame: TelemetryFrame) -> str:
        return json.dumps(_json_safe(compact_frame(frame, self.vehicle, self.candidate_stride)),
                          separators=(",", ":"))

    def _broadcast(self, event: str, body: str) -> None:
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            if q.full():
                try:
                    q.get_nowait()                  # drop the oldest frame for a slow client
                except queue.Empty:
                    pass
            q.put((event, body))


class RealtimePacingSink(TelemetrySink):
    """Sleep in `write()` so simulation time never runs ahead of wall-clock time.

    `speed` > 1 runs faster than real time. Register this sink last so the
    pause happens after every other sink has seen the frame.
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self._t0_wall: Optional[float] = None
        self._t0_sim = 0.0

    def write(self, frame: TelemetryFrame) -> None:
        if self._t0_wall is None:
            self._t0_wall, self._t0_sim = time.monotonic(), frame.timestamp
            return
        due = self._t0_wall + (frame.timestamp - self._t0_sim) / self.speed
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
