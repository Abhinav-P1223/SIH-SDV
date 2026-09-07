"""SIH26037 UI server — a read-only console over the frozen autonomy stack.

WHAT THIS IS
    An HTTP server that serves the operations console and streams telemetry to it. It runs
    scenarios through the EXISTING safe interface (`scripts.run_scenario.run_with_sinks`) with one
    extra `TelemetrySink` attached. Adding a sink is the documented extension point:
    "A WebSocket / REST / MATLAB adapter is just another sink" (autonomy/telemetry/telemetry.py).

WHAT IT IS NOT
    It is not part of the autonomy loop. It computes no plan, no risk and no control. It never
    writes to the simulation. Nothing under `autonomy/` or `simulation/` is imported for anything
    other than reading, and nothing there was modified.

        scenario -> Simulation (unmodified) -> TelemetryPublisher -> ConsoleStreamSink (here)
                                                                  -> SSE -> browser

ENDPOINTS
    /                     the console page
    /static/<file>        css / js
    /api/scenarios        scenario list, read from simulation/scenarios/*.yaml
    /api/state            what the server is doing right now
    /api/run   (POST)     start a scenario   {scenario, perception, speed}
    /api/pause (POST)     pause / resume the running scenario
    /api/reset (POST)     stop and clear
    /api/stream           text/event-stream of compact frames
    /api/results          docs/FINAL_SYSTEM_RESULTS.json, served verbatim

Frames are produced by `autonomy.telemetry.dashboard.compact_frame`, imported unchanged, so the UI
sees exactly the contract the existing dashboard already publishes.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autonomy.core.config import load_vehicle_parameters                      # noqa: E402
from autonomy.telemetry.dashboard import compact_frame, _json_safe            # noqa: E402
from autonomy.telemetry.telemetry import TelemetryFrame, TelemetrySink        # noqa: E402
from scripts.run_scenario import run_with_sinks                               # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
SCENARIO_DIR = ROOT / "simulation" / "scenarios"
RESULTS = ROOT / "docs" / "FINAL_SYSTEM_RESULTS.json"

QUEUE_DEPTH = 64
KEEPALIVE_S = 1.0

# The five scenarios the demo script walks through, in order.
DEMO_ORDER = [
    "UNMARKED_VILLAGE_ROAD",
    "DENSE_MARKET_MIXED_TRAFFIC",
    "SUDDEN_CATTLE_CROSSING",
    "SUDDEN_PEDESTRIAN_DART",
    "NARROW_LANE_REVERSE_RECOVERY",
]


def scenario_names() -> list[str]:
    """Scenario ids, read from the actual YAML files. Nothing is invented."""
    names = sorted(p.stem.upper() for p in SCENARIO_DIR.glob("*.yaml"))
    demo = [n for n in DEMO_ORDER if n in names]
    return demo + [n for n in names if n not in demo]


class ConsoleStreamSink(TelemetrySink):
    """Fans compact frames out to SSE clients, and paces the run to wall-clock time.

    Pacing lives here rather than in a second sink so that pause/resume can hold the run inside a
    single place. `write` is called by the publisher on the simulation thread.
    """

    def __init__(self, server: "ConsoleServer", speed: float = 1.0) -> None:
        self.server = server
        self.speed = max(speed, 0.0)
        self.params = load_vehicle_parameters()
        self._wall0: Optional[float] = None
        self._sim0: Optional[float] = None

    def write(self, frame: TelemetryFrame) -> None:
        if self.server.stop_flag.is_set():
            raise _Cancelled()
        self.server.pause_gate.wait()                      # blocks while paused
        if self.server.stop_flag.is_set():
            raise _Cancelled()

        payload = _json_safe(compact_frame(frame, self.params, candidate_stride=2))
        self.server.publish(payload)

        if self.speed > 0.0:                               # real-time pacing
            if self._wall0 is None:
                self._wall0, self._sim0 = time.perf_counter(), frame.timestamp
            else:
                target = self._wall0 + (frame.timestamp - self._sim0) / self.speed
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, 0.5))
                elif delay < -1.0:                          # fell far behind: re-base, do not spiral
                    self._wall0, self._sim0 = time.perf_counter(), frame.timestamp

    def close(self) -> None:
        pass


class _Cancelled(Exception):
    """Raised inside the sink to unwind a run that the user reset."""


class ConsoleServer:
    def __init__(self, port: int = 8770) -> None:
        self.port = port
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.latest: Optional[dict[str, Any]] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.pause_gate = threading.Event()
        self.pause_gate.set()
        self.status = "idle"          # idle | running | paused | finished | error
        self.current = {"scenario": None, "perception": None, "speed": 1.0}
        self.summary: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.httpd: Optional[ThreadingHTTPServer] = None

    # -- SSE fan-out ---------------------------------------------------- #
    def publish(self, payload: dict[str, Any]) -> None:
        # Control events are broadcast but are NOT telemetry: keeping one as `latest` would hand a
        # late-joining client an event instead of a frame, and would survive a reset.
        if "_event" not in payload:
            self.latest = payload
        blob = json.dumps(payload, separators=(",", ":"))
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(blob)
                except queue.Full:
                    try:
                        q.get_nowait()          # drop the oldest, keep the newest
                        q.put_nowait(blob)
                    except queue.Empty:
                        pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def event(self, kind: str, data: dict[str, Any] | None = None) -> None:
        self.publish({"_event": kind, **(data or {})})

    # -- run control ---------------------------------------------------- #
    def start_run(self, scenario: str, perception: str, speed: float) -> tuple[bool, str]:
        if self.status in ("running", "paused"):
            return False, "a scenario is already running; reset first"
        if scenario not in scenario_names():
            return False, f"unknown scenario {scenario!r}"
        if perception not in ("ground_truth", "sensors"):
            return False, f"unknown perception mode {perception!r}"

        self.stop_flag.clear()
        self.pause_gate.set()
        self.latest = None
        self.summary = None
        self.error = None
        self.status = "running"
        self.current = {"scenario": scenario, "perception": perception, "speed": speed}

        def worker() -> None:
            sink = ConsoleStreamSink(self, speed)
            try:
                # The existing runner. No autonomy code is touched; one sink is appended.
                result = run_with_sinks(scenario, log_dir=None, console=False,
                                        perception=perception, extra_sinks=[sink])
                m = result.metrics.to_dict()
                self.summary = {
                    "scenario": result.scenario_name,
                    "perception": perception,
                    "completed": bool(m.get("scenario_completed")),
                    "termination": m.get("termination_reason"),
                    "collisions": m.get("collision_count"),
                    "min_clearance_m": m.get("minimum_obstacle_clearance"),
                    "replans": m.get("replanning_count"),
                    "plan_changes": m.get("plan_change_count"),
                    "emergency_brakes": m.get("emergency_brake_activations"),
                    "reverse_manoeuvres": m.get("reverse_manoeuvres"),
                    "reverse_distance_m": m.get("reverse_distance_m"),
                    "latency_mean_ms": m.get("planning_latency_mean_ms"),
                    "latency_max_ms": m.get("planning_latency_max_ms"),
                    "duration_s": m.get("time_to_completion"),
                    "path_length_m": m.get("path_length"),
                    "average_speed": m.get("average_speed"),
                }
                self.status = "finished"
                self.event("done", {"summary": self.summary})
            except _Cancelled:
                self.status = "idle"
                self.event("cancelled")
            except Exception as exc:                       # surfaced in the UI, never swallowed
                self.error = f"{type(exc).__name__}: {exc}"
                self.status = "error"
                self.event("error", {"message": self.error})

        self.thread = threading.Thread(target=worker, daemon=True, name="ui-scenario")
        self.thread.start()
        return True, "started"

    def toggle_pause(self) -> str:
        if self.status == "running":
            self.pause_gate.clear()
            self.status = "paused"
        elif self.status == "paused":
            self.pause_gate.set()
            self.status = "running"
        return self.status

    def reset(self) -> None:
        self.stop_flag.set()
        self.pause_gate.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.thread = None
        self.status = "idle"
        self.latest = None
        self.summary = None
        self.error = None
        self.event("reset")

    def state(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "current": self.current,
            "summary": self.summary,
            "error": self.error,
            "clients": len(self.clients),
            "has_frame": self.latest is not None,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "SIH26037-Console/1"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> ConsoleServer:
        return self.server.app                              # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        pass                                                # quiet; errors still surface in the UI

    # -- helpers -------------------------------------------------------- #
    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routes --------------------------------------------------------- #
    def do_GET(self) -> None:                               # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._file(STATIC / "console.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            name = path[len("/static/"):]
            if "/" in name or "\\" in name or name.startswith("."):
                self._json({"error": "bad path"}, 400); return
            f = STATIC / name
            types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                     ".svg": "image/svg+xml", ".json": "application/json"}
            self._file(f, types.get(f.suffix, "application/octet-stream"))
        elif path == "/api/scenarios":
            self._json({"scenarios": scenario_names(), "demo": DEMO_ORDER,
                        "modes": ["ground_truth", "sensors"]})
        elif path == "/api/state":
            self._json(self.app.state())
        elif path == "/api/latest":
            self._json(self.app.latest or {})
        elif path == "/api/results":
            if not RESULTS.exists():
                self._json({"error": "docs/FINAL_SYSTEM_RESULTS.json not found"}, 404); return
            self._file(RESULTS, "application/json; charset=utf-8")
        elif path == "/api/stream":
            self._stream()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:                              # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            b = self._body()
            ok, msg = self.app.start_run(
                str(b.get("scenario", "")),
                str(b.get("perception", "sensors")),
                float(b.get("speed", 1.0)),
            )
            self._json({"ok": ok, "message": msg, "state": self.app.state()}, 200 if ok else 400)
        elif path == "/api/pause":
            self._json({"ok": True, "status": self.app.toggle_pause()})
        elif path == "/api/reset":
            self.app.reset()
            self._json({"ok": True, "state": self.app.state()})
        else:
            self._json({"error": "not found"}, 404)

    def _file(self, f: Path, ctype: str) -> None:
        if not f.exists():
            self._json({"error": f"missing {f.name}"}, 404); return
        self._send(200, f.read_bytes(), ctype)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = self.app.subscribe()
        try:
            if self.app.latest is not None:                 # new client sees the current frame at once
                self._sse(json.dumps(self.app.latest, separators=(",", ":")))
            while True:
                try:
                    self._sse(q.get(timeout=KEEPALIVE_S))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
            pass
        finally:
            self.app.unsubscribe(q)

    def _sse(self, blob: str) -> None:
        self.wfile.write(b"event: frame\ndata: " + blob.encode("utf-8") + b"\n\n")
        self.wfile.flush()


def serve(port: int = 8770, open_browser: bool = True) -> ConsoleServer:
    app = ConsoleServer(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    httpd.app = app                                          # type: ignore[attr-defined]
    app.httpd = httpd
    threading.Thread(target=httpd.serve_forever, daemon=True, name="ui-http").start()
    url = f"http://127.0.0.1:{port}/"
    print(f"  SIH26037 console  ->  {url}")
    print("  Ctrl+C to stop")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    return app


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SIH26037 autonomy console (read-only UI)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    app = serve(args.port, open_browser=not args.no_browser)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  stopping")
        app.reset()
        if app.httpd:
            app.httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
