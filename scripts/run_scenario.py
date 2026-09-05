"""Run a scenario closed-loop and print the measured metrics.

    python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING
    python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet --no-logs
    python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --dashboard --realtime   # live view at http://127.0.0.1:8765/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.core.config import AutonomyConfig, ObjectProfiles, PerceptionConfig, load_vehicle_parameters  # noqa: E402
from autonomy.telemetry.dashboard import DashboardSink, RealtimePacingSink  # noqa: E402
from autonomy.telemetry.telemetry import (ConsoleSink, CsvSummarySink, JsonLinesSink, TelemetryPublisher,  # noqa: E402
                                          TelemetrySink)
from simulation.runner import RunResult, Simulation, run_scenario  # noqa: E402
from simulation.scenarios.loader import load_scenario  # noqa: E402


def run_with_sinks(name: str, log_dir: str | None, console: bool, perception: str | None,
                   extra_sinks: list[TelemetrySink]) -> RunResult:
    """`simulation.runner.run_scenario` with additional sinks (dashboard, real-time pacing) appended."""
    cfg = AutonomyConfig.load()
    profiles = ObjectProfiles.load()
    perc = PerceptionConfig.load()
    if perception:
        perc = perc.with_mode(perception)
    scenario = load_scenario(name, profiles)
    publisher = TelemetryPublisher()
    if log_dir is not None:
        publisher.add(JsonLinesSink(Path(log_dir) / f"{scenario.name.lower()}.jsonl"))
        publisher.add(CsvSummarySink(Path(log_dir) / f"{scenario.name.lower()}_summary.csv"))
    if console:
        publisher.add(ConsoleSink())
    for sink in extra_sinks:                       # pacing sink last so every sink sees the frame first
        publisher.add(sink)
    sim = Simulation(scenario, cfg, load_vehicle_parameters(), profiles, publisher, perception=perc)
    return RunResult(sim.run(), [], scenario.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="SUDDEN_CATTLE_CROSSING")
    ap.add_argument("--quiet", action="store_true", help="suppress per-cycle console output")
    ap.add_argument("--no-logs", action="store_true", help="do not write logs/*.jsonl and *.csv")
    ap.add_argument("--perception", choices=["ground_truth", "sensors"], default=None,
                    help="override perception.mode from config/sensors.yaml")
    ap.add_argument("--dashboard", nargs="?", const=8765, type=int, metavar="PORT",
                    help="serve the live dashboard on http://127.0.0.1:PORT/ (default 8765); stays up until Ctrl+C")
    ap.add_argument("--realtime", nargs="?", const=1.0, type=float, metavar="SPEED",
                    help="pace the run at wall-clock speed (SPEED x real time, default 1.0)")
    args = ap.parse_args()

    log_dir = None if args.no_logs else "logs"
    dashboard = DashboardSink(port=args.dashboard) if args.dashboard is not None else None
    if dashboard is None and args.realtime is None:
        result = run_scenario(args.scenario, log_dir=log_dir, console=not args.quiet, keep_frames=False,
                              perception=args.perception)
    else:
        extra: list[TelemetrySink] = []
        if dashboard is not None:
            extra.append(dashboard)
            print(f"Dashboard: {dashboard.url}  (open it now; the run starts immediately)", flush=True)
        if args.realtime is not None:
            extra.append(RealtimePacingSink(speed=args.realtime))
        result = run_with_sinks(args.scenario, log_dir, not args.quiet, args.perception, extra)
    m = result.metrics
    print("\n=== METRICS:", result.scenario_name, "===")
    print(json.dumps(m.to_dict(), indent=2))
    ok = m.scenario_completed and m.collision_count == 0
    print("\nRESULT:", "PASS" if ok else "FAIL", f"({m.termination_reason})")
    if dashboard is not None:
        print(f"\nDashboard still serving the final frame at {dashboard.url} - press Ctrl+C to exit.", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            dashboard.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
