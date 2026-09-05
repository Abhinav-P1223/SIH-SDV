"""Run a scenario closed-loop and print the measured metrics.

    python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING
    python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet --no-logs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.runner import run_scenario  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="SUDDEN_CATTLE_CROSSING")
    ap.add_argument("--quiet", action="store_true", help="suppress per-cycle console output")
    ap.add_argument("--no-logs", action="store_true", help="do not write logs/*.jsonl and *.csv")
    args = ap.parse_args()

    result = run_scenario(args.scenario, log_dir=None if args.no_logs else "logs",
                          console=not args.quiet, keep_frames=False)
    m = result.metrics
    print("\n=== METRICS:", result.scenario_name, "===")
    print(json.dumps(m.to_dict(), indent=2))
    ok = m.scenario_completed and m.collision_count == 0
    print("\nRESULT:", "PASS" if ok else "FAIL", f"({m.termination_reason})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
