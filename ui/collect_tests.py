"""Run the real scenario tests and record, per scenario, what each test proves.

    pytest -v  ->  parse node ids and outcomes  ->  ui/replay/tests.json

Nothing is asserted here that pytest did not report. The docstring shown in the UI is read from
the test function itself, so the description and the outcome cannot drift apart.

Usage:
    python -m ui.collect_tests            # scenario suites (about 6 min)
    python -m ui.collect_tests --all      # every test in tests/ (15-25 min)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "ui" / "replay" / "tests.json"

# Which scenario each test FILE exercises. Files that cover several scenarios are parametrised,
# and the scenario is then read from the `[PARAM]` in the node id instead.
FILE_SCENARIO = {
    "tests/scenarios/test_sudden_cattle_crossing.py": "SUDDEN_CATTLE_CROSSING",
    "tests/scenarios/test_sudden_pedestrian_dart.py": "SUDDEN_PEDESTRIAN_DART",
    "tests/scenarios/test_narrow_lane_boxed_in.py": "NARROW_LANE_BOXED_IN",
    "tests/scenarios/test_mixed_traffic_curve.py": "MIXED_TRAFFIC_CURVE",
    "tests/scenarios/test_parameter_sweep.py": "SUDDEN_CATTLE_CROSSING",
    "tests/unit/test_reverse_recovery.py": "NARROW_LANE_REVERSE_RECOVERY",
    "tests/unit/test_reverse.py": "NARROW_LANE_BOXED_IN",
}

TARGETS = [
    "tests/scenarios",
    "tests/unit/test_reverse_recovery.py",
    "tests/unit/test_reverse.py",
    "tests/unit/test_interactive_traffic.py",   # covers UNPROTECTED_TURN + NARROW_LANE_MUTUAL_YIELD
]

NODE = re.compile(r"^(?P<file>tests[/\\][^:]+)::(?P<name>[^\s]+)\s+(?P<outcome>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)")
SCENARIO_PARAM = re.compile(r"\[([A-Z][A-Z0-9_]{4,})\]")


ALL_SCENARIOS = {p.stem.upper() for p in (ROOT / "simulation" / "scenarios").glob("*.yaml")}


def inspect(paths: list[Path]) -> tuple[dict, dict]:
    """Per test function: the first docstring line, and the scenario it names (if exactly one).

    The scenario is read from the function's own source. A file such as
    `test_interactive_traffic.py` exercises two different scenarios, so mapping by filename alone
    silently drops one of them — which is exactly what the UI test caught.
    """
    docs: dict[tuple[str, str], str] = {}
    scen: dict[tuple[str, str], str] = {}
    for p in paths:
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, OSError):
            continue
        rel = p.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")):
                continue
            docs[(rel, node.name)] = (ast.get_docstring(node) or "").strip().split("\n")[0]
            body = ast.get_source_segment(src, node) or ""
            named = {n for n in ALL_SCENARIOS if n in body}
            if len(named) == 1:
                scen[(rel, node.name)] = next(iter(named))
    return docs, scen


def humanise(name: str) -> str:
    """`test_it_actually_reversed_to_get_out` -> `It actually reversed to get out`."""
    base = name.split("[")[0]
    s = base[5:] if base.startswith("test_") else base
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else base


def run(targets: list[str]) -> dict:
    """Run pytest and read the results from JUnit XML.

    Not from stdout: the project sets `addopts = "-q"` in pyproject.toml, so `-v` never produces
    per-test lines. JUnit XML is structured, unambiguous, and carries the outcome and duration of
    every case regardless of verbosity settings.
    """
    import tempfile
    import xml.etree.ElementTree as ET

    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "report.xml"
        cmd = [sys.executable, "-m", "pytest", *targets,
               f"--junitxml={xml}", "--tb=no", "-p", "no:cacheprovider"]
        print("  " + " ".join(cmd[2:]))
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if not xml.exists():
            print(proc.stdout[-2000:])
            raise SystemExit("pytest produced no JUnit report")
        root = ET.parse(xml).getroot()

    files: set[Path] = set()
    rows = []
    for case in root.iter("testcase"):
        classname = case.get("classname", "")          # e.g. tests.scenarios.test_x
        name = case.get("name", "")
        rel = classname.replace(".", "/") + ".py"
        if not rel.startswith("tests/"):
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "FAILED"
        elif case.find("skipped") is not None:
            outcome = "SKIPPED"
        else:
            outcome = "PASSED"
        files.add(ROOT / rel)
        param = SCENARIO_PARAM.search(name)
        rows.append({
            "file": rel,
            "name": name,
            "scenario": param.group(1) if param else FILE_SCENARIO.get(rel),
            "outcome": outcome,
            "seconds": round(float(case.get("time", 0.0)), 2),
        })

    docs, scen = inspect(sorted(files))
    for r in rows:
        base = r["name"].split("[")[0]
        r["title"] = humanise(r["name"])
        r["doc"] = docs.get((r["file"], base), "")
        # precedence: the [PARAM] in the node id, then the scenario named in the function body,
        # then the file-level mapping
        if not r["scenario"]:
            r["scenario"] = scen.get((r["file"], base)) or FILE_SCENARIO.get(r["file"])

    by_scenario: dict[str, list] = {}
    for r in rows:
        if r["scenario"]:
            by_scenario.setdefault(r["scenario"], []).append(r)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    return {
        "generated_wall_s": round(wall, 1),
        "command": " ".join(cmd[2:]),
        "exit_code": proc.returncode,
        "total": len(rows),
        "counts": counts,
        "by_scenario": by_scenario,
        "ungrouped": [r for r in rows if not r["scenario"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Record per-scenario test outcomes for the UI")
    ap.add_argument("--all", action="store_true", help="run the whole tests/ tree")
    args = ap.parse_args()
    targets = ["tests"] if args.all else TARGETS

    print(f"running {'all' if args.all else 'scenario'} tests ...")
    data = run(targets)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")

    print(f"  {data['total']} tests in {data['generated_wall_s']}s  ->  {data['counts']}")
    for sc, rows in sorted(data["by_scenario"].items()):
        ok = sum(1 for r in rows if r["outcome"] == "PASSED")
        print(f"    {sc:32s} {ok}/{len(rows)} passed")
    if data["ungrouped"]:
        print(f"    (unmapped to a scenario: {len(data['ungrouped'])})")
    print(f"  wrote {OUT}")
    return 0 if data["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
