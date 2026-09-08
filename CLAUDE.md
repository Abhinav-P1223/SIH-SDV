# SIH26037 — project context

**Read this first.** It is the orientation document for anyone — human or AI — picking this
repository up cold.

> **Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian
> Roads.** A closed-loop autonomy stack that plans through drivable space rather than lane markings,
> plus two learned perception components validated offline on real Indian-road imagery, plus a
> Three.js console for presenting it.

---

## 1. Status — read before changing anything

| | |
|---|---|
| **Autonomy implementation** | **FROZEN** at `2ccd232`. Do not modify. |
| Evidence package | Generated at `f3f8eef` |
| UI | Active development, isolated in `ui/` |
| Tests | **348 passing** in `tests/`, plus **25** in `ui/tests/` |
| Validation | 11 scenarios × 2 perception modes = 22 runs, **22/22 completed, 0 collisions** |

**Frozen means frozen.** These paths must not change without the lead's approval:

```
autonomy/            planner, risk, prediction, tracking, fusion, control, vehicle dynamics
simulation/scenarios/   the 11 validated scenario definitions
config/              tuned parameters the validation numbers depend on
tests/               the 348-test suite
docs/FINAL_SYSTEM_*  generated evidence — regenerate, never hand-edit
```

Check you have not touched them:

```bash
git diff 2ccd232 HEAD -- autonomy/ simulation/ config/ tests/    # must print nothing
```

**Safe to change:** `ui/`, `visualization/`, `docs/` (except the generated reports), new scripts.

---

## 2. Run it in two minutes

```bash
python -m venv .venv && .venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                    # one command, everything

python -m ui.server                                # the console  -> http://127.0.0.1:8770/
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet    # one headless run
python -m pytest tests/ -q                         # the frozen suite, 15-25 min
```

**No dataset is required** for any of the above. Both trained model checkpoints ship inside the
repository.

---

## 3. What is where

```
autonomy/          FROZEN. The stack: perception → fusion → tracking → prediction → risk
                   → behaviour FSM → planner → collision check → control → vehicle dynamics
simulation/        World, sensor models, agents, scenario loader, the closed-loop runner
config/            FROZEN. autonomy.yaml, vehicle.yaml, sensors.yaml, object_profiles.yaml
scripts/           Entry points: run_scenario, final_validation, build_final_report, training,
                   evaluation, dataset selection and auditing
road_perception/   Fast-SCNN drivable-space segmentation + the planner bridge
perception_detector/  Faster R-CNN detector, UVH-26 fine-tuning, monocular geometry
dataset_adapters/  nuScenes readers (fusion validation, 2D box projection)
ui/                The Three.js console. Read-only over the frozen stack. See ui/README.md
tests/             FROZEN. 348 tests
docs/              Reports, architecture, handoff, setup
Datasets/          gitignored. Never commit dataset files
```

**Start reading here**, in this order:

1. `docs/PROJECT_OVERVIEW.md` — the whole project in 10–15 minutes
2. `autonomy/core/types.py` — every data contract in the system
3. `simulation/runner.py` — the closed loop, in one readable function
4. `autonomy/planning/planner.py` — generate → check → score → select
5. `ui/README.md` — if you are working on the console

---

## 4. Requirements

Everything installs with **one command**. `requirements.txt` declares all nine packages actually
imported:

| Package | Floor | For |
|---|---|---|
| `numpy` | 1.26 | All geometry, filtering and metrics |
| `PyYAML` | 6.0 | Config and scenario loading |
| `matplotlib` | 3.8 | Figures and the debug viewer |
| `pytest` | 8.0 | Both test suites |
| `torch` | 2.0 | Both learned models. The full suite imports it, so it is not optional |
| `torchvision` | 0.15 | Detector architecture and COCO weights |
| `scipy` | 1.11 | Connected components and distance transforms |
| `pillow` | 10.0 | Image loading |
| `requests` | 2.31 | Dataset subset download |

**Python 3.11+. CPU only, no GPU.**

**The UI adds no dependency.** Its server is Python standard library only; Three.js r160 and
OrbitControls are vendored into `ui/static/vendor/`, so the console works offline with no build
step and no CDN.

**On Linux**, install CPU-only PyTorch first or pip pulls multi-gigabyte CUDA wheels this project
never uses:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 5. The claim that must never be blurred

The project contains **two separate validations**. Do not merge them in any document, slide or
answer:

| | |
|---|---|
| **Closed-loop autonomy** | Validated end to end through **simulated** camera, LiDAR and radar models. 22/22 runs, 0 collisions |
| **Learned camera detector** | Validated **independently and offline** on held-out UVH-26 imagery. mAP 0.238 → 0.325 |

**The simulator renders no image frames**, so no detector consumed camera pixels in any closed-loop
run. This project does **not** claim end-to-end learned camera perception in closed loop.

---

## 6. Headline numbers

Every figure below comes from `docs/FINAL_SYSTEM_RESULTS.json`. Never quote a number that is not in
that file or in the code.

| Metric | Value |
|---|---|
| Scenarios / runs | 11 / 22 |
| Completion | 22 / 22 |
| Collisions | **0** |
| Worst minimum clearance | 0.51 m |
| Worst p95 re-plan latency | 55.0 ms (budget 100 ms) |
| Worst single re-plan | 189.2 ms |
| Emergency brakes | 7, all in sensors mode; 0 in ground truth |
| Re-planning cycles | 5,601 |
| Tests | 348 passed, 0 failed |
| Drivable-space IoU | 0.877 |
| Detector mAP | 0.238 → **0.325** |
| Auto-rickshaw recall | 0.000 → **0.347** |

---

## 7. Limitations — state these, do not hide them

- **Simulated sensors, not recorded ones.** Real-data validation covers the fusion interface
  (nuScenes) and the two learned models, not the closed loop.
- **The detector is not in the closed loop**, and at 287 ms per image it is an offline evidence
  pipeline, not a real-time front end.
- **UVH-26 is elevated CCTV, not dashcam.** CCTV-to-dashcam transfer is entirely unvalidated.
- **Bicycle detection collapsed** to 0.000 recall on 32 test boxes. Oversampling was tested and did
  not fix it. **Bus and truck regressed.**
- **Pedestrian and animal detection are untouched** — UVH-26 contains neither class.
- **Temporal fusion ships disabled**, because enabling it removes an over-confidence the collision
  margins were tuned against.
- **Reverse is a single committed leg**, not a multi-point turn.
- **The MATLAB and Simulink exports have never been run in MATLAB.**

---

## 8. Working conventions

- **Never invent a number.** If a value is not in the telemetry, the results JSON or the code, say
  it is unavailable. The UI renders `NOT AVAILABLE` rather than filling a gap.
- **Report defects rather than hiding them.** Several findings in this repository are negative
  results that were kept because they are true.
- **Regenerate, never hand-edit** the reports: `python scripts/build_final_report.py`.
- **Datasets stay out of git.** `Datasets/` is gitignored and must remain so.
- **Git identity:** `Abhinav-P1223`. Use only that account.
- Branch for feature work; do not commit straight to `main`.

---

## 9. Common commands

```bash
# console
python -m ui.server                       # http://127.0.0.1:8770/
python -m ui.capture --all                # re-record all replays (~40 min)
python -m ui.collect_tests                # re-record per-scenario test outcomes (~11 min)

# autonomy
python scripts/run_scenario.py NARROW_LANE_REVERSE_RECOVERY --quiet
python scripts/run_scenario.py DENSE_MARKET_MIXED_TRAFFIC --dashboard --realtime
python scripts/final_validation.py        # the 22-run evidence, ~15 min
python scripts/build_final_report.py      # regenerate the reports from that JSON

# tests
python -m pytest tests/ -q                # frozen suite, 348
python -m pytest ui/tests -q              # UI suite, 25
```

The 11 scenario ids: `UNMARKED_VILLAGE_ROAD`, `DENSE_MARKET_MIXED_TRAFFIC`,
`SUDDEN_CATTLE_CROSSING`, `SUDDEN_PEDESTRIAN_DART`, `NARROW_LANE_REVERSE_RECOVERY`,
`NARROW_LANE_BOXED_IN`, `NARROW_LANE_MUTUAL_YIELD`, `UNSIGNALIZED_INTERSECTION`,
`HIGHWAY_MERGE_SLOW_VEHICLES`, `UNPROTECTED_TURN`, `MIXED_TRAFFIC_CURVE`.

---

## 10. Documentation index

| Document | For |
|---|---|
| `docs/PROJECT_OVERVIEW.md` / `.html` | The whole project, 10–15 min. Start here |
| `ui/README.md` | The console: running it, where its data comes from, its boundaries |
| `docs/SETUP_GUIDE.md` | Get running in ten minutes, dataset download links |
| `docs/TEAM_HANDOFF.md` | Full audit: architecture, commands, frozen areas, workflow |
| `docs/CODEBASE_MAP.md` | Where everything lives |
| `docs/FINAL_SYSTEM_REPORT.md` | The evidence package |
| `docs/FINAL_SYSTEM_RESULTS.json` | Machine-readable results. The source of every number |
| `docs/TECHNICAL_ARCHITECTURE.md` | Runtime data flow |
| `docs/PHASE*.md` | Per-phase reports, each with its own limitations section |
