# Team handoff

Everything a new teammate needs. Verified against the repository at commit `f3f8eef`; the autonomy
implementation is frozen at `2ccd232`. Where a claim could not be checked on this machine it says
so rather than guessing.

For a ten-minute start use [SETUP_GUIDE.md](SETUP_GUIDE.md). For the directory map use
[CODEBASE_MAP.md](CODEBASE_MAP.md).

---

## 1. What this project is

An adaptive path-planning and collision-avoidance stack for unstructured Indian roads, SIH26037. A
closed-loop simulator drives a vehicle through eleven scenarios using simulated camera, LiDAR and
radar. The planner does not need lane markings: it plans inside a drivable corridor.

Alongside it, two learned perception components were built and validated **offline**: a drivable
space segmenter and a camera object detector fine-tuned on Indian traffic imagery.

**Measured result: 22 runs, 22 of 22 completions, 0 collisions, 348 tests passing.**

---

## 2. The distinction you must not blur

> The learned detector was validated independently on held-out UVH-26 Indian-scene imagery, and its
> output contract was verified against the unmodified tracking pipeline. The autonomy stack was
> validated independently, end to end, through simulated multimodal sensors. These are two separate
> validations. **The simulator renders no image frames**, so no detector consumed camera pixels
> during the closed-loop runs.

Never say the detector drove the 22 closed-loop runs, and never say end-to-end learned camera
perception was validated in closed loop. `docs/DEMO_SCRIPT.md` carries a "do not say" list.

---

## 3. Runtime architecture, traced from the code

The single entry point is `simulation/runner.py`. `Simulation.step()` is one 50 Hz tick and calls
the whole pipeline in this order. Verified by reading `runner.py`, not copied from design notes.

| Stage | Module | Class or function | Input | Output |
|---|---|---|---|---|
| World update | `simulation/world/world.py` | `World.update` | dt, ego pose | agents advanced |
| Sensing | `simulation/sensors/models.py` | `SensorSuite.sense` | agents, ego, t | `list[Detection]` |
| Fusion and tracking | `autonomy/perception/tracker.py` | `SensorFusionTracker.ingest` | detections, t, ego | tracks |
| Object states | same | `get_object_states` | t | `list[ObjectState]` |
| Prediction | `autonomy/prediction/predictor.py` | `ConstantVelocityPredictor.predict` | objects, t | `list[ObjectPrediction]` |
| Risk | `autonomy/risk/risk_engine.py` | `RiskEngine.evaluate` | ego, plan, objects, predictions, road | `RiskSummary` |
| Behaviour | `autonomy/behavior/state_machine.py` | `BehaviorStateMachine.decide` | risk, ego, feasibility | `BehaviorDecision` |
| Planning | `autonomy/planning/planner.py` | `Planner.plan` | ego, decision, predictions | `PlannerOutput` |
| Collision check | `autonomy/planning/collision_checker.py` | `CollisionChecker.check_all` | candidates, predictions | feasibility, reasons |
| Control | `autonomy/control/tracker.py` | `TrajectoryTracker.track` | ego, trajectory | `ControlCommand` |
| Safety | `autonomy/safety/supervisor.py` | `SafetySupervisor.check` | command, risk, plan | possibly overridden command |
| Vehicle | `autonomy/vehicle/models.py` | `KinematicBicycleModel.step` | state, command, dt | new `VehicleState` |

**The ego moves only at that last line.** Nothing else writes the ego pose.

Configuration comes from `config/autonomy.yaml`, `config/vehicle.yaml`, `config/sensors.yaml` and
`config/object_profiles.yaml`. `config/sensors.yaml: perception.mode` selects `sensors` or
`ground_truth`.

### Three interfaces that make parts swappable

- **`Detection`** is the perception boundary. Simulated sensors emit it; so does the learned camera
  detector. The tracker cannot tell them apart.
- **`RoadModel` / `DrivableSpace`** is the map boundary. Scenario YAML produces one; so does the
  learned segmentation.
- **`ObjectStateProvider`** is the fusion boundary. Ground-truth mode supplies the world provider,
  sensors mode supplies the fusion tracker.

---

## 4. Environment

| | |
|---|---|
| Python | **3.11 or newer**, required by `pyproject.toml`. Developed on 3.11 |
| OS | Windows, Linux, macOS. Developed on Windows 11 |
| GPU | **None needed.** All models are CPU-only |
| Environment variables | **None required** |
| External tools | None. MATLAB export files are generated but were never run in MATLAB |
| CI | **None.** There is no `.github/workflows` |

### Dependencies

`pip install -r requirements.txt` installs everything. The full list, with what each is for:

| Package | Floor | Used for |
|---|---|---|
| `numpy` | 1.26 | Everything. All geometry, filtering and metrics |
| `PyYAML` | 6.0 | Config and scenario loading |
| `matplotlib` | 3.8 | Figures, the debug viewer |
| `pytest` | 8.0 | The test suite |
| `torch` | 2.0 | Both learned models. **The full test suite imports it**, so it is not optional |
| `torchvision` | 0.15 | The detector architecture and its COCO weights |
| `scipy` | 1.11 | Connected components and distance transforms in `road_perception` |
| `pillow` | 10.0 | Image loading across the perception packages |
| `requests` | 2.31 | Dataset subset download in `scripts/select_uvh26_subset.py` |

Versions verified on the development machine: numpy 2.4.2, PyYAML 6.0.3, matplotlib 3.10.8,
pytest 9.0.2, torch 2.11.0, torchvision 0.26.0, scipy 1.17.1, pillow 12.1.0, requests 2.32.5. Only
floors are declared, not exact pins.

**Linux:** install CPU-only PyTorch first, or pip will pull multi-gigabyte CUDA wheels that this
CPU-only project never uses:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`pip install -e .` now exposes all six packages, but it is not required; every documented command
runs from the repository root.

---

## 5. Clean-machine setup

```bash
git clone https://github.com/Abhinav-P1223/SIH-SDV.git
cd SIH-SDV
git checkout main

python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

# Linux only, first, to avoid a CUDA download you do not need:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -c "import numpy, yaml, matplotlib, torch, torchvision, scipy, PIL, requests; print('ok')"

python -m pytest tests/unit -q -m "not slow"        # about 3 minutes
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet
python -m pytest tests/ -q                          # 15 to 25 minutes
python scripts/final_validation.py                  # about 15 minutes
```

No datasets are needed for any of that.

---

## 6. Local-only files

Everything below is **gitignored on purpose**. Nothing here may ever be committed.

| Path | Contents | Required? | Size | Source | Redistributable |
|---|---|---|---|---|---|
| `Datasets/v1.0-mini/` | nuScenes v1.0-mini | Only for fusion and detector evaluation | 5.1 GB | Register at nuscenes.org | No |
| `Datasets/idd-lite/` | IDD-Lite segmentation | Only for segmentation work | 42 MB | Register at idd.insaan.iiit.ac.in | No |
| `Datasets/uvh26_subset/` | 750-image UVH-26 subset | Only for detector fine-tuning | 2.5 GB | `python scripts/select_uvh26_subset.py --download` | CC BY 4.0, but keep it out of git |
| `perception_detector/checkpoints/*oversampled*.pt` | Phase 7E experiment | No | 4 x 76 MB | Regenerate, or ask the lead | Yes, as a release artefact |
| `~/.cache/torch/hub/checkpoints/` | COCO pretrained weights | Auto-downloaded on first use | 88 MB | torchvision downloads it | Yes |

**Tracked in git:** code, config, docs, scripts, the reproducibility manifests
(`docs/uvh26_manifest.json`), the machine-readable results, and two model checkpoints.

### The UVH-26 subset is reproducible

`docs/uvh26_manifest.json` names all 750 images with seed 20260907. Running the download script
reproduces the identical selection; a hash of the ordered file list confirms it.

---

## 7. Model checkpoints

| Checkpoint | Phase | Purpose | Path | Size | Tracked | Needed at runtime |
|---|---|---|---|---|---|---|
| **`fasterrcnn_uvh26.pt`** | **7C** | **PRIMARY FINAL DETECTOR** | `perception_detector/checkpoints/` | 76 MB | **yes, in git** | No, offline only |
| `fastscnn_iddlite.pt` | 3B | Drivable-space segmentation | `road_perception/checkpoints/` | 3.0 MB | **yes, in git** | No, offline only |
| `fasterrcnn_uvh26_oversampled.pt` | 7E | **Experiment. Do not use.** | `perception_detector/checkpoints/` | 76 MB | no, gitignored | No |
| `fasterrcnn_uvh26_oversampled_ep{1,2,3}.pt` | 7E | Per-epoch, diagnostic only | same | 76 MB each | no | No |

**The primary final model is the Phase 7C `fasterrcnn_uvh26.pt`.** It was chosen over the Phase 7E
oversampled variant on mAP 0.325 against 0.282 and on motorcycle and auto-rickshaw recall. The
Phase 7E model is an experimental diagnostic and **must not be presented as the deliverable**.

Both tracked checkpoints arrive with a plain `git clone`, so no manual model download is needed.

**Policy for future checkpoints:** `perception_detector/checkpoints/*.pt` is gitignored. Ship new
large models as GitHub Release artefacts, never as git blobs. Do not rewrite history to remove the
one already there.

---

## 8. Command reference

Run every command from the repository root.

| Command | Does | Runtime | Needs local data |
|---|---|---|---|
| `pip install -r requirements.txt` | Install everything | 2 to 10 min | no |
| `python -m pytest tests/ -q` | Full suite, expect 348 passed | 15 to 25 min | datasets for 0 skips |
| `python -m pytest tests/unit -q -m "not slow"` | 292 unit tests | about 3 min | no |
| `python scripts/run_scenario.py <NAME> --quiet` | One scenario, writes `logs/` | about 20 s | no |
| `python scripts/run_scenario.py <NAME> --dashboard 8765` | Live dashboard on port 8765 | until Ctrl+C | no |
| `python scripts/run_scenario.py <NAME> --realtime 1.0` | Paced at wall-clock speed | real time | no |
| `python scripts/generate_results.py` | Regenerates `docs/STAGE1_RESULTS.md` | several min | no |
| `python scripts/final_validation.py` | 22 runs, writes `docs/phase8_final_validation.json` | about 15 min | no |
| `python scripts/audit_stress.py --perception sensors` | Stress battery | several min | no |
| `python scripts/audit_adaptivity.py --perception sensors` | Adaptivity variants | several min | no |
| `python scripts/sweep.py --scenario X --param Y --values ...` | Parameter sweep | varies | no |
| `python scripts/build_final_report.py` | Regenerates the four final documents and figures | seconds | no |
| `python scripts/export_matlab.py` | MATLAB/Simulink export, never run in MATLAB | seconds | no |
| `python scripts/eval_uvh26_ab.py` | Detector A/B/C on the locked test split | about 4 min | **UVH-26 subset** |
| `python scripts/eval_segmentation.py --root Datasets/idd-lite/idd20k_lite` | Segmentation metrics and figures | about 6 min | **IDD-Lite** |
| `python scripts/validate_nuscenes_fusion.py` | Fusion interface against real nuScenes | several min | **nuScenes** |
| `python scripts/select_uvh26_subset.py --download` | Fetches the 750-image subset, 2.5 GB | 20 to 40 min | network |

Scenario names: `UNMARKED_VILLAGE_ROAD`, `UNSIGNALIZED_INTERSECTION`,
`HIGHWAY_MERGE_SLOW_VEHICLES`, `DENSE_MARKET_MIXED_TRAFFIC`, `SUDDEN_CATTLE_CROSSING`,
`NARROW_LANE_REVERSE_RECOVERY`, `NARROW_LANE_BOXED_IN`, `UNPROTECTED_TURN`,
`NARROW_LANE_MUTUAL_YIELD`, `SUDDEN_PEDESTRIAN_DART`, `MIXED_TRAFFIC_CURVE`.

**UNVERIFIED — requires teammate environment:** the dashboard and real-time flags were not
exercised during this audit, and the MATLAB export has never been opened in MATLAB.

---

## 9. Test suite

**348 tests: 292 unit, 55 scenario, 1 integration.**

```bash
python -m pytest tests/ -q
```

reproduces `348 passed, 0 failed, 0 skipped` **when the datasets and checkpoints are present**.
Without them, about 35 tests skip cleanly. That is correct behaviour, not breakage.

| Group | Tests | Notes |
|---|---|---|
| `tests/unit/` | 292 | Fast except the perception files |
| `tests/scenarios/` | 55 | Each runs a full closed-loop scenario; slow |
| `tests/integration/` | 1 | End-to-end tracking |

Data-gated: `test_nuscenes_adapter.py` skips as a whole module (13 tests) without nuScenes;
`test_road_perception.py` (7), `test_uvh26_finetune.py` (8), `test_camera_detector.py` (5) and
`test_perception_planner_bridge.py` (2) gate individual tests.

`tests/scenarios/test_parameter_sweep.py` is marked `slow`; deselect with `-m "not slow"`.

**Never weaken a test to make it pass.** Latency assertions already carry a documented 2x/3x
allowance because wall-clock timing inside a shared pytest process is not the same measurement as a
standalone run.

---

## 10. Reproducing the final validation

```bash
python scripts/final_validation.py
```

Runs 11 scenarios in both perception modes, 22 runs, about 15 minutes, no datasets needed. Writes
`docs/phase8_final_validation.json`. Then:

```bash
python scripts/build_final_report.py
```

regenerates `FINAL_SYSTEM_REPORT.md`, `FINAL_SYSTEM_RESULTS.json`, `TECHNICAL_ARCHITECTURE.md`,
`DEMO_SCRIPT.md` and three figures from that JSON, so the documents cannot drift apart.

**Validated result: 11 scenarios, 22 runs, 22 of 22 completions, 0 collisions.** Worst minimum
clearance 0.51 m, worst p95 replanning latency 55.0 ms against a 100 ms budget, 0 emergency brakes
under ground truth and 7 under simulated sensors.

Re-read section 2 before writing anything about what this validates.

---

## 11. DO NOT MODIFY WITHOUT TEAM LEAD APPROVAL

| Area | Why |
|---|---|
| `autonomy/planning/` | Two tuning invariants will silently break the vehicle. `weights.consistency` must stay clearly below `weights.lateral` or the ego parks off its route; the jerk-limited speed profile must be seeded with the current acceleration or the car crawls |
| `autonomy/risk/`, `autonomy/perception/`, `autonomy/prediction/` | The collision margin scales with each object's *predicted* uncertainty. Changing it re-opens a collision found by the stress battery |
| `autonomy/control/`, `autonomy/vehicle/` | Reverse control is non-minimum-phase and deliberately gain-limited and clamped |
| `autonomy/behavior/state_machine.py` | Dual hysteresis was tuned to stop state chatter |
| `autonomy/safety/supervisor.py` | Authoritative override. Weakening it invalidates every safety claim |
| `simulation/scenarios/*.yaml` | The validation set. Changing a scenario invalidates the measured results |
| `tests/` | 348 passing tests are the evidence |
| `docs/uvh26_manifest.json` | The locked 750-image selection, seed 20260907 |
| `docs/FINAL_SYSTEM_*.{md,json}`, `docs/PHASE*.md` | Measured results. Regenerate, never hand-edit |
| `config/*.yaml` | A test asserts every dataclass default matches its YAML value |

---

## 12. Safe areas to work in

| Area | Path | Notes |
|---|---|---|
| Debug viewer | `visualization/debug_view.py` | matplotlib run viewer. Extend freely |
| Dashboard front end | `autonomy/telemetry/dashboard_page.py` | The served HTML and JS |
| New visualisations | new files under `visualization/` | Read `logs/*.jsonl` and `docs/*.json` |
| Presentation assets | `docs/img/` | Add figures |
| Documentation | `docs/*.md` except the generated ones | |
| Demo wrapper | a new script under `scripts/` | Compose existing entry points |
| Result exploration | notebooks or scripts reading `docs/*.json` | All results are machine-readable |

### The UI boundary, which is not negotiable

```
UI  ->  reads telemetry, results and configuration
UI  is NOT  part of the planning or control loop
```

The dashboard is a telemetry **sink**: it receives frames after the fact and cannot influence them.
Any change that lets a UI write into the autonomy path needs explicit approval, because it would put
presentation code inside a safety-critical loop.

---

## 13. Git workflow

`main` is the frozen, stable integration branch.

```bash
git fetch origin && git status -sb        # is local behind remote?
git log --oneline main..origin/main       # exactly what you are missing

git checkout main && git pull origin main
git checkout -b feature/your-thing
# work, then
git add <files> && git commit -m "..."
git push -u origin feature/your-thing
```

Then open a pull request against `main`. Anything touching a section 11 area needs review by the
team lead. Do not commit directly to `main`, and **never rewrite history**: the Phase 7C checkpoint
is already in it and rewriting would break every existing clone.

Before opening a PR, run `python -m pytest tests/ -q` and say in the description whether it was 348
passed or how many skipped and why.

---

## 14. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: autonomy` | Wrong working directory | Run from the repository root |
| `ModuleNotFoundError: torch` / `scipy` / `PIL` | Install skipped or partial | `pip install -r requirements.txt` |
| Tests skip instead of passing | Datasets or checkpoints absent | Expected. See section 6 |
| pip downloads gigabytes on Linux | Default index serves CUDA wheels | Use `--index-url https://download.pytorch.org/whl/cpu` |
| First detector run stalls | torchvision is downloading COCO weights, 88 MB | Wait once; it caches |
| `SyntaxError` on modern syntax | Python older than 3.11 | Upgrade |
| Latency assertion fails | Machine busy or thermally throttled | Re-run alone. The same code measured 22.7, 47.8 and 128 ms in one session |
| Path errors on Linux | Windows-developed paths | All code uses `pathlib`; report any literal backslash you find |
| `git push` warns about a large file | The 76 MB Phase 7C checkpoint | Expected. Do not add more; use Releases |

---

## 15. Genuinely outstanding before a clean teammate can work

1. **No CI.** Nothing runs the tests automatically on push.
2. **No dependency pinning.** Only floors are declared, so a future release could break the build.
3. **UNVERIFIED:** the dashboard, the real-time pacing flag and the MATLAB export were not exercised
   during this audit.

Two gaps this audit found have since been **fixed**: `requirements.txt` now declares all nine
packages, and `pyproject.toml` now includes all six source packages. A clean
`pip install -r requirements.txt` is all a teammate needs.
