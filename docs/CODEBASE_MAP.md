# Codebase map

Compact directory and module map. Verified against the repository at commit `f3f8eef`, not copied
from design notes.

**Modify column:** FROZEN needs team-lead approval, OPEN is safe teammate work, CARE means it can be
extended but its existing behaviour is depended on by tests.

## Top level

| Path | Purpose | Modify |
|---|---|---|
| `autonomy/` | The autonomy stack. Everything from perception interfaces to control. | **FROZEN** |
| `simulation/` | World, agents, sensor models, scenario YAML, the run loop. | **FROZEN** |
| `road_perception/` | Phase 3B/3C drivable-space segmentation and the planner bridge. | CARE |
| `perception_detector/` | Phase 6/7 camera object detection and the UVH-26 dataset code. | CARE |
| `dataset_adapters/` | nuScenes readers, including 3D-to-2D box projection. | CARE |
| `visualization/` | `debug_view.py`, a matplotlib run viewer. | **OPEN** |
| `scripts/` | 21 command-line entry points. All runnable, all verified. | CARE |
| `tests/` | 348 tests. | **FROZEN** (do not weaken) |
| `config/` | Four YAML files that drive every tunable. | **FROZEN** |
| `docs/` | Reports, machine-readable results, figures. | **OPEN** |
| `matlab/`, `matlab_export/` | Generated MATLAB/Simulink export. Never run in MATLAB. | CARE |
| `logs/` | Per-scenario JSONL and CSV telemetry. Gitignored except the keep-file. | generated |
| `Datasets/` | **Gitignored.** Local-only datasets. See `docs/SETUP_GUIDE.md`. | local only |

## `autonomy/` — the frozen core

| Module | Key file | Does |
|---|---|---|
| `core/` | `types.py`, `config.py`, `interfaces.py`, `geometry.py` | Dataclasses (`Detection`, `ObjectState`, `Trajectory`, `VehicleState`), YAML config loading, the abstract interfaces, geometry helpers |
| `perception/` | `tracker.py` | `SensorFusionTracker`: Kalman filter, extended-Kalman radar update, gating, association, lifecycle, duplicate merging, Phase 4 timestamp handling |
| `prediction/` | `predictor.py` | `ConstantVelocityPredictor`: constant-acceleration with per-class intent priors |
| `risk/` | `risk_engine.py` | Two time-to-collision views (route and physical), risk levels and scores |
| `behavior/` | `state_machine.py` | 7-state FSM with dual hysteresis |
| `planning/` | `planner.py`, `candidate_generator.py`, `collision_checker.py`, `scorer.py` | Corridor-frame lattice, quintic laterals, reverse candidates, swept-footprint checking, weighted scoring |
| `control/` | `lateral.py`, `longitudinal.py`, `tracker.py` | Stanley (with a reverse branch), PID, command assembly |
| `safety/` | `supervisor.py` | Authoritative override |
| `vehicle/` | `models.py`, `actuators.py` | Kinematic bicycle with a reverse gear; angle and rate saturation |
| `metrics/` | `collector.py` | Every metric the reports quote |
| `telemetry/` | `telemetry.py`, `dashboard.py`, `dashboard_page.py` | Publisher and sinks, live dashboard |

## `simulation/`

| Module | Key file | Does |
|---|---|---|
| `runner.py` | `Simulation`, `run_scenario` | **The main loop.** Owns one step of the whole pipeline |
| `world/` | `world.py`, `road.py` | Agent container, `DrivableSpace` road model |
| `agents/` | `agent.py`, `behaviors.py`, `interaction.py` | Agent state and behaviours, including opt-in reactive traffic |
| `sensors/` | `models.py` | Camera, LiDAR and radar models producing `Detection` |
| `scenarios/` | `loader.py` + 11 YAML files | Scenario definitions |

## Perception packages (outside the frozen core)

| Package | Key files | Does |
|---|---|---|
| `road_perception/` | `model.py` (Fast-SCNN), `dataset.py`, `drivable.py`, `planner_bridge.py`, `integration.py` | Segmentation to drivable corridor to `DrivableSpace`, with a safety gate |
| `perception_detector/` | `detector.py`, `geometry.py`, `uvh26.py`, `detect_eval.py` | Camera detector, monocular box-to-`Detection` geometry, UVH-26 loader, detection scoring |
| `dataset_adapters/` | `nuscenes.py`, `nuscenes_2d.py` | nuScenes table reader and 3D-to-2D projection |

## `scripts/` by purpose

| Purpose | Scripts |
|---|---|
| Run the sim | `run_scenario.py`, `generate_results.py`, `sweep.py` |
| Validate | `final_validation.py`, `audit_stress.py`, `audit_adaptivity.py` |
| Reports | `build_final_report.py`, `export_matlab.py` |
| Segmentation | `train_segmentation.py`, `eval_segmentation.py`, `run_perception_loop.py`, `figure_perception_loop.py` |
| Detection | `train_uvh26_detector.py`, `eval_uvh26_ab.py`, `eval_detector.py`, `diagnose_uvh26_regression.py` |
| Datasets | `select_uvh26_subset.py`, `audit_uvh26_subset.py`, `audit_datasets.py` |
| Fusion | `validate_nuscenes_fusion.py`, `validate_temporal_fusion.py` |

## Tests, 348 total

| Group | Files | Tests |
|---|---|---|
| `tests/unit/` | 20 files | 292 |
| `tests/scenarios/` | 6 files | 55 |
| `tests/integration/` | 1 file | 1 |

**22 test functions are gated on local datasets or checkpoints** and skip cleanly when those are
absent, in `test_camera_detector.py` (5), `test_road_perception.py` (7),
`test_uvh26_finetune.py` (8) and `test_perception_planner_bridge.py` (2). All 13 tests in
`test_nuscenes_adapter.py` skip as a module when nuScenes is missing.
