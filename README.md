# SIH26037 — Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads

A **closed-loop autonomy stack with simulated multi-sensor perception**, plus two learned
perception components validated offline against real Indian-road imagery.

Simulated camera, LiDAR and radar observe the world. A Kalman-filter tracker fuses their detections
into object tracks with covariance. A corridor-frame lattice planner produces trajectories, a
Stanley and PID controller turns them into steering, acceleration and braking, actuator limits shape
those commands, and a kinematic bicycle model moves the vehicle. **Lane markings are optional**: the
planner works on a drivable-space corridor and a reference direction, which is the point on
unstructured roads.

```
World -> Sensors (camera/LiDAR/radar) -> Detections -> Tracker/Fusion -> ObjectState[]
      -> Prediction -> Risk -> Behaviour FSM -> Planner -> Controller (Stanley + PID)
      -> Safety Supervisor -> Actuators -> Vehicle Dynamics -> World
                                   \-> Telemetry (JSONL / CSV / console / live dashboard)
```

## Measured results

| Metric | Result |
|---|---|
| Scenarios, runs | 11, 22 (two perception modes) |
| Completion | **22 / 22** |
| Collisions | **0** |
| Worst minimum clearance | 0.51 m |
| Worst p95 replanning latency | 55.0 ms against a 100 ms budget |
| Emergency brakes | 0 ground truth, 7 simulated sensors |
| Tests | **348 passed, 0 failed, 0 skipped** |

Full evidence in [docs/FINAL_SYSTEM_REPORT.md](docs/FINAL_SYSTEM_REPORT.md), machine-readable in
[docs/FINAL_SYSTEM_RESULTS.json](docs/FINAL_SYSTEM_RESULTS.json).

## Learned perception, and exactly what it does and does not show

Two learned components were built and validated **offline**, on real imagery:

- **Drivable-space segmentation** (Fast-SCNN on IDD-Lite): drivable IoU 0.877. Its corridor output
  becomes the planner's own road type through a safety gate.
- **Camera object detection** (Faster R-CNN MobileNetV3, fine-tuned on a 750-image UVH-26 subset):
  motorcycle recall improved from 0.151 to 0.379, and auto-rickshaw became detectable at all, since
  the generic COCO model has no such class.

> **The distinction that matters.** The learned detector was validated independently on held-out
> UVH-26 imagery, and its output contract was verified against the unmodified tracking pipeline. The
> autonomy stack was validated independently, end to end, through simulated multimodal sensors.
> These are two separate validations. **The simulator renders no image frames**, so no detector
> consumed camera pixels during the closed-loop runs.

## Quick start

Needs **Python 3.11 or newer**. CPU only, no GPU. **No dataset is required** to run the simulator.

```bash
git clone https://github.com/Abhinav-P1223/SIH-SDV.git
cd SIH-SDV
python -m venv .venv && .venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

That one command installs everything: numpy, PyYAML, matplotlib, pytest, torch, torchvision, scipy,
pillow and requests. It pulls PyTorch, so expect a few hundred megabytes.

**On Linux**, install CPU-only PyTorch first so pip does not fetch multi-gigabyte CUDA wheels you
will never use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run commands from the repository root.

**Run a scenario** (about 20 seconds):

```bash
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet
```

**Run the tests** (15 to 25 minutes; `tests/unit` alone is about 3 minutes):

```bash
python -m pytest tests/ -q
```

**Reproduce the final validation** (about 15 minutes):

```bash
python scripts/final_validation.py
```

**Demo:** follow [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md), a timed five-minute flow with a
"do not say" list so nothing unsupported is claimed on stage.

## Prerequisites you may or may not need

Both model checkpoints ship **inside the repository**, so a plain clone is enough for everything
above. Datasets are only needed for perception work and are gitignored:

| Dataset | Size | Needed for |
|---|---|---|
| nuScenes v1.0-mini | 5.1 GB | Fusion validation, detector evaluation |
| IDD-Lite | 42 MB | Segmentation training and evaluation |
| UVH-26 subset | 2.5 GB | Detector fine-tuning, reproducible from the tracked manifest |

Never commit dataset files. See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

## Documentation

| Document | For |
|---|---|
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Get it running in ten minutes |
| [docs/TEAM_HANDOFF.md](docs/TEAM_HANDOFF.md) | Everything: architecture, commands, frozen areas, workflow |
| [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) | Where everything lives |
| [docs/TEAM_ONBOARDING_CHECKLIST.md](docs/TEAM_ONBOARDING_CHECKLIST.md) | Tick-box first day |
| [docs/TECHNICAL_ARCHITECTURE.md](docs/TECHNICAL_ARCHITECTURE.md) | Runtime data flow |
| [docs/FINAL_SYSTEM_REPORT.md](docs/FINAL_SYSTEM_REPORT.md) | The evidence package |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/INTERFACES.md](docs/INTERFACES.md) | Original design and data contracts |
| `docs/PHASE*.md` | Per-phase reports, each with its own limitations section |

## Limitations, stated plainly

- **Simulated sensors, not recorded ones.** Real-data validation covers the fusion interface
  (nuScenes) and the two learned models (real images), not the closed loop.
- **The learned detector is not in the closed loop**, and at about 250 ms per image it is an offline
  evidence pipeline rather than a real-time front end.
- **Camera perception is unvalidated on dashcam imagery.** UVH-26 is elevated CCTV; the segmentation
  work used still photographs with assumed calibration.
- **Bicycle detection is data-limited** at 106 training boxes, and an oversampling experiment
  confirmed no sampling scheme repairs it. Pedestrian and animal detection were not improved by the
  fine-tuning data, which contains neither class.
- **Temporal fusion ships disabled**, because enabling it makes the stack more conservative by
  removing an over-confidence the collision margins were tuned against.
- **The MATLAB and Simulink exports have never been run in MATLAB.**

The autonomy implementation is **frozen** at commit `2ccd232`. See
[docs/TEAM_HANDOFF.md](docs/TEAM_HANDOFF.md) section 11 before changing anything under `autonomy/`,
`simulation/scenarios/`, `tests/` or `config/`.
