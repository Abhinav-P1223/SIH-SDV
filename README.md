# SIH26037 — Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads

**A self-driving decision system built for roads that have no lane markings.**

Most autonomous-driving software assumes painted lanes, orderly traffic and vehicles that behave
predictably. Indian roads offer none of that. This project builds the part of a self-driving system
that decides **where to go and when to stop** — and it does that without ever needing a lane
marking.

---

## In one minute

**The problem.** A car that can only follow painted lines is useless on a village road, in a market
lane, or anywhere an auto-rickshaw, a cow and a motorcycle share the same six metres of tarmac.

**What we built.** A complete decision-making loop. Sensors observe the world, the software works
out where every moving thing is and where it is heading, judges the risk, chooses a safe path
through the open space, and steers and brakes the vehicle along it. It re-plans the whole route ten
times a second, and adjusts steering and braking fifty times a second.

**What we proved.** We put it through 11 hard situations — a cow stepping out, a pedestrian darting
into the road, a lane so narrow the car must reverse to escape, an unmarked village road, a crowded
market — and ran each one twice, under two different sensing conditions.

| What we measured | Result | Why it matters |
|---|---|---|
| Situations completed | **22 of 22** | It never got stuck or gave up |
| Collisions | **0** | It never hit anything, in any run |
| Closest it ever came | **0.51 m** | Half a metre of clearance at the worst moment |
| Time taken to decide | **55 ms**, budget 100 ms | It re-plans in about a twentieth of a second, comfortably inside the limit |
| Automated checks | **348 passing** | The behaviour is locked in and re-verifiable on demand |

**Two things a real Indian road demands, which we specifically handled:**

- **It reverses to get out of trouble.** When a lane is blocked ahead and the car cannot turn
  around, it backs out along a curved path rather than freezing in place.
- **It reads the road surface itself.** A neural network trained on real Indian road photographs
  finds the drivable area, so the planner works from open space rather than from lane paint.

**What we deliberately do not claim.** The full driving loop was validated in simulation, not on a
real vehicle. The two neural networks were validated separately, on real Indian road photographs.
We keep those two claims apart on purpose, and every document in this repository states the
distinction. That honesty is the point: nothing here rests on a demo video that could have been
staged. Every number above regenerates from a single command in about fifteen minutes.

---

## What still stands between this and a real vehicle

Stated plainly, because the gap is real:

1. **Run it on recorded real-world sensor data**, end to end, rather than simulated sensors.
2. **Speed up the camera detector.** It takes about 287 ms per image today; a car needs roughly 50.
3. **Get more training data for bicycles and pedestrians.** Our fine-tuning dataset contains
   neither in useful quantity. We tested whether a sampling trick could compensate, and confirmed it
   cannot — only more data will.
4. **Vehicle integration and road trials**, with the safety approvals that implies.

---

# For engineers

Everything below is technical detail. Everything above is the whole story.

## Architecture

```
World -> Sensors (camera/LiDAR/radar) -> Detections -> Tracker/Fusion -> ObjectState[]
      -> Prediction -> Risk -> Behaviour FSM -> Planner -> Controller (Stanley + PID)
      -> Safety Supervisor -> Actuators -> Vehicle Dynamics -> World
                                   \-> Telemetry (JSONL / CSV / console / live dashboard)
```

Simulated camera, LiDAR and radar observe the world. A Kalman-filter tracker fuses their detections
into object tracks with covariance. A corridor-frame lattice planner produces trajectories, a
Stanley and PID controller turns them into steering, acceleration and braking, actuator limits shape
those commands, and a kinematic bicycle model moves the vehicle. **Lane markings are optional**: the
planner works on a drivable-space corridor and a reference direction, which is the point on
unstructured roads.

## Measured results

| Metric | Result |
|---|---|
| Scenarios, runs | 11, 22 (two perception modes) |
| Completion | **22 / 22** |
| Collisions | **0** |
| Worst minimum clearance | 0.51 m |
| Worst p95 replanning latency | 55.0 ms against a 100 ms budget |
| Emergency brakes | 0 ground truth, 7 simulated sensors |
| Tests | **348 passed, 0 failed** with datasets present; about 35 skip without them, which is correct |

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

It writes `logs/sudden_cattle_crossing.jsonl` plus a summary CSV. The other ten scenario names are
`DENSE_MARKET_MIXED_TRAFFIC`, `HIGHWAY_MERGE_SLOW_VEHICLES`, `MIXED_TRAFFIC_CURVE`,
`NARROW_LANE_BOXED_IN`, `NARROW_LANE_MUTUAL_YIELD`, `NARROW_LANE_REVERSE_RECOVERY`,
`SUDDEN_PEDESTRIAN_DART`, `UNMARKED_VILLAGE_ROAD`, `UNPROTECTED_TURN` and
`UNSIGNALIZED_INTERSECTION` — one YAML each in `simulation/scenarios/`. Add `--dashboard` for a live
browser view, or `--realtime` to pace the run at wall-clock speed.

**Run the tests** (15 to 25 minutes; `tests/unit` alone is about 3 minutes):

```bash
python -m pytest tests/ -q
```

**Reproduce the final validation** (about 15 minutes):

```bash
python scripts/final_validation.py
```

**Open the console** — a Three.js operations view of recorded runs:

```bash
python -m ui.server
```

Then press **Start demo**. It plays five scenarios in order, each narrated from its own recorded
telemetry: object detected, risk rising, decision, emergency brake, reverse recovery, safe passage.
A Tests tab shows the real pytest cases covering whichever scenario is on screen. No extra
dependency, no build step, works offline. Details in [ui/README.md](ui/README.md).

**Demo:** follow [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md), a timed five-minute flow with a
"do not say" list so nothing unsupported is claimed on stage.

## Datasets, and where to download them

Both model checkpoints ship **inside the repository**, so a plain clone is enough for everything
above. **No dataset is required** to run the simulator, the scenarios or the final validation.
Datasets are only needed if you want to re-train or re-evaluate the perception models, and all
three are gitignored.

| Dataset | Size | Needed for | Download |
|---|---|---|---|
| **nuScenes v1.0-mini** | 5.1 GB | Fusion validation, detector evaluation | [direct tarball](https://www.nuscenes.org/data/v1.0-mini.tgz) &middot; [dataset page](https://www.nuscenes.org/nuscenes#download) |
| **IDD-Lite** (`idd20k_lite`) | 42 MB | Segmentation training and evaluation | [idd.insaan.iiit.ac.in/dataset/download](https://idd.insaan.iiit.ac.in/dataset/download/) — free account required |
| **UVH-26** (750-image subset) | 2.5 GB | Detector fine-tuning, reproducible from the tracked manifest | `python scripts/select_uvh26_subset.py --download` &middot; [source on Hugging Face](https://huggingface.co/datasets/iisc-aim/UVH-26) |

Extract into `Datasets/v1.0-mini/`, `Datasets/idd-lite/` and `Datasets/uvh26_subset/` respectively.
`Datasets/` is gitignored and must stay that way — **never commit dataset files**. Licences:
nuScenes is non-commercial research use, IDD requires accepting its terms at registration, UVH-26
is CC BY 4.0. Step-by-step instructions in [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

## Documentation

| Document | For |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **Start here.** Project status, frozen areas, conventions, commands |
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | The whole project in 10–15 minutes |
| [ui/README.md](ui/README.md) | The Three.js console |
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
- **The learned detector is not in the closed loop**, and at about 287 ms per image it is an offline
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
