# SIH26037 — Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads

A **closed-loop autonomy stack with simulated multi-sensor perception**.
Simulated camera, LiDAR and radar observe the world; a Kalman-filter tracker
fuses their detections into object tracks with covariance; a corridor-frame
lattice planner produces trajectories; a Stanley + PID controller turns them
into steering / acceleration / braking; actuator limits shape those commands;
a kinematic bicycle model moves the vehicle. Lane markings are optional: the
planner works on a drivable-space corridor and a reference direction. There is
no machine learning: object classes come from the simulated camera's noisy
classifier, and the stack is honest about that.

```
World -> Sensors (camera/LiDAR/radar) -> Detections -> Tracker/Fusion -> ObjectState[]
      -> Prediction -> Risk -> Behaviour FSM -> Planner -> Tracker (Stanley + PID)
      -> Safety Supervisor -> Actuators -> Vehicle Dynamics -> World
                                   \-> Telemetry (JSONL / CSV / console / debug view)
```

`config/sensors.yaml: perception.mode` selects `sensors` (default) or
`ground_truth` (the baseline that isolates planning behaviour).

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design,
[docs/INTERFACES.md](docs/INTERFACES.md) for the data contracts, and
[docs/STAGE1_RESULTS.md](docs/STAGE1_RESULTS.md) for measured results.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate         # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m pytest tests -q                              # unit + integration + scenario tests
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING  # closed-loop run (sensors), console + logs/
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --perception ground_truth
python scripts/run_scenario.py SUDDEN_PEDESTRIAN_DART  # exercises the emergency-brake path
python scripts/run_scenario.py MIXED_TRAFFIC_CURVE     # curved corridor, merging auto-rickshaw, erratic pedestrian
python scripts/run_scenario.py NARROW_LANE_BOXED_IN   # boxed in by a pushcart: backs up, then takes the gap
python scripts/sweep.py                                # parameter sweep -> scenario success rate
python scripts/audit_adaptivity.py --perception sensors # jury audit: perturbations change the plan
python scripts/audit_stress.py --perception sensors     # jury audit: 12 stress cases, PASS/DEGRADED/FAIL

# engineering debug view from the telemetry log
python -m visualization.debug_view logs/sudden_cattle_crossing.jsonl --time 6.7 --save frame.png
python -m visualization.debug_view logs/sudden_cattle_crossing.jsonl --animate --save run.gif

# regenerate docs/STAGE1_RESULTS.md from fresh runs (never edit its numbers by hand)
python scripts/generate_results.py
```

### Live dashboard

```bash
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --dashboard --realtime   # http://127.0.0.1:8765/
python scripts/run_scenario.py MIXED_TRAFFIC_CURVE --dashboard 9000 --realtime 2 --perception sensors
```

`--dashboard [PORT]` adds a `DashboardSink` (stdlib `http.server`, no build
tooling, no external JS) that serves a self-contained page: top-down canvas
(corridor, ego footprint, tracked objects with type labels, predicted paths,
candidate and selected trajectories), behaviour state and reason, speed, risk
level / score / TTC, planner feasible / rejected counts with the rejection
histogram, the safety-override flag and rolling speed / steering strips. Frames
stream over Server-Sent Events at ~10 Hz (`/stream`); `/latest` and `/metrics`
return JSON. `--realtime [SPEED]` paces the run at wall-clock speed so the page
shows the manoeuvre rather than only the end state. The server stays up after
the run until Ctrl+C.

### MATLAB / Simulink exports (untested in MATLAB)

```bash
python scripts/export_matlab.py        # -> matlab/buses.m, behavior_transitions.{m,csv}, behavior_states.csv, replay_telemetry.m
```

`matlab/` is generated from the Python sources: `Simulink.Bus` definitions
introspected from the dataclasses in `autonomy/core/types.py`, the behaviour
FSM's declared transition table for a Stateflow chart, and a replay script
that loads a JSONL telemetry log and re-checks the scenario-test invariants.
No MATLAB was available while writing them, so they are **untested in MATLAB**;
see [matlab/README.md](matlab/README.md).

Requirements: Python 3.11+, numpy, PyYAML; matplotlib for the debug view;
pytest for tests. No MATLAB needed for Stage 1.

## What a run looks like

```
t=  1.50s | CRUISE  | v=10.00 m/s ... | risk NONE   0.04 TTC    inf | sel d+1.75_v10.0 J= 0.693 feas 42/43 | 33.9 ms
t=  2.00s | CAUTION | v= 8.74 m/s ... | risk MEDIUM 0.21 TTC  3.80s | sel d+1.75_v6.0  J= 0.217 feas 43/43 | 35.2 ms
        -> Route at desired speed intersects a predicted object in 3.8 s (beyond the avoidance horizon); reducing speed. ...
t=  3.20s | AVOID   | v= 5.95 m/s ... | risk HIGH   0.33 TTC  2.90s | sel d+1.75_v5.6  J= 0.923 feas 41/43 | 41.3 ms
        -> Predicted trajectory of cattle_1 (CATTLE) intersects ego trajectory within 2.9 s (risk 0.33).
t=  6.70s | AVOID   | v= 2.86 m/s ... | risk HIGH   0.70 TTC  1.40s | sel d+2.25_v7.0  J= 2.515 feas 32/43 | 27.9 ms
t=  8.10s | CRUISE  | v= 6.05 m/s ... | risk LOW    0.10 TTC    inf | sel d+2.25_v10.0 J= 1.903 feas 26/43 | 31.9 ms
        -> Risk cleared (max risk 0.10); resuming cruise.
```

Every number is produced by the running simulation. The `sel` column is the
selected candidate (lateral end offset, terminal speed), `J` its total cost,
`feas` feasible/total candidates, and the last column the measured planner
latency.

![cattle avoid](docs/img/cattle_avoid_t6.7.png)

## Repository layout

```
autonomy/        the stack: core (types, geometry, config), vehicle, prediction, risk,
                 behavior, planning, control, safety, metrics, telemetry (incl. live dashboard sink)
matlab_export/   generator for matlab/ (Simulink buses, FSM table, telemetry replay; untested in MATLAB)
simulation/      world (drivable-space corridor), agents (behaviours), scenarios (YAML), runner
visualization/   read-only matplotlib debug view over telemetry frames
config/          vehicle.yaml, autonomy.yaml (all tunables), object_profiles.yaml
tests/           unit / integration / scenarios
scripts/         run_scenario.py, generate_results.py, export_matlab.py
docs/            ARCHITECTURE.md, INTERFACES.md, STAGE1_RESULTS.md, img/
logs/            telemetry output (git-ignored)
```

## Scenarios

| Scenario | What it exercises |
|---|---|
| `UNMARKED_VILLAGE_ROAD` | SIH #1: 5.6 m road narrowing to 4.6 m, parked pushcart, slow bicycle, walking pedestrian, wandering cow |
| `UNSIGNALIZED_INTERSECTION` | SIH #2: crossroads without signals, car / auto-rickshaw / motorcycle crossing from both sides, corner pedestrian |
| `HIGHWAY_MERGE_SLOW_VEHICLES` | SIH #3: 20 m/s ego, slow truck, auto-rickshaw merging with gap acceptance, oncoming bus, later car merge |
| `DENSE_MARKET_MIXED_TRAFFIC` | SIH #4: market street, parked carts, slow auto, filtering motorcycle, crossing and walking pedestrians, standing cow |
| `SUDDEN_CATTLE_CROSSING` | SIH #5 and the acceptance scenario: predicted crossing, CAUTION -> AVOID -> CRUISE, pass and return to route |
| `SUDDEN_PEDESTRIAN_DART` | short-range dart: independent safety supervisor, EMERGENCY_BRAKE and recovery |
| `MIXED_TRAFFIC_CURVE` | 40-degree curve (R = 60 m), auto-rickshaw merging then following the road, erratic pedestrian, FOLLOW state |
| `NARROW_LANE_BOXED_IN` | pushcart abandoned across a 7 m lane, too close to steer around from rest: REVERSING recovery, then the bypass |

## Adding a scenario

Scenarios are data: copy a file in `simulation/scenarios/`, change the
corridor (explicit polylines, or `segments` of `straight` / `arc` for curves),
ego setup, goal and agents (behaviours `STATIC`, `CONSTANT_VELOCITY`,
`CROSSING`, `MERGING`, `ERRATIC`; types from `config/object_profiles.yaml`;
positions in `x, y` or corridor `s, d`). Nothing in `autonomy/` needs to change.

## Engineering rules kept

* The ego is never teleported; only `VehicleModel.step()` moves it.
* No scripted ego manoeuvres. Agents are scripted; the ego's response emerges.
* No lane dependence. `lane_markings: NONE` in every scenario.
* No magic numbers: every tunable is in `config/`.
* Deterministic: seeded agents, fixed timestep.
* Metrics are computed from what happened, and `STAGE1_RESULTS.md` is
  generated from real runs.
