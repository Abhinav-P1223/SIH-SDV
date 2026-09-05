# SIH26037 — Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads

Stage 1: **closed-loop autonomy core**. A planner produces trajectories, a
controller turns them into steering / acceleration / braking, actuator limits
shape those commands, and a kinematic bicycle model moves the vehicle. Ground
truth obstacles enter through the same `ObjectState` interface that sensor
fusion will use in Stage 2. Lane markings are optional: the planner works on a
drivable-space corridor and a reference direction.

```
World -> ObjectState[] -> Prediction -> Risk -> Behaviour FSM -> Planner
      -> Tracker (Stanley + PID) -> Safety Supervisor -> Actuators -> Vehicle Dynamics -> World
                                   \-> Telemetry (JSONL / CSV / console / debug view)
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design,
[docs/INTERFACES.md](docs/INTERFACES.md) for the data contracts, and
[docs/STAGE1_RESULTS.md](docs/STAGE1_RESULTS.md) for measured results.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate         # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m pytest tests -q                              # unit + integration + scenario tests
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING  # closed-loop run, console + logs/
python scripts/run_scenario.py SUDDEN_PEDESTRIAN_DART  # exercises the emergency-brake path
python scripts/run_scenario.py MIXED_TRAFFIC_CURVE     # curved corridor, merging auto-rickshaw, erratic pedestrian
python scripts/sweep.py                                # parameter sweep -> scenario success rate

# engineering debug view from the telemetry log
python -m visualization.debug_view logs/sudden_cattle_crossing.jsonl --time 6.7 --save frame.png
python -m visualization.debug_view logs/sudden_cattle_crossing.jsonl --animate --save run.gif

# regenerate docs/STAGE1_RESULTS.md from fresh runs (never edit its numbers by hand)
python scripts/generate_results.py
```

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
                 behavior, planning, control, safety, metrics, telemetry
simulation/      world (drivable-space corridor), agents (behaviours), scenarios (YAML), runner
visualization/   read-only matplotlib debug view over telemetry frames
config/          vehicle.yaml, autonomy.yaml (all tunables), object_profiles.yaml
tests/           unit / integration / scenarios
scripts/         run_scenario.py, generate_results.py
docs/            ARCHITECTURE.md, INTERFACES.md, STAGE1_RESULTS.md, img/
logs/            telemetry output (git-ignored)
```

## Scenarios

| Scenario | What it exercises |
|---|---|
| `SUDDEN_CATTLE_CROSSING` | the prompt's acceptance scenario: predicted crossing, CAUTION -> AVOID -> CRUISE, pass and return to route |
| `SUDDEN_PEDESTRIAN_DART` | short-range dart: independent safety supervisor, EMERGENCY_BRAKE and recovery |
| `MIXED_TRAFFIC_CURVE` | 40-degree curve (R = 60 m), auto-rickshaw merging then following the road, erratic pedestrian, FOLLOW state |

## Adding a scenario

Scenarios are data: copy a file in `simulation/scenarios/`, change the
corridor (explicit polylines, or `segments` of `straight` / `arc` for curves),
ego setup, goal and agents (behaviours `STATIC`, `CONSTANT_VELOCITY`,
`CROSSING`, `MERGING`, `ERRATIC`; types from `config/object_profiles.yaml`;
positions in `x, y` or corridor `s, d`). Nothing in `autonomy/` needs to change.

## Engineering rules kept in Stage 1

* The ego is never teleported; only `VehicleModel.step()` moves it.
* No scripted ego manoeuvres. Agents are scripted; the ego's response emerges.
* No lane dependence. `lane_markings: NONE` in every scenario.
* No magic numbers: every tunable is in `config/`.
* Deterministic: seeded agents, fixed timestep.
* Metrics are computed from what happened, and `STAGE1_RESULTS.md` is
  generated from real runs.
