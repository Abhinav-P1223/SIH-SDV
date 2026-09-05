# MATLAB / Simulink exports (generated, UNTESTED IN MATLAB)

Everything in this directory is produced by `python scripts/export_matlab.py`
from the Python sources; do not edit by hand. **None of it has been executed in
MATLAB**: no MATLAB installation was available. The files follow documented
Simulink.Bus, `table` and `jsondecode` syntax and are the intended starting
point for the Stage 2 Simulink/Stateflow model; expect to fix small syntax
slips on first run.

| File | Content | Source |
|---|---|---|
| `buses.m` | `Simulink.Bus` objects for VehicleState, ControlCommand, ObjectState, RiskSummary, SpeedPolicy, BehaviorDecision, SimulationMetrics; enum code table in the header | dataclasses in `autonomy/core/types.py` |
| `behavior_transitions.m` / `.csv` | the FSM's declared transition table (14 rows, priority order) with guard names and doc | `autonomy/behavior/state_machine.py: TRANSITIONS` |
| `behavior_states.csv` | one row per `BehaviorState`: int code, severity, default speed policy | `state_machine.py: SEVERITY`, `_policy` |
| `replay_telemetry.m` | loads a JSONL telemetry log, plots speed / steering / state timeline / min clearance, asserts the scenario-test invariants | `autonomy/telemetry/telemetry.py: JsonLinesSink` |

## Running (in MATLAB, untested)

```matlab
cd <repo>/matlab
buses                                    % creates VehicleState, ControlCommand, ... in the base workspace
behavior_transitions                     % tables T (transitions) and S (states)
replay_telemetry('../logs/sudden_cattle_crossing.jsonl')   % produce the log first with scripts/run_scenario.py
```

`replay_telemetry` reads actuator limits from `config/vehicle.yaml` and the
safety margin from `config/autonomy.yaml` (planning section) with a minimal
two-level YAML reader, then asserts: no collision, scenario completed, minimum
clearance >= safety margin, steering angle / rate, acceleration / deceleration
and speed within the vehicle limits.

## Mapping notes

* Enums become `int32`; int-valued enums (`RiskLevel`) keep their value, string
  enums (`BehaviorState`, `ObjectType`) use their declaration index. The codes
  are listed in the header of `buses.m` and in `behavior_states.csv`.
* `Optional[float]` becomes `double` with `NaN` for `None`; `Optional[int]`
  becomes `int32` with `-1` for `None`.
* Strings (`reason`, `id`, `source`), lists (`assessments`), dicts (`triggers`,
  `rejection_histogram`) and tuples are not bus elements; each is listed as a
  `skipped field` comment above its bus so nothing disappears silently.
* `BehaviorDecision.speed_policy` is a nested `Bus: SpeedPolicy`.
* Transitions in `behavior_transitions.csv` whose `dwell_guarded_from` column
  is non-empty are de-escalations from those states and additionally require
  `time_in_state >= behavior.min_dwell_s` (see `BehaviorStateMachine.decide`).
