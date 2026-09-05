"""Write every MATLAB/Simulink artefact into one directory (default `matlab/`)."""
from __future__ import annotations

import shutil
from pathlib import Path

from autonomy.behavior.state_machine import TRANSITIONS
from matlab_export.behavior import (STATE_FIELDS, TRANSITION_FIELDS, render_transitions_m, state_rows,
                                    transition_rows, write_csv)
from matlab_export.buses import BUS_CLASSES, render_buses_m

TEMPLATES = Path(__file__).parent / "templates"

FILES = ["buses.m", "behavior_transitions.m", "behavior_transitions.csv", "behavior_states.csv",
         "replay_telemetry.m", "README.md"]


def render_readme() -> str:
    buses = ", ".join(c.__name__ for c in BUS_CLASSES)
    return f"""# MATLAB / Simulink exports (generated, UNTESTED IN MATLAB)

Everything in this directory is produced by `python scripts/export_matlab.py`
from the Python sources; do not edit by hand. **None of it has been executed in
MATLAB**: no MATLAB installation was available. The files follow documented
Simulink.Bus, `table` and `jsondecode` syntax and are the intended starting
point for the Stage 2 Simulink/Stateflow model; expect to fix small syntax
slips on first run.

| File | Content | Source |
|---|---|---|
| `buses.m` | `Simulink.Bus` objects for {buses}; enum code table in the header | dataclasses in `autonomy/core/types.py` |
| `behavior_transitions.m` / `.csv` | the FSM's declared transition table ({len(TRANSITIONS)} rows, priority order) with guard names and doc | `autonomy/behavior/state_machine.py: TRANSITIONS` |
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
"""


def export(out_dir: Path | str = "matlab") -> list[Path]:
    """Generate all files into `out_dir` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "buses.m").write_text(render_buses_m(), encoding="utf-8")
    rows = transition_rows()
    (out / "behavior_transitions.m").write_text(render_transitions_m(rows), encoding="utf-8")
    write_csv(out / "behavior_transitions.csv", TRANSITION_FIELDS, rows)
    write_csv(out / "behavior_states.csv", STATE_FIELDS, state_rows())
    shutil.copyfile(TEMPLATES / "replay_telemetry.m", out / "replay_telemetry.m")
    (out / "README.md").write_text(render_readme(), encoding="utf-8")
    return [out / name for name in FILES]
