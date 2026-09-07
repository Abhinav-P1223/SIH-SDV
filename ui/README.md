# SIH26037 — Autonomy Console (UI)

A presentation-ready operations console over the **frozen** autonomy stack (`2ccd232`).

The UI is a **read-only** visualization, telemetry and scenario-control layer. It computes no plan,
no risk and no control command, and it never writes to the simulation.

---

## Run it

```bash
python -m ui.server
```

Opens <http://127.0.0.1:8770/> in your browser.

```bash
python -m ui.server --port 9000 --no-browser     # options
```

**Dependencies:** none beyond what the project already requires. The server is Python standard
library only; the page uses no framework, no build step and no CDN.

---

## How it gets telemetry

The console attaches **one extra `TelemetrySink`** to the existing runner. That is the documented
extension point — `autonomy/telemetry/telemetry.py` says *"A WebSocket / REST / MATLAB adapter is
just another sink."*

```
scenario -> Simulation (unmodified) -> TelemetryPublisher -> ConsoleStreamSink -> SSE -> browser
```

Runs are launched through `scripts.run_scenario.run_with_sinks`, the interface the CLI already uses.
Frames are built by `autonomy.telemetry.dashboard.compact_frame`, imported unchanged, so the console
sees exactly the contract the existing dashboard already publishes. **No autonomy file was modified
and no telemetry field was invented.**

---

## Screens

| Tab | Shows |
|---|---|
| **Live** | Scene, behaviour/safety rail, perception + planning + system strip, "why did we replan?", event timeline |
| **Validation** | The 22-run final validation, served verbatim from `docs/FINAL_SYSTEM_RESULTS.json` |
| **Perception** | The Phase 7 detector A/B, including the regressions |

### Live scene

Everything drawn is a telemetry field:

| Drawn | Source field |
|---|---|
| Road corridor + boundaries | `road.left_boundary`, `road.right_boundary` |
| Direction reference (dashed) | `road.reference` — a direction, **not** a lane line |
| Ego vehicle footprint | `ego.x/y/yaw` + `vehicle.length/width/footprint_center_offset` |
| Ego path history | accumulated `ego.x/y` |
| Tracked objects | `objects[]` — type, id, position, heading, extent |
| Object velocity vector | `objects[].vx/vy`, drawn as 1 s of travel |
| Predicted motion | `predictions[].x/y`, with a horizon-end marker |
| Candidate trajectories | `plan.candidates[].x/y`, coloured by `.feasible` |
| Selected trajectory | `plan.selected.x/y`, glowing; red when `fallback` or `degraded` |
| Emergency brake | `safety.override_active` — ego turns red, banner appears |

The whole console's accent colour is driven by `risk.max_level`, so the chrome shifts from cyan to
amber to red as the situation degrades.

### "Why did we replan?"

Captured on every behaviour-state change. **The stack exposes no causal reason field**, so the panel
is explicitly labelled *derived* and uses only measured values: the FSM's own `behavior.reason`
string, the highest-risk track from `risk.worst_object_id`, physical TTC, the feasible/total
candidate counts, the commanded acceleration and the selected candidate's clearance. Nothing is
inferred beyond that.

---

## Controls

| Control | Effect |
|---|---|
| **Scenario** | The 11 scenarios, read from `simulation/scenarios/*.yaml`. Demo scenarios listed first |
| **Perception** | `sensors` or `ground_truth` — the existing `PerceptionConfig` modes |
| **Speed** | Wall-clock pacing: 1×, 2×, 4×, or Max (unpaced) |
| **Run / Pause / Reset** | Start, hold, and stop the run |
| **Demo** | Runs the five demo scenarios back to back |

Pause holds the simulation thread inside the sink between frames; it does not alter any state.
Speed only changes pacing. **No control mutates safety-critical configuration**, and there is no way
to inject a detection, move the ego or change a scenario from the UI.

### Demo mode

Press **Demo**. It runs, in order:

1. Unmarked village road
2. Dense market mixed traffic
3. Sudden cattle crossing
4. Sudden pedestrian dart
5. Narrow lane reverse recovery

Each runs to completion, pauses ~2 s on its summary, then advances. Scenario behaviour is completely
unchanged — Demo only automates the Run button.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/scenarios` | Scenario list, demo order, perception modes |
| `GET /api/state` | Server status |
| `GET /api/stream` | SSE stream of compact frames |
| `GET /api/results` | `docs/FINAL_SYSTEM_RESULTS.json`, verbatim |
| `POST /api/run` | `{scenario, perception, speed}` |
| `POST /api/pause` | Toggle pause |
| `POST /api/reset` | Stop and clear |

---

## Tests

UI tests live **outside `tests/`** so the frozen autonomy suite keeps its exact 348-test count.

```bash
python -m pytest ui/tests -q      # 14 tests, about 40 s
python -m pytest tests/ -q        # the frozen suite, unchanged
```

The UI suite verifies the server starts, the scenario list comes from real YAML files, static assets
exist and reference nothing external, a live run produces frames carrying every field the scene
draws, `/api/results` matches the JSON byte for byte, reset clears state, bad requests are rejected,
path traversal is refused, and the UI source calls no autonomy mutator.

---

## Display

Built for **1920×1080**. Verified layouts at 1440×900 and 1366×768 — the rail narrows and the strip
reflows to two rows. Not optimised for phones, by design.

---

## What the UI must never do

It does not calculate a plan, replace risk logic, write vehicle state, bypass the controller, inject
detections, modify scenario truth, or change a planner decision. If a value is not in the telemetry
frame, the console shows `—` rather than filling the gap.
