# SIH26037 — Autonomy Console (UI)

A Three.js operations console over the **frozen** autonomy stack (`2ccd232`).

Read-only visualization, replay and scenario control. It computes no plan, no risk and no control
command, and it never writes to the simulation.

---

## Start it

```bash
python -m ui.server
```

Opens <http://127.0.0.1:8770/>. Press **Start demo** on the title screen, or **Open console**.

```bash
python -m ui.server --port 9000 --no-browser
```

**Dependencies: none beyond the project's existing ones.** The server is Python standard library
only. Three.js r160 and OrbitControls are **vendored into `ui/static/vendor/`**, so the console
works with no network and no build step. Nothing is fetched from a CDN.

### Demo mode

Press **Start demo**, or the **Demo** button in the transport panel. It plays the five demo
scenarios in order — village road, dense market, cattle crossing, pedestrian dart, reverse recovery
— advancing automatically. **Prev** / **Next** step manually. Demo mode only automates playback; it
changes no scenario behaviour.

---

## Where the data comes from

**Replay is the default and the demo path.** Every frame was published by the frozen stack during a
real run and recorded to disk:

```
scenario -> Simulation (unmodified) -> TelemetryPublisher -> CaptureSink -> ui/replay/*.json.gz
                                                          -> browser (scrub, seek, inspect)
```

Recording uses `scripts.run_scenario.run_with_sinks` with one extra `TelemetrySink` attached — the
documented extension point ("*a WebSocket / REST / MATLAB adapter is just another sink*"). Frames
are built by `autonomy.telemetry.dashboard.compact_frame`, imported unchanged.

Nothing is interpolated, smoothed or synthesised. Floats are rounded to 3 dp (a millimetre) purely
to keep files small; no displayed value changes.

Re-record at any time:

```bash
python -m ui.capture                 # the 5 demo scenarios, sensor mode
python -m ui.capture --all           # all 11 scenarios, both perception modes
python -m ui.capture --scenario DENSE_MARKET_MIXED_TRAFFIC --mode ground_truth
```

**Live mode also works.** `POST /api/run` runs a scenario through the same interface and streams
frames over SSE. Replay is preferred for judging because it is scrubbable and deterministic.

### Data source per metric

| Panel | Source |
|---|---|
| Simulation time, ego pose, speed, steering | `frame.ego` |
| Acceleration, brake | `frame.control` |
| Behaviour state, reason, target speed | `frame.behavior` |
| TTC physical, risk score, worst object | `frame.risk` |
| Min clearance, collisions | `frame.metrics` |
| Tracks, class, velocity, heading | `frame.objects` |
| Truth agents | `frame.agents` |
| Candidates, feasible, latency, rejections | `frame.plan` |
| Emergency brake | `frame.safety.override_active` |
| Replans | count of recorded `planning_cycle` frames up to now |
| Validation & Perception screens | `docs/FINAL_SYSTEM_RESULTS.json`, served as-is |

Anything genuinely absent renders as **NOT AVAILABLE**. Nothing is filled in.

---

## Screens

**Drive** — the Three.js scene plus telemetry. Road surface and boundaries, dashed direction
reference (*not* a lane line), ego vehicle built from primitives with working steering and brake
lights, tracked objects coloured by class with screen-space labels and velocity vectors, amber
prediction trails, the full candidate fan coloured by feasibility, the selected trajectory as a
glowing tube (red when `fallback`/`degraded`), and the travelled path. Click any object to inspect
it. Camera: **Follow / Chase / Top / Orbit** (keys `1`–`4`).

**Scenarios** — all eleven, with real per-scenario results. Click one to open its replay.

**Validation** — the 22-run evidence, clearance and latency charts, ground-truth vs sensor
comparison, and the eight runtime wiring checks.

**Perception** — the Phase 7 detector A/B including the bus, truck and bicycle regressions, with the
two-validations distinction stated on the page.

### Transport

`◀◀` back 2 s · play/pause (**Space**) · `▶▶` forward 2 s · restart · speed 0.25×–4× · scrubbable
timeline with coloured event marks. Every timeline event is clickable and seeks to its timestamp.

---

## "Why did we replan?"

Captured at each behaviour-state change. **The stack exposes no causal reason field**, so the panel
is labelled *derived* on screen and uses only measured values: the FSM's own `behavior.reason`
string, the highest-risk track from `risk.worst_object_id`, physical TTC, feasible/total candidate
counts, the state transition, commanded acceleration, and the selected candidate's clearance.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/replays` | Recorded runs with summaries |
| `GET /api/replay/<file>` | One recorded run (gzip) |
| `GET /api/results` | `FINAL_SYSTEM_RESULTS.json` (NaN → null so browsers can parse it) |
| `GET /api/scenarios` | Scenario list from the YAML files |
| `GET /api/stream` | SSE live telemetry |
| `POST /api/run` / `pause` / `reset` | Live run control |

---

## Tests

```bash
python -m pytest ui/tests -q      # 21 tests, about 40 s
python -m pytest tests/ -q        # the frozen 348-test suite, unchanged
```

UI tests live **outside `tests/`** so the frozen suite keeps its exact count. They verify the server
starts, scenarios come from real YAML, replay frames carry every field the scene draws, replay
outcomes match `FINAL_SYSTEM_RESULTS.json`, served results are strict JSON, path traversal is
refused, no asset points at a CDN, and the UI source calls no autonomy mutator.

---

## Display

Built for **1920×1080**; verified at that size. Breakpoints for 1440×900 and 1366×768 narrow the
rail and shorten the timeline. Not designed for phones.

## Boundaries

The UI does not calculate a plan, replace risk logic, write vehicle state, bypass the controller,
inject detections, modify scenario truth, or change a planner decision.
