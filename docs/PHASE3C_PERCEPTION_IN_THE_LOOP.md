# Phase 3C — the learned corridor drives the existing planner

    IDD-Lite image -> Fast-SCNN -> drivable mask -> corridor -> SAFETY GATE -> DrivableSpace
                   -> existing planner -> existing risk -> existing controller -> vehicle model

**No file under `autonomy/` or `simulation/` was changed.** Phase 3C adds one module and two
scripts. The planner cannot tell whether its road came from scenario YAML or from a camera,
because in both cases it receives the same `DrivableSpace` object.

## 1. Where the learned corridor enters

`simulation/runner.py` takes the road from `scenario.world.road` and hands it to the planner, the
predictor and the risk engine. So the smallest possible adapter is one that **produces a
`DrivableSpace`**, and that is exactly what `road_perception/planner_bridge.py` does. The runner,
the planner, the collision checker, the scorer, the behaviour FSM, the Stanley/PID controller, the
safety supervisor and the kinematic bicycle model are all used unmodified and unaware.

| Stage | Where |
|---|---|
| Image to 7-class mask | `road_perception/integration.py`, Phase 3B, unchanged |
| Mask to corridor in the ego frame | `road_perception/drivable.py`, Phase 3B, unchanged |
| **Gate, anchor, `DrivableSpace`** | **`road_perception/planner_bridge.py`, new** |
| Everything downstream | shipped code, untouched |

## 2. The interface, stated exactly

`extract_corridor` returns three polylines in metres in the **ego frame**: x forward, y left.
`DrivableSpace` is in the **world frame**. `AnchorPose` is the only thing that relates them and it
has **no default value**, so a caller cannot accidentally forget that a corridor without a pose
means nothing. The bridge never integrates ego motion and never invents a pose, because a still
photograph carries neither.

## 3. The safety gate

Phase 3B measured this model's weakness precisely: 46 of 204 validation images have a corridor
boundary off by more than 3 m on one side. Geometry that wrong must not reach a planner that
trusts its road model. Every corridor therefore clears an explicit gate first.

| Check | Threshold | Why |
|---|---|---|
| Stations | at least 5 | Fewer than this is not a road |
| Finite coordinates | all | A NaN would enter the planner's arc-length maths silently |
| Monotonic range | strict | Stations must march away from the vehicle |
| **Usable width walk** | **2.6 m** | Vehicle 1.8 m plus 0.4 m each side |
| **Usable lookahead** | **8 m** | Far enough to plan a stop inside |
| Station spacing | 6 m, growing with range | Catches a genuine hole in coverage |
| Width ceiling | 30 m | The mask has bled into pavement or sky |
| Centreline continuity | 3.0 m per metre | A corridor that teleports would whip the controller |
| **Heading continuity** | **30 degrees per metre** | A saw-tooth centreline is extraction noise, not a road |

**A rejected corridor is rejected.** It is not repaired, not smoothed into plausibility, and never
replaced by the simulator's ground-truth road. The bridge has exactly one source of geometry: the
mask it was handed. When the gate refuses, `road` is `None`, the reason is recorded with a stable
cause code, and the episode stops. A test asserts precisely this, because it is the claim a jury
would attack first.

### Two gate designs that were wrong, and how that was caught

Both were found by running the gate on **ground-truth** masks. Ground truth failing is proof the
gate is wrong, not the data.

**A global minimum-width test rejected 204 of 204 ground-truth corridors.** A road converges under
perspective, so its farthest station always sits near the vanishing point and the narrowest width
across a whole corridor is always tiny. The fix is to walk outward from the vehicle and take the
contiguous stretch that is genuinely wide enough, then **clip the corridor to it**. An accepted
corridor is often shorter than what was extracted, and that is the point.

**A fixed station-spacing test rejected 175 of 204.** Stations are spaced by image rows, which
under perspective means 0.12 m apart at the bumper and 9.6 m apart near the horizon. The allowance
now grows with range.

**The heading check was calibrated, not guessed.** Corridors that made the vehicle time out were
measured against corridors it drove cleanly, using ground-truth masks only. The centreline heading
step separates them: successful corridors sit at 23 degrees per metre at the median, stuck ones at
49. A 30 degree limit keeps 70 of 117 driveable ground-truth corridors while removing 31 of the 37
stuck ones. The threshold was then applied **unchanged** to predictions, so both modes face the
same bar. The mechanism is not mysterious: inside a saw-tooth corridor the planner's Frenet frame
is ill-conditioned, no candidate is feasible, and the vehicle sits still until the run times out.

There is also deliberately **no width-slope check**. Under a fixed field of view the corridor
cannot be wider than the view cone, which spans 1.16 m of width per metre of range at 60 degrees,
so a well-formed corridor's width must climb at roughly that rate. Measured on ground truth the
width slope runs 1.19 m/m at the median and 6.1 m/m at p99, so any threshold either admits
everything or rejects valid ground truth. Width is guarded by a floor and a ceiling instead.

## 4. The A/B experiment

Each validation image is one independent episode. The ego starts at the origin facing along the
corridor, which makes the ego frame and the world frame the same frame and the anchor an identity.

| | Mode A | Mode B |
|---|---|---|
| Drivable mask from | ground-truth label | **the trained network** |
| Corridor extraction | identical | identical |
| Safety gate | identical | identical |
| Planner, risk, controller, vehicle | identical | identical |

Everything downstream of the mask is the same code on the same configuration. The comparison is
therefore a comparison of perception, which is the only thing this phase changes.

## 5. Results

204 validation images, one episode per image per mode, produced by
`scripts/run_perception_loop.py` and stored in `docs/phase3c_results.json`.

### 5.1 What the safety gate did

| | Mode A, ground truth | Mode B, predicted |
|---|---|---|
| Accepted | 76 / 204 | **74 / 204** |
| Refused: heading step | 78 | 63 |
| Refused: short lookahead | 35 | 40 |
| Refused: too narrow | 15 | 27 |
| Both modes accepted, the paired set | | **54** |

The gate is conservative and that is deliberate. It refuses about 63% of ground-truth corridors
too, because most IDD-Lite frames are congested streets where the visible drivable region is
short, pinched by parked vehicles, or too jagged to be a road. **A gate that only refused
predictions would be measuring the model; this one measures the geometry.** Mode B is refused
slightly more often, and the extra refusals fall where you would expect from Phase 3B: 27 too
narrow against 15, because the model under-predicts road width.

### 5.2 The A/B comparison, on the 54 images both modes accepted

Identical planner, risk engine, behaviour FSM, controller, safety supervisor and vehicle model.
The only difference is where the drivable mask came from.

| | Mode A | Mode B |
|---|---|---|
| Episodes | 54 | 54 |
| **Reached the goal** | **53** | **50** |
| Collisions | 0 | 0 |
| Corridor width, median | 5.67 m | 5.33 m |
| Usable lookahead, median | 33.4 m | 33.4 m |
| Distance driven, median | 30.59 m | 30.46 m |
| Planning cycles, median | 54 | 54 |
| Planning latency, median | 12.7 ms | 10.9 ms |
| Max cross-track error, median | 0.30 m | 0.29 m |
| Safety interventions | 0 | 2 |
| Perception cost, median | 8.5 ms | **32.3 ms** |
| Infeasible planning cycles | 600 | 2378 |

Corridor geometry, Mode B measured against Mode A on the same images: left boundary RMSE 0.66 m,
right 0.52 m, width bias **-1.09 m**.

### 5.3 Reading it honestly

**The learned corridor is drivable.** 50 of 54 episodes reached the goal on a corridor whose
geometry came from a camera, with no collisions and no planner change. Ground truth managed 53.
The cost of replacing the ideal corridor with a learned one is **three episodes in 54**, and four
Mode B episodes timed out against one for Mode A.

**The predicted corridor is systematically narrower**, by 1.09 m at the median, inherited directly
from Phase 3B's drivable recall of 0.891. Missed road pixels can only shrink a corridor. That is
the conservative direction, but it is a bias rather than a safety margin, and on a 5.3 m Indian
street it is a real constraint.

**Infeasible planning cycles are four times higher in Mode B** (2378 against 600). Almost all of
them come from the four timed-out episodes, where the vehicle sits with no feasible candidate
until the run ends. This is the most sensitive indicator in the table: the planner notices the
degraded geometry long before the completion count does.

**Perception costs 32.3 ms per frame** against 8.5 ms for reading a mask off disk. Against a 10 Hz
planning budget that is 32%, and planning latency itself was unchanged.

**Tracking quality is unaffected.** Max cross-track error is 0.29 m in Mode B against 0.30 m in
Mode A. Once a corridor clears the gate, the controller does not care where it came from.

### 5.4 Visual evidence

Four panels each: photograph, prediction, corridor with the gate-approved stretch marked, and the
trajectory the existing planner drove. Chosen by measurement, not by eye.

| File | Case |
|---|---|
| `docs/img/p3c_normal.png` | Closest agreement, boundary error 0.23 m, goal reached |
| `docs/img/p3c_difficult.png` | Worst accepted case, still driven |
| `docs/img/p3c_rejected.png` | A narrow village road the gate refused, planner never invoked |

## 6. Limitations, stated plainly

**This is not a camera-only closed-loop drive.** IDD-Lite is 204 unrelated photographs at
320x227 with no ego motion, no calibration and no time ordering. An episode perceives **once** and
then drives the resulting geometry. Perception is genuinely in the loop in the sense that the
planner's road came from a real network running on a real image; it is not in the loop in the
sense of re-perceiving as the vehicle moves. Re-running the network each cycle on the same
photograph would add cost and no information, and inventing a sequence of poses would be exactly
the fake integration this phase forbids. The boundary is drawn here rather than hidden.

**Scale is assumed, not calibrated.** IDD-Lite ships no intrinsics, so the image-to-ground step
uses documented assumptions: camera height 1.4 m, 60 degree horizontal field of view, flat road.
The corridor's shape is learned; its scale comes from those numbers.

**No dynamic objects.** IDD-Lite carries no tracked objects and this phase adds no detector, so
episodes run with zero agents. Tracking, prediction and risk are exercised, but on an empty road.
The eight original scenarios remain the place where those are tested against traffic.

**Boundary error is inherited from Phase 3B** and is the dominant source of any Mode A to Mode B
difference. It is not fixed here; Phase 3C measures what it costs.
