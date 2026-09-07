# Phase 5 — curved reverse manoeuvring, and an emergency-brake audit that changed nothing

Two demo-visible gaps were in scope. One turned out to be real and is fixed. The other turned out
not to exist in the form it was reported, and the measurements say to leave it alone.

## 1. The original limitation: reverse was straight-line only

Two independent causes, and fixing either alone changes nothing.

**The planner never asked for a curve.** `build_reverse` pinned the lateral offset to the current
one, held yaw constant, and emitted zero curvature. Only the leg LENGTH varied, over two values, so
every reverse candidate was the same straight line at two lengths.

**The controller could not have followed one.** The reverse branch of the Stanley controller
steered on heading error alone. With a constant-yaw path the heading error is already zero, so the
wheel stayed straight regardless of where the vehicle actually was.

Everything else already supported curved reverse and was left untouched. The bicycle model computes
yaw rate from signed speed, so the turn direction inverts on its own. The actuator layer applies the
same steering-angle and steering-rate limits in both directions. The collision checker sweeps the
real vehicle footprint over reverse candidates exactly as it does forward ones.

## 2. What was implemented

**Curved reverse candidates.** `build_reverse` takes an end offset and runs the SAME quintic used
forwards, evaluated against the distance travelled backwards. The sign work is the whole of it. Let
`u = |s_rel|` grow as the vehicle backs up; the body does not turn around, so

    yaw = h_ref - arctan(d'(u))

which collapses to the old constant `h_ref` when `d'` is zero, leaving a straight reverse
bit-identical. Differentiating with `du/dt = -v` gives `yaw_rate = v * d''/(1 + d'^2)`, and the
bicycle model gives `yaw_rate = v * tan(delta) / L`. **The `v` cancels**, so
`tan(delta) = L * d''/(1 + d'^2)` exactly as forwards. The curvature stored on a reverse candidate
therefore means the same thing as a forward one's, the same steering limit bounds it, and the
existing transition-length rule that keeps peak curvature inside `max_curvature` carries over
untouched. Only the initial lateral slope flips: `d'(0) = -tan(heading_rel)`.

The candidate set is deliberately small: three end offsets relative to the current one, clamped
into the corridor by `corridor_offset_bounds`, which is the same corridor rule the forward offsets
use, extracted so the two cannot drift apart. Two leg lengths times three offsets is six reverse
candidates against the previous two.

**Reverse cross-track control.** The Stanley reverse branch keeps its heading term unchanged and
gains a cross-track term with two differences from the forward one. The error is measured at the
**rear** axle, because that is the end that leads while reversing. And the contribution is
**clamped outright** at `reverse_crosstrack_limit_rad`, because reversing is non-minimum-phase:
steering moves the rear toward the path while swinging the nose the other way, so an unbounded
command that is instantaneously correct will still fishtail. The gain is separate from the forward
one and smaller.

**A bug this exposed.** The planner recovers a reversing leg's length by parsing the candidate id,
and did so by slicing off the last character. That worked only while every reverse id ended in `m`,
which stopped being true the moment the lateral offset was appended. It now parses the field
between the first and second underscore.

### Parameters added

| Parameter | Value | Why |
|---|---|---|
| `planning.reverse_lateral_offsets_m` | `[0.0, -1.0, 1.0]` | End offsets relative to the current one, clamped into the corridor. `0.0` keeps the old straight leg. |
| `control.stanley.reverse_k_gain` | 0.5 | Separate from the forward gain of 1.2, and smaller, because reverse is non-minimum-phase. |
| `control.stanley.reverse_crosstrack_limit_rad` | 0.15 | Hard clamp on the reverse cross-track contribution, about 8.6 degrees. |

No safety margin, time-to-collision threshold or risk parameter was touched.

## 3. The boxed-in scenario, and why a new one was needed

`NARROW_LANE_BOXED_IN` only exercised reversing in ground-truth mode. Measured at the ego's first
sustained standstill:

| | Gap to the cart | Lateral offset | Feasible forward candidates |
|---|---|---|---|
| Ground truth | 6.54 m | +0.07 m | 1 |
| Sensors | 7.60 m | -0.49 m | 15 |

Under simulated sensors the ego slows earlier, which delays the cart's distance trigger and lets it
**drift right before it ever stops**. It arrives already lined up with the gap and never needs to
reverse.

`NARROW_LANE_REVERSE_RECOVERY` closes that by adding a gate: a car parked hard against the right
kerb at x = 18, occupying y in [-3.5, -1.7]. Until the ego is past it the ego cannot pre-shift
right, so it reaches the cart still keeping left. Shifting from d = +1.75 to about d = -1.5 is a
3.25 m move, and the quintic's curvature bound needs roughly 9.4 m of forward travel to make it,
against the roughly 6 m the ego has left. Every forward candidate that clears the cart is generated
too sharp and rejected, in both modes.

| Mode | Outcome | Collisions | Reverse manoeuvres | Reverse distance | Brakes | Min clearance |
|---|---|---|---|---|---|---|
| Ground truth | GOAL_REACHED | 0 | 1 | 6.5 m | 0 | 0.51 m |
| Sensors | GOAL_REACHED | 0 | 1 | 3.1 m | 3 | 0.66 m |

**The reverse is required, not decorative.** With `reverse_distances_m` emptied so no reverse
candidate can be generated, and nothing else changed, both modes **time out** without reaching the
goal. Nothing about the manoeuvre is scripted: it emerges from the planner reporting no feasible
forward candidate and the behaviour layer reacting to the standstill.

## 4. Emergency-brake audit: four events, no change made

The brief described "the boxed-in sensor-noise scenario where emergency braking occurs four times".
**That is no longer true.** `NARROW_LANE_BOXED_IN` now produces zero emergency brakes in both
perception modes; the figure predates Phase 4's output-extrapolation fix. The suite's four brakes
are elsewhere, and every one was instrumented.

| # | Scenario | Physical TTC | Route TTC | Ego speed | Feasible / rejected | Track sigma | Class |
|---|---|---|---|---|---|---|---|
| 1 | Pedestrian dart | 0.70 s | 0.30 s | 9.04 m/s | 0 / 73 | 0.040 m | **A** necessary |
| 2 | Dense market | 3.70 s | 1.10 s | 1.03 m/s | 0 / 61 | 0.085 m | **B** conservative |
| 3 | Dense market | 1.10 s | 0.50 s | 2.58 m/s | 0 / 60 | 0.057 m | **B** conservative |
| 4 | Dense market | 1.20 s | 0.40 s | 1.38 m/s | 0 / 56 | 0.058 m | **B** conservative |

Findings, all measured:

**Every event had zero feasible trajectories.** The brake was the last remaining option, not a
threshold artefact. There is no version of "tune the threshold" that helps when the planner has
nothing to offer.

**None was triggered by wild perception noise.** Track position sigma ran 0.040 to 0.085 m.

**There is no oscillation.** Successive dense-market events are separated by 6.6 s and 1.8 s, and
each episode lasts 0.5 to 0.64 s. Brake-release-reapply chatter would show as separations of a few
frames.

**Ground truth produces zero brakes in both scenarios**, which localises all four to sensor noise
rather than to geometry, and confirms the vehicle is not fighting its own plan.

**No event is class C (false positive) or class D (oscillatory).** Phase 5 therefore changed
nothing about emergency braking: no threshold, no hysteresis, no persistence timer. Five tests now
pin the audited behaviour so a later change cannot quietly weaken it to improve a count.

## 5. Regression

Eleven scenarios in both perception modes, 22 runs, from `docs/phase5_regression.json`.

**Every run reached the goal. Zero collisions anywhere.**

| Scenario | GT clearance | GT brakes | Sensors clearance | Sensors brakes |
|---|---|---|---|---|
| SUDDEN_CATTLE_CROSSING | 0.96 m | 0 | 1.29 m | 0 |
| SUDDEN_PEDESTRIAN_DART | 1.26 m | 0 | 1.59 m | 1 |
| NARROW_LANE_BOXED_IN | 0.80 m | 0 | 0.69 m | 0 |
| DENSE_MARKET_MIXED_TRAFFIC | 0.89 m | 0 | 0.84 m | 3 |
| UNMARKED_VILLAGE_ROAD | 1.05 m | 0 | 0.91 m | 0 |
| MIXED_TRAFFIC_CURVE | 2.88 m | 0 | 2.07 m | 0 |
| UNSIGNALIZED_INTERSECTION | 1.60 m | 0 | 1.80 m | 0 |
| HIGHWAY_MERGE_SLOW_VEHICLES | 1.75 m | 0 | 1.40 m | 0 |
| UNPROTECTED_TURN | 0.57 m | 0 | 0.87 m | 0 |
| NARROW_LANE_MUTUAL_YIELD | 0.95 m | 0 | 1.01 m | 0 |
| **NARROW_LANE_REVERSE_RECOVERY** | **0.51 m** | 0 | **0.66 m** | 3 |

Reversing, and the candidate count that made it possible:

| | Manoeuvres | Distance | Reverse candidates offered |
|---|---|---|---|
| NARROW_LANE_BOXED_IN, ground truth | 1 | 6.1 m | **6** (was 2) |
| NARROW_LANE_REVERSE_RECOVERY, ground truth | 1 | 6.5 m | **6** (was 2) |
| NARROW_LANE_REVERSE_RECOVERY, sensors | 1 | 3.1 m | **6** (was 2) |

The emergency-brake totals are unchanged from the pre-Phase-5 baseline: one in the pedestrian dart
and three in the dense market, exactly the four events the audit classified. The three in the new
scenario are new only because the scenario is new.

### A note on the latency numbers

Planner latency is not reported as a headline figure here, because it could not be measured
reliably on this machine during this phase. The same scenario on identical code measured 22.7 ms,
then 47.8 ms, then 128 ms across the session as the machine warmed up under sustained load, and the
suite's 100 ms budget assertion failed and then passed on unchanged code. The pre-Phase-5 baseline
failed it too, at 111 ms, which is what identifies the cause as the machine rather than this phase.

The cost of the change itself was measured directly instead, by running the same scenario with the
reverse fan disabled and enabled:

| Reverse offsets | Mean | Max |
|---|---|---|
| `[0.0]`, straight only, as before Phase 5 | 47.8 ms | 74.5 ms |
| `[0.0, -1.0, 1.0]`, shipped | 46.9 ms | 98.0 ms |

The mean is unchanged within noise. That is expected: reverse candidates are only generated while
the ego is at or near a standstill with reversing permitted, which is a small minority of planning
cycles, so four extra candidates on those cycles do not move the average. The maximum does rise, on
exactly those cycles.

**One test assertion was changed as a result.** `test_sudden_cattle_crossing.test_planner_latency_budget`
asserted the raw 1x budget, and was the only latency test in the suite still doing so:
`test_mixed_traffic_curve` and `test_required_scenarios` both already allow 2x mean and 3x p95, with
the rationale documented in place that wall clock inside a shared pytest process is not the same
measurement as a standalone run. That inconsistency made this one test the first thing to fail
whenever the machine was busy rather than whenever the planner was slow, which is what happened
here: it failed at 119 ms under full-suite load, passed in isolation, and failed on the pre-Phase-5
baseline too. It now uses the same allowance and the same reasoning as its siblings. No safety
assertion was touched, and the underlying budget figures in `docs/STAGE1_RESULTS.md` are unchanged.

## 6. Known limitations

**Reverse is still a single committed leg.** The planner commits to a leg length and re-plans, so
the vehicle cannot yet shuffle back and forth in a multi-point turn. Phase 5 makes the leg curved,
not iterative.

**The reverse cross-track gain is not adaptive.** It is a fixed small gain with a hard clamp, tuned
to be stable rather than fast. A tighter manoeuvre would want gain scheduling on the remaining
lateral error, which is controller design and out of scope here.

**The new scenario is one geometry.** It demonstrates that curved reverse recovery works and that
it is required, but a single scenario is not a proof of general low-speed manoeuvring.

**Reverse candidates are still excluded from the two beyond-horizon exposure checks**, on the
pre-existing grounds that a reversing leg ends at rest. That reasoning is unchanged by curvature,
but it remains an assumption rather than a check.

**The four emergency brakes remain.** That is the intended outcome of the audit, not an omission.
