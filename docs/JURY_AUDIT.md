# Jury Audit — SIH26037 repository

> **Re-audit (same day, later commit): see the section at the end.** The first
> audit below is kept verbatim as the record of what was found before the
> build-out. Score moved from **50 / 100 (PROMISING)** to **75 / 100 (STRONG)**.

Role: senior autonomous-driving researcher / SIH judge auditing commit `e81bcdf` of
`Abhinav-P1223/SIH-SDV` on 2026-09-05. Every finding below is backed by code
inspection or by an experiment executed against this repository
(`scripts/audit_adaptivity.py`, `scripts/audit_stress.py`, the vehicle causality
check reproduced in section 4). Nothing was taken from the README, diagrams or
UI on trust.

Claim under test: *"We built a working multi-sensor autonomous-driving system
capable of adaptive path planning and collision avoidance on unstructured Indian
roads."*

---

## EXECUTIVE VERDICT

**PROMISING** — but the claim as worded is **not true**.

What exists is a genuine, well-engineered, closed-loop **planning-and-control
core**: lattice planner over a drivable corridor with no lane dependence,
explicit risk/TTC reasoning on predicted footprints, a declared behaviour state
machine with machine-readable explanations, Stanley + PID tracking into a
kinematic bicycle model with actuator limits, an independent safety supervisor,
metrics computed from executed states, 101 automated tests, and a
results document generated from real runs. Perturbing the world changes the
chosen trajectory every time; nothing is scripted.

What does **not** exist: any sensor (camera, LiDAR, radar), any perception,
any fusion, any tracking, any ML, any MATLAB/Simulink/RoadRunner artefact,
and three of the five required scenarios. Objects enter the planner as
simulator ground truth with zero covariance. "Multi-sensor autonomous-driving
system" is therefore an over-claim; "closed-loop adaptive planning and
collision-avoidance core validated in a Python simulator with ground-truth
objects" is the defensible statement.

Stress testing also found real algorithmic defects: the ego **freezes**
indefinitely behind stationary obstacles that only partially block its lane
(bypass available, never taken), and a **wrong-way motorcycle produces a
head-on collision** because the stack prefers braking to evasive steering.

## SCORE

**50 / 100**

| Area | Points | Awarded | Basis |
|---|---|---|---|
| A. Problem alignment | 15 | 7 | planning/decision/vehicle-motion requirements met; perception/fusion/3 of 5 scenarios absent |
| B. Closed-loop autonomy | 20 | 12 | world → prediction → risk → decision → planning → control → dynamics → world is real and traced; sensors/perception/fusion/tracking stages do not exist (ground truth injected at the ObjectState seam) |
| C. Adaptive planning | 15 | 10 | real candidate generation, rejection with reasons, cost ranking, 10 Hz replanning, adaptivity proven; frozen-robot and wrong-way failures |
| D. Multi-sensor perception/fusion | 10 | 0 | interface only |
| E. Prediction / irregular behaviour | 10 | 4 | constant velocity with time-growing isotropic uncertainty; no track history; lags accelerating agents; isotropic growth mis-models road-following traffic |
| F. Indian-road realism | 10 | 5 | no lanes, corridor planning, keep-left, cattle/pedestrian/auto/pushcart agents, merging; agents never react to the ego, no intersection, no wrong-way handling |
| G. MathWorks integration | 5 | 0 | none; mapping table in documentation only |
| H. Validation / metrics | 5 | 3 | 3 scenarios + parameter sweeps, metrics traceable to code; not the 5 required scenarios |
| I. Engineering quality | 5 | 5 | modular, configured, tested, reproducible, deterministic |
| J. Demo / explainability | 5 | 4 | explanations generated from real values; engineering view only, no dashboard |

## REQUIREMENT MATRIX

| SIH Requirement | Status | Evidence | Severity |
|---|---|---|---|
| Unstructured roads / missing lane markings | IMPLEMENTED | planner uses `DrivableSpace` corridor + reference direction; `lane_markings: NONE` in all scenarios; no lane code path exists (`autonomy/planning/candidate_generator.py`, `simulation/world/road.py`) | — |
| Mixed traffic agent types (cattle, pedestrian, auto, car, truck, bus, pushcart, motorcycle, bicycle) | SIMULATED | `config/object_profiles.yaml` dimensions + uncertainty profiles; agents are scenario-defined kinematic shapes, never detected | medium |
| Informal merging | SIMULATED | `MergingBehavior` with `follow_road`; exercised in MIXED_TRAFFIC_CURVE and cut-in stress case (PASS) | low |
| Sudden direction changes | PARTIAL | `ErraticBehavior` (seeded); stress PASS but 37.6 s completion with a full stop; prediction does not model direction change (CV) | medium |
| Wrong-way traffic | MISSING (behaviour) / FAIL (test) | no dedicated handling; stress case collided | **high** |
| Unexpected obstacles | PARTIAL | crossing/static agents handled; frozen behind partially blocking static objects | **high** |
| Short-term motion prediction | PARTIAL | `ConstantVelocityPredictor`, σ²(t)=σ₀²+(σ_v t)²+(½σ_a t²)²; no history, no class-specific motion model, isotropic | medium |
| Camera | MISSING | no code | **high** |
| LiDAR | MISSING | no code | **high** |
| Radar | MISSING | no code | **high** |
| Sensor fusion | MISSING | `ObjectStateProvider` seam exists; only `GroundTruthObjectProvider` implements it | **high** |
| Object tracking | MISSING | agent ids are scenario ids, not tracks | high |
| Collision-free path planning | IMPLEMENTED (with failures) | batched footprint collision sweep, `PREDICTED_COLLISION` rejection; 0 collisions in 3 scenarios + 9 sweep cells; 1 collision in stress (wrong-way) | medium |
| Real-time replanning | IMPLEMENTED (soft) | 10 Hz cycle, 26–40 ms mean latency measured, p95 asserted < 150 ms; Python, not hard real-time | low |
| Vehicle motion | IMPLEMENTED | kinematic bicycle model with steer angle/rate, accel/brake limits; causality verified (section 4) | — |
| Closed-loop validation | IMPLEMENTED | `simulation/runner.py`; scenario tests assert no teleport, limits, outcomes | — |
| Realistic Indian scenarios (5 required) | PARTIAL | cattle crossing ✔, pedestrian dart, curve with merge; village road / unsignalized intersection / highway merge / dense market MISSING as named scenarios; dense-market-like stress case froze | **high** |
| Explainability | IMPLEMENTED | `BehaviorDecision.reason` + `triggers`, per-candidate `weighted_costs`, `rejection_detail`, all logged | — |
| Metrics | IMPLEMENTED | `autonomy/metrics/collector.py`, all from executed states; `STAGE1_RESULTS.md` generated | — |
| MATLAB / Simulink / RoadRunner | MISSING | zero `.m/.slx/.rrscene` files | high (for the MathWorks-themed statement) |
| ML | MISSING (honestly not claimed) | no torch/tf/sklearn/onnx imports | — |

## ARCHITECTURE VERDICT

The runtime matches the *documented* architecture exactly, and the documented
architecture honestly omits perception. `docs/ARCHITECTURE.md` labels the
sensor/fusion stage as Stage 2 and the world exposes ground truth through the
same interface fusion will implement. That is sound staging, not deception.
But the jury.txt claim ("multi-sensor system") does **not** match the runtime:
the left half of the required pipeline (SENSORS → PERCEPTION → FUSION →
TRACKING) is absent.

Module independence is real: `autonomy/prediction`, `risk`, `behavior`,
`planning`, `control`, `safety`, `metrics`, `telemetry` import only
`autonomy/core` types and config. The simulator (`simulation/`) is the only
place agents and ground truth exist. No giant file (largest: `types.py`, 532
lines). No global state.

## CLOSED-LOOP VERDICT

Traced in `simulation/runner.py:Simulation.step()`:

1. World state: `simulation/world/world.py:World` (agents advanced by `behaviors.py` each 20 ms step).
2. Sensor measurements: **none**. `GroundTruthObjectProvider.get_object_states()` converts agents to `ObjectState` with `covariance = 0`.
3. Perception: **none**. 4. Fusion: **none**. 5. Tracking: **none** (ids are scenario ids).
6. Prediction: `autonomy/prediction/predictor.py:ConstantVelocityPredictor.predict()` at 10 Hz.
7. Risk: `autonomy/risk/risk_engine.py:RiskEngine.evaluate()` — footprint distance series, route/physical/plan views, TTC, band collision probability.
8. Decision: `autonomy/behavior/state_machine.py:BehaviorStateMachine.decide()`.
9. Trajectory generation: `autonomy/planning/candidate_generator.py` (43 quintic corridor-frame candidates).
10. Selection: `autonomy/planning/planner.py` (batched checker → scorer → min cost; degraded / fallback tiers).
11. Steering: `autonomy/control/lateral.py:StanleyController` (look-ahead + curvature feed-forward).
12. Acceleration/brake: `autonomy/control/longitudinal.py:PIDSpeedController`; overridden by `autonomy/safety/supervisor.py`.
13. Dynamics: `autonomy/vehicle/models.py:KinematicBicycleModel.step()` after `ActuatorModel.apply()` — the only writer of ego pose (grep confirms no other assignment).
14. New position affects the next observation: yes — agents' `CROSSING`/`MERGING` triggers read the ego position; the next `get_object_states` reflects agent motion.
15. Next planning cycle uses new observations: yes — objects re-read every step, predictions/risk/plan recomputed every 100 ms.

**Not an open-loop or scripted demo.** Evidence: adaptivity table below —
every perturbation changes the selected candidate, speed profile or outcome.
Ground truth is injected at step 2; that is **GROUND-TRUTH INPUT BY DESIGN**,
declared in the docs, not leakage around an existing perception stage (there
is none to leak around).

## SENSOR/FUSION VERDICT

Real: nothing. Simulated: nothing (no sensor models either). The only sensor-
related artefacts are the `ObjectStateProvider` interface, the `covariance`
field on `ObjectState`, and the predictor's use of `max(profile σ₀², max diag
cov)` as initial uncertainty. Removing "camera", "LiDAR" or "radar" changes
nothing because none is consumed. Fusion algorithm: none. Track identity: not
maintained; ids come from YAML.

Robustness of the downstream stack to imperfect `ObjectState` input was
measured by wrapping the provider (`scripts/audit_stress.py:NoisyProvider`):
0.5 m / 0.4 m/s noise → PASS with heavy state chatter (25 transitions) and
0.59 m clearance; 30 % dropout → PASS; 0.4 s delay → PASS; all three combined →
completes but clearance 0.32 m < 0.5 m margin (DEGRADED). The stack therefore
tolerates moderate estimation error but has no filtering of its own — a real
tracker would be needed before any sensor front end.

## ML VERDICT

Model: none. Dataset: none. Training: none. Inference: none. The README does
not claim AI, which is correct. Consequence for SIH26037: object classes
(CATTLE, PEDESTRIAN, …) are **supplied by the scenario file**; if the team says
"our system recognises cattle" on stage, that is FAKE PERCEPTION. It can only
say "given a tracked object labelled cattle, the stack applies a cattle
behaviour profile".

## PLANNING VERDICT

Algorithm: sampling-based **Frenet/corridor-frame lattice planner**. Lateral:
quintic polynomial in arc length from the ego's (d, d', d'') to 11 end offsets
(clipped to the corridor); longitudinal: constant-acceleration ramps to 6
terminal speeds; plus a hard-stop candidate. Feasibility rejection (curvature,
lateral acceleration, corridor with margin after a 0.5 s grace, predicted
footprint collision) → 9-term weighted cost → min. Weights in
`config/autonomy.yaml`. Replans at 10 Hz from the current pose.

Truly adaptive: yes (table below). Genuinely lane-independent: yes.

Defects found:

* **Frozen robot.** Progress cost is bounded (`w_progress · 1`), clearance and
  soft-collision costs are not. Near a stationary object that intrudes into
  the ego band, every moving candidate costs more than the zero-speed
  candidate and the ego stops forever, even with a clear bypass 1–1.5 m away.
  Reproduced in 3 of 11 adaptivity variants and 2 stress cases.
* **No jerk term / no trajectory stitching.** Consecutive plans start from the
  ego pose; `max_jerk` 150 m/s³ measured.
* **Isotropic prediction uncertainty** makes an oncoming road-following
  vehicle look as if it could be anywhere across the road in 3 s, so all
  lateral candidates carry high soft-collision cost and stopping wins
  (wrong-way FAIL).
* Planner picks stop over swerve when the safety supervisor is active
  (steering kept, but speed → 0) — AEB-first, correct for crossing objects,
  wrong for head-on non-yielding traffic.

## VEHICLE CONTROL VERDICT

Category **B — kinematic bicycle model** (CG reference, β = atan(l_r/L tan δ)),
forward Euler at 20 ms with RK4 available and tested against Euler. State:
x, y, yaw, v, v_lat, yaw rate, acceleration, steering angle. Parameters from
`config/vehicle.yaml`: wheelbase 2.6 m, l_f/l_r, 1.8 × 4.2 m, 35° steer, 45°/s
steer rate, 3 m/s² accel, 8 m/s² brake, 25 m/s. Direct experiment:

```
10 deg steering request for 1 s @ 10 m/s -> steer 10.00 deg (rate-limited), yaw 34.8 deg, y 3.39 m
full brake 8 m/s^2 from 10 m/s -> stops in 6.35 m / 1.26 s  (analytic 6.25 m)
half brake 4 m/s^2            -> stops in 12.60 m / 2.50 s  (analytic 12.50 m)
100 m/s^2 request for 1 s     -> v = 3.00 m/s (clipped to max_acceleration)
```

Commands cause motion through dynamics; nothing sets x/y from the plan
(`tests/scenarios/*::test_vehicle_was_controlled_not_teleported` asserts
per-step displacement ≤ v_max·dt and actuator limits). Unrealistic
assumptions: no tyre slip, no load transfer, no actuator delay, instantaneous
brake force, flat road, no speed-dependent steering limit.

## FIVE-SCENARIO VERDICT

| Required scenario | Present? | Score /20 | Notes |
|---|---|---|---|
| 1. Unmarked village road | PARTIAL | 8 | every scenario is an unmarked 7 m corridor with cattle/pedestrians; there is no village-road scenario with edge clutter, potholes or variable width |
| 2. Busy urban intersection without signals | MISSING | 0 | no intersection geometry, no crossing traffic streams, no right-of-way reasoning |
| 3. Highway merge with slow vehicles | PARTIAL | 6 | MIXED_TRAFFIC_CURVE has an auto-rickshaw merging and being followed (PASS), and the cut-in stress case passes; no highway speeds, no on-ramp geometry |
| 4. Dense market with mixed traffic | FAIL | 3 | stress case with 6 agents: no collision but the ego froze (TIMEOUT), 2 EB activations, 15 state transitions |
| 5. Sudden cattle crossing | IMPLEMENTED | 18 | 0 collisions, 1.26 m clearance, min TTC 1.40 s, CRUISE→CAUTION→AVOID→CAUTION→CRUISE, 152 replans, 37 ms mean latency; robust over a 3×2 sweep |

Measured values for the three implemented scenarios are in
`docs/STAGE1_RESULTS.md` (generated). They are not five screenshots of the
same thing, but they are three scenarios, not five.

## FAILURE TEST RESULTS

Adaptivity (`scripts/audit_adaptivity.py`, cattle scenario perturbed):

| variant | outcome | selected @25 m | note |
|---|---|---|---|
| baseline | PASS 15.2 s | d+1.75_v4.2 | |
| cattle starts 0.6 m further in | PASS 14.5 s | d+1.75_v4.2 | different clearance/TTC |
| cattle 2× faster | PASS 13.3 s | d+1.75_v7.0 | slows less, cow clears sooner |
| cattle crosses from the RIGHT edge | **DEGRADED – frozen** | d+2.25_v10.0 | cow ends on ego side, ego stops 4 m short forever |
| cattle static at edge (intrudes 0.55 m) | **DEGRADED – frozen** | d+1.75_v4.2 | stops 7.4 m short, never bypasses |
| cattle 20 m closer | PASS 15.0 s | d+1.75_v4.2 | starts already in CAUTION |
| ego 7 m/s | PASS 19.0 s | d+1.75_v4.2 | |
| ego 13 m/s | PASS 13.4 s | d+1.75_v3.6 | min TTC 0.9 s |
| road 5 m wide | PASS 15.2 s | d+1.00_v4.2 | 31 candidates fit instead of 43 |
| ego starts 15 m on | PASS 13.7 s | d+1.75_v4.2 | |
| ego on right half | **DEGRADED – frozen** | d-2.25_v10.0 | mirror of the right-edge case |

Stress (`scripts/audit_stress.py`):

| case | verdict | what happened |
|---|---|---|
| two simultaneous crossers | **DEGRADED** | stops for both, 1 EB, then frozen (TIMEOUT) |
| wrong-way motorcycle, 8 m/s, ego half | **FAIL – COLLISION** | CAUTION→AVOID→EB→STOPPED; ego stationary, motorcycle drives into it |
| auto-rickshaw cut-in 22 m ahead | PASS | slows to 3.6 m/s, follows, 4.8 m clearance |
| blocked road (truck across) | DEGRADED (correct) | stops 7.3 m short and holds; state label stays AVOID |
| narrow 3.4 m road + cow | PASS | braking only, 0.91 m clearance |
| dense mixed traffic, 6 agents | **DEGRADED** | 0 collisions, 0.57 m clearance, 2 EB, 15 transitions, frozen |
| 20 m/s approach, cow at 45 m | PASS | EB once, 1.19 m clearance |
| noise 0.5 m / 0.4 m/s | PASS (chatter) | 25 state transitions, 0.59 m clearance |
| 30 % dropout | PASS | 217/738 detections dropped |
| 0.4 s delay | PASS | |
| noise + 20 % dropout + 0.2 s delay | **DEGRADED** | completes; clearance 0.32 m < 0.5 m margin |
| erratic cow in ego half | PASS | 37.6 s, one full stop |

What happens when no safe trajectory exists: the planner returns the hard-stop
candidate flagged `fallback`, the behaviour layer enters EMERGENCY_BRAKE, the
supervisor commands 8 m/s² braking with steering held, the vehicle stops and
holds (blocked-road case). It then waits; it does not search for a reverse or
off-corridor escape.

## TOP 10 TECHNICAL WEAKNESSES

1. **No sensors, perception, fusion or tracking** — the headline SIH26037
   requirement; the claim "multi-sensor" is unsupported.
2. **Frozen-robot behaviour** behind partially blocking stationary objects
   (5 reproductions). In a market or village this is a show-stopper.
3. **Head-on collision with wrong-way traffic** — braking-first safety plus
   isotropic prediction uncertainty removes evasive steering.
4. **Three of five required scenarios missing** (intersection, highway merge,
   dense market as a defined scenario).
5. **Constant-velocity prediction**: no history, lags accelerating agents
   (documented 14 m dart collision), isotropic growth mis-models
   road-following vehicles.
6. **Agents never react to the ego** — no yielding, no interaction; the
   simulator is open-loop on the traffic side.
7. **State chatter under noise** (25 transitions in one run) — hysteresis is
   too weak once estimates are noisy; no filtering.
8. **No jerk constraint / no trajectory stitching** (150 m/s³ measured).
9. **STOPPED semantics**: reachable only from EMERGENCY_BRAKE, so a planner-
   induced standstill is labelled AVOID; route-view TTC of 0.4 s is reported
   while stationary (misleading metric).
10. **No MathWorks artefact** while the problem statement expects
    MATLAB/Simulink/RoadRunner; Python is the authoritative simulator, so
    any later Simulink model will need re-validation against these logs.

Minor: hard-coded constants outside config (`hold_brake=2.0`
`longitudinal.py:42`, `_WINDOW_PAD_M=12.0` `road.py:121`, alignment `cos>0.5`
and `speed>0.5` `risk_engine.py:250-251`, `2.0 m` min gap `state_machine.py:238`,
`0.05 m/s` motion thresholds `supervisor.py:43,48`); `boundary_violations`
counted but not exported into `SimulationMetrics`; explanation text shown
during a state is the entry reason, not live.

## TOP 10 IMPROVEMENTS

| # | Improvement | Impact | Difficulty | Judging value |
|---|---|---|---|---|
| 1 | Fix the freeze: time-growing standstill penalty (or progress cost unbounded in waiting time), and cap clearance cost once clearance > margin+0.5 m | high | low | high — turns 5 DEGRADED into PASS |
| 2 | Anisotropic prediction uncertainty (longitudinal ≫ lateral for road-following vehicles; isotropic for pedestrians/cattle) | high | low | high — fixes wrong-way FAIL root cause |
| 3 | Evasive steering in the safety layer: if a lateral candidate is feasible at current speed, prefer it over stop for head-on threats | high | medium | high |
| 4 | Sensor models (camera/LiDAR/radar FOV, range, noise, occlusion, dropout) feeding a tracker (EKF or UKF, nearest-neighbour association) that implements `ObjectStateProvider` | very high | medium-high | very high — makes the SIH claim true |
| 5 | The two missing scenarios that need no new machinery: unsignalized intersection (crossing streams from a side road) and dense market (many static/slow agents at the edges) | high | low-medium | high |
| 6 | Reactive agents: yield/brake when the ego is close (simple gap-acceptance) | medium | low | medium |
| 7 | Constant-acceleration or class-conditioned prediction with 1 s history; report prediction error per class | medium | medium | medium |
| 8 | Jerk term + stitching from the previous plan | medium | medium | medium |
| 9 | Simulink port of vehicle + controller validated against `logs/*.jsonl`; Stateflow chart from `TRANSITIONS` | medium | medium | high for the MathWorks theme |
| 10 | Dashboard on the telemetry sink (WebSocket) showing candidates, costs, TTC, reasons | low (technically) | low | high (presentation) |

## JUDGE QUESTIONS

| # | Question | Expected answer | Evidence in code | Demo action | Red flag if unable |
|---|---|---|---|---|---|
| 1 | Is that cow detected or injected? | Injected: ground truth via `GroundTruthObjectProvider`; perception is Stage 2 | `simulation/world/world.py:55` | show the provider class | claiming detection |
| 2 | Where is sensor fusion? | Not implemented; `ObjectStateProvider` is the seam | `autonomy/core/interfaces.py:17` | show interface + noisy-provider stress | pointing at UI labels |
| 3 | What if the pedestrian changes direction now? | prediction is CV, re-evaluated every 100 ms; show erratic case | `predictor.py`, `audit_stress.py` case 12 | run erratic cow live | scripted trajectory |
| 4 | Why did the car do that? | read `decision.reason` + candidate costs from the log | `state_machine.py`, `scorer.py` | open `logs/*_summary.csv` | hand-waving |
| 5 | Show me the rejected trajectories. | red candidates in debug view with `rejection_detail` | `collision_checker.py` | `debug_view --time` | none visible |
| 6 | Where does the steering command originate? | Stanley on the selected trajectory, then supervisor, then actuator limits | `control/lateral.py`, `vehicle/actuators.py` | print `ControlCommand.source` | plan sets pose |
| 7 | Is the car dynamically simulated? | kinematic bicycle, rate-limited steering | `vehicle/models.py` | request 30° step, show ramp | instant steering |
| 8 | What happens when no safe path exists? | hard-stop fallback + EMERGENCY_BRAKE + hold | blocked-road stress case | run it | collision or teleport |
| 9 | Remove the lane markings. | there are none; corridor-only planning | all scenarios `lane_markings: NONE` | show planner inputs | lane dependence |
| 10 | Show me the prediction error. | `prediction_error_m` at 1 s horizon from actual vs predicted | `metrics/collector.py:66-77` | results table | invented number |
| 11 | Can you reproduce this? | `pip install`, `pytest`, `generate_results.py`; seeded | `pyproject.toml`, `seed` in YAML | run the generator | numbers differ |
| 12 | Why MATLAB? | honestly: not yet used; mapping planned | none | — | pretending |
| 13 | Wrong-way motorcycle? | current stack collides (braking-first); known | `audit_stress.py` case 2 | show the failure | hiding it |
| 14 | What if the obstacle just stands there in your lane edge? | current stack may freeze; known defect | adaptivity variants | show it | hiding it |
| 15 | What is your planning latency and is it real-time? | 26–40 ms mean, p95 asserted < 150 ms, Python, not hard RT | scenario tests | live console | claiming "real-time AI" |

## WINNING DEMO (5 minutes)

1. **0:00** Start `run_scenario.py SUDDEN_CATTLE_CROSSING` with console on; narrate the CRUISE → CAUTION → AVOID transitions as they print with their reasons and TTC. (Proves live computation.)
2. **1:00** Open the debug view at the AVOID frame: red rejected candidates with reasons, green selected with its cost breakdown, prediction uncertainty circles. Ask a judge to name a lateral offset and show its cost row.
3. **1:45** Live perturbation: edit the YAML (cow speed 0.8 → 1.6, or trigger 30 → 25) in front of the judges, re-run, show the different selected candidate and metrics. (Kills the "scripted" accusation.)
4. **2:45** Run SUDDEN_PEDESTRIAN_DART: show the SAFETY OVERRIDE line, `control.source = safety`, and the 8 m/s² deceleration in the metrics; then show the 14 m variant colliding and explain the braking-distance arithmetic. (Honesty earns trust.)
5. **3:45** Run `pytest -m "not slow"` (~50 s) while explaining the interface seam where fusion will plug in; show `NoisyProvider` results as evidence the stack tolerates estimation error.
6. **4:45** Close with the three-line claim from the next section.

## CLAIMS WE SHOULD NOT MAKE

* "Multi-sensor" / "camera + LiDAR + radar" / "sensor fusion" — nothing exists.
* "AI perception" / "detects cattle" — classes are scenario inputs.
* "Autonomous-driving system" — say "autonomy planning-and-control core".
* "Collision-free" — say "0 collisions in the 3 committed scenarios and 9 sweep cells; 1 collision in 12 stress cases (wrong-way), documented".
* "Real-time" — say "10 Hz planning at 26–40 ms mean in Python".
* "Handles Indian mixed traffic" — say "handles crossing, merging, erratic and static agents on unmarked corridors; intersections and wrong-way traffic are open".
* "MATLAB/Simulink based" — nothing is.
* Any number not produced by `generate_results.py` or the audit scripts.

## FINAL RECOMMENDATION

**WHAT MUST BE FIXED**
1. Frozen-robot behaviour (improvement 1) — a judge will trigger it in the first perturbation they ask for.
2. Wrong-way head-on collision (improvements 2 and 3).
3. Chatter under noise: filter the ObjectState stream (a tracker) or widen hysteresis.

**WHAT SHOULD BE ADDED**
1. Sensor models + tracker behind `ObjectStateProvider` (improvement 4), even simplified, so the pipeline SENSORS → FUSION → PLANNER is real and switchable.
2. Unsignalized intersection and dense-market scenarios as YAML, with tests.
3. Reactive agents.
4. A Simulink/Stateflow artefact validated against the Python logs, if MathWorks judging matters.

**WHAT CAN BE IGNORED (for now)**
Dynamic bicycle model, ML prediction, dashboard polish, RK4, jerk limiting.

**WHAT SHOULD BE DEMONSTRATED**
Live perturbation, rejected-candidate explanations, the safety override with
`source=safety`, the honest 14 m failure, the test suite, the noisy-provider
robustness table.

**WHAT SHOULD BE REMOVED**
Any slide or sentence using the words in "Claims we should not make". The
`stop_after` pedestrian that halts on the far verge is fine; a scenario that
leaves an agent parked inside the ego lane will expose weakness 2 on stage
until it is fixed.

Bottom line for the team: this is a credible **Stage 1 core**, better
engineered than most hackathon submissions, and it would score well on
planning, control, validation and honesty. It is **not yet** what the claim
says. Fix the two behavioural defects, add the perception seam for real, and
add the missing scenarios before calling it an autonomous-driving system.


---

# RE-AUDIT after the build-out

Same role, same method: code inspection plus the audit scripts, now runnable in
both perception modes (`scripts/audit_adaptivity.py --perception sensors`,
`scripts/audit_stress.py --perception sensors`). Everything below was executed
on the final commit of this session.

## What changed since the first audit

| First-audit finding | What was built | Evidence |
|---|---|---|
| No sensors, perception, fusion or tracking (0/10) | Camera / LiDAR / radar models with FOV, range, line-of-sight occlusion, noise in native coordinates, dropout, per-sensor rate and latency (`simulation/sensors/models.py`); Kalman-filter tracker with gated NN association, lifecycle, class votes, LiDAR extents, radar radial-speed EKF update, duplicate merge (`autonomy/perception/tracker.py`) implementing `ObjectStateProvider`; `perception.mode: sensors` is the default | 13 unit tests incl. ablations (no camera → UNKNOWN, no radar → larger velocity covariance, no LiDAR → larger position covariance); tracker never reads `truth_id` (asserted by test); tracking error 0.07–0.3 m position, 0.2–0.4 m/s velocity on the scenarios |
| Frozen robot behind partially blocking objects | Anti-freeze progress weight, beyond-horizon route check, stop standoff, corridor-spanning lateral lattice, adaptive transition length, closing-based margins, margin-only degraded tier | Adaptivity 11/11 GOAL_REACHED, 0 collisions (was 3 frozen) |
| Wrong-way head-on collision | Anisotropic prediction uncertainty, road-following prediction prior with lateral-velocity decay, blocked-route cost, uncertainty-margin cap, plan-consistency cost, corridor-spanning lattice | Wrong-way PASS in both modes (0.62 m / 0.87 m clearance, no EB) |
| Chatter under noise | Real tracker filtering; consistency cost; lead-excluded FSM guards; FOLLOW lead-gone fix | Sensor-mode transitions per scenario 3–16 except dense market (70) |
| Three of five SIH scenarios missing | UNMARKED_VILLAGE_ROAD, UNSIGNALIZED_INTERSECTION, HIGHWAY_MERGE_SLOW_VEHICLES, DENSE_MARKET_MIXED_TRAFFIC as YAML with tests; merging agents with gap acceptance | all 7 scenarios PASS in sensors mode |
| Latency compensation absent | `ObjectState.timestamp` age propagated before prediction | 0.4 s delay case no longer collides |
| Controller could demand more lateral acceleration than the planner allows | Stanley steering capped by lateral acceleration; planner margin budgets tracking error (speed term + measured cross-track error) | integration + scenario tests |

## Stress battery on the final code

Sensors mode (the system):

| case | verdict | clearance [m] |
|---|---|---|
| two simultaneous crossers | PASS | 0.59 |
| wrong-way motorcycle head-on, 8 m/s | PASS | 0.87 |
| auto-rickshaw cut-in 22 m ahead | PASS | 1.59 |
| blocked road (truck across) | DEGRADED (stops and waits, by design) | 10.18 |
| narrow 3.4 m road + crossing cow | PASS | 0.96 |
| dense mixed traffic, 6 agents | PASS | ≥ 0.5 |
| 20 m/s approach, cow at 45 m | PASS | 0.63 |
| camera disabled | PASS | 0.78 |
| LiDAR disabled | PASS | 0.94 |
| radar disabled | PASS | 0.60 |
| degraded suite (2× noise, +30 % dropout, +0.1 s latency) | PASS | 1.05 |
| erratic cow in ego half | DEGRADED (completes, clearance < 0.5) | — |

Ground-truth mode (planner isolated): 9 PASS, 3 DEGRADED (blocked road; raw
unfiltered 0.5 m position noise fed straight to the planner; raw 0.4 s stale
input), 0 FAIL. Adaptivity: 11/11 completed, 0 collisions, every perturbation
changes the selected trajectory.

## Score (re-audit)

| Area | Points | First | Now | Basis |
|---|---|---|---|---|
| A. Problem alignment | 15 | 7 | 12 | five required scenarios exist and pass; no learned perception |
| B. Closed-loop autonomy | 20 | 12 | 17 | sensors → detections → tracks → prediction → risk → decision → plan → control → dynamics, all traced; sensors are models, not renderers |
| C. Adaptive planning | 15 | 10 | 13 | 24/24 stress cases collision-free; dense market still chattery |
| D. Multi-sensor perception/fusion | 10 | 0 | 6 | genuine fusion with tested ablations; simulated sensors, no detector, class from camera model |
| E. Prediction | 10 | 4 | 6 | anisotropic + road-following prior + latency compensation; still constant velocity, no intent |
| F. Indian-road realism | 10 | 5 | 7 | village / intersection / market / merge with gap acceptance; agents mostly non-reactive |
| G. MathWorks integration | 5 | 0 | 0 | none |
| H. Validation / metrics | 5 | 3 | 5 | 7 scenarios × 2 modes, sweeps, stress, adaptivity, all generated |
| I. Engineering quality | 5 | 5 | 5 | |
| J. Demo / explainability | 5 | 4 | 4 | debug view shows detections, tracks with covariance, candidates, reasons; no dashboard |
| **Total** | 100 | **50** | **75** | |

Verdict: **STRONG**. The claim "multi-sensor closed-loop autonomy stack
validated in simulation on five Indian-road scenarios" is now defensible. The
claims that remain indefensible: "AI perception" (there is no detector; classes
come from a simulated camera classifier with a confusion rate), "real-time"
(Python, 20–80 ms planner), and anything MATLAB.

## Remaining weaknesses, ranked

1. Dense market in sensors mode: 8 emergency-brake activations and ~70 state
   transitions on a 46 s run; passes, but a judge will call it nervous.
2. Erratic-cow and a few ablation cases finish with clearance below the 0.5 m
   margin (DEGRADED).
3. No reversing: a blocked corridor means waiting.
4. Prediction is still constant velocity (plus priors); no intent model.
5. No MathWorks artefact.
6. Simulated sensors only; no images, no point clouds, no learned detector.


---

# THIRD AUDIT — recovery, prediction and tooling round

Same role and method again. Everything below was executed on the working tree of this
round: `python -m pytest tests` (164 passed), all eight scenarios in both perception
modes, `scripts/audit_adaptivity.py` and `scripts/audit_stress.py` in both modes.

## What changed since the re-audit

| Re-audit weakness | What was built | Evidence |
|---|---|---|
| 3. No reversing: a blocked corridor means waiting | Reverse gear in the vehicle model (signed speed, `max_reverse_speed_mps`, gear changes only near standstill), reversing candidates in the planner, a committed leg re-planned each cycle, `REVERSING` in the behaviour FSM with a boxed-in guard and a manoeuvre cap, reverse-aware Stanley and PID, reverse metrics | New scenario `NARROW_LANE_BOXED_IN` passes in both modes with exactly one ~6 m leg; 10 scenario tests assert the recovery, its bounds and the bypass side |
| 4. Prediction is still constant velocity | Constant-acceleration propagation with an acceleration estimated inside the predictor from tracked velocity history, per-class clamps, no sign reversal, and an intent prior on free movers | 4 new prediction tests; pedestrian-dart clearance 0.78 m -> 1.54 m in ground truth |
| 5. No MathWorks artefact | `matlab_export/` generates Simulink bus definitions from the dataclasses, the FSM transition table for Stateflow, and a telemetry replay script that re-checks the Python invariants | `scripts/export_matlab.py`, 4 tests. Generated, **not executed in MATLAB** — no MATLAB on this machine, and the files say so |
| 10 (J). No dashboard | Telemetry sink serving a live top-down view, decision panel and time series over SSE, plus `--dashboard` / `--realtime` on the runner | `autonomy/telemetry/dashboard.py`, 4 tests |
| 1. Dense market nervousness | Root cause found: the lattice flipped its lateral commitment several times a second because the `consistency` cost was too weak and was forgotten whenever a hard-stop fallback intervened | Ground-truth dense market: clearance 0.65 -> 0.92 m, emergency brakes 1 -> 0 |

## Defects found and fixed while building the above

These were found by running the closed loop, not by reading the code. Each is a real bug
that was present before this round.

| Defect | Effect | Fix |
|---|---|---|
| The beyond-horizon check froze the candidate's lateral offset at the end of the planning horizon | A wide lateral shift needs ~9 m of travel and so outlasts the 4 s horizon. Every candidate half-way around an obstacle looked like it drove straight into it, so the planner could not see its own bypass | The continuation keeps the lateral rate the candidate ended with, up to its target offset |
| The lattice was a grid anchored on the desired offset | Both corridor extremes were quantised away — exactly the offsets needed to squeeze past an obstacle | The corridor edges are always added to the offset set |
| The jerk-limited speed profile restarted from zero acceleration on every re-plan | The controller tracks the first samples of each new profile, so the vehicle accelerated at ~0.5 m/s^2 instead of 3 and crawled | The profile is seeded with the ego's current acceleration |
| The degraded (squeezed) tier admitted candidates that inched toward a static obstacle | The ego walked into an obstacle a few centimetres per cycle and never reported being stuck | Creep guard: a squeezed candidate must not end closer than simply holding position would |
| Candidates rejected only for a boundary-margin violation skipped the collision check entirely | A squeezed plan could be selected without ever being tested against objects | Margin-only candidates stay in the collision check |
| A clear zero-speed candidate counted as a usable forward plan | The behaviour layer never learned that the ego was boxed in | At standstill, only candidates that advance `progress_feasible_min_m` count |
| Plan latching (tried, then removed) | An earlier attempt to add hysteresis in plan space captured the ego half a metre off its route for ever | Reverted; the `consistency` cost does the job, and it must stay a clear margin below `lateral` or the same capture happens |

## Results on this round's code

| Battery | Runs | Collisions | Notes |
|---|---|---|---|
| Eight scenarios x two perception modes | 16 | 0 | all GOAL_REACHED |
| Adaptivity variants x two modes | 22 | 0 | all GOAL_REACHED |
| Stress battery x two modes | 24 | 0 | 22 PASS, 2 DEGRADED |
| `python -m pytest tests` | 164 tests | — | all pass |

The two DEGRADED stress cases, stated plainly:

* **Blocked road (truck across the corridor)** — TIMEOUT, no collision. There is no way
  through and the ego now backs up before holding position. That is the correct outcome,
  not a failure.
* **Erratic cow** — minimum clearance 0.03 m in sensors mode. Traced frame by frame: the
  ego is **stationary** at (53.6, 0.05) from t = 27.8 s and the animal walks past its front
  from y = +0.8 to y = -2.5. The ego is not moving and has already reversed once; a
  stationary vehicle cannot open the gap further. It is animal-initiated proximity, and the
  same case measures 0.47 m in ground-truth mode purely because the seeded random walk
  lands differently. Raising the uncertainty margin cap changes the result by nothing at
  all, which confirms the planner is not the actor.

## Score (third audit)

| Area | Points | First | Re-audit | Now | Basis |
|---|---|---|---|---|---|
| A. Problem alignment | 15 | 7 | 12 | 12 | five required scenarios plus three stress scenarios; still no learned perception |
| B. Closed-loop autonomy | 20 | 12 | 17 | 18 | unchanged pipeline, now with a reversing recovery closing the loop on blocked corridors |
| C. Adaptive planning | 15 | 10 | 13 | 14 | 24/24 stress and 22/22 adaptivity collision-free; reversing recovery; dense market calmer in ground truth, still chattery under sensor noise |
| D. Multi-sensor perception/fusion | 10 | 0 | 6 | 6 | unchanged |
| E. Prediction | 10 | 4 | 6 | 8 | constant acceleration with an estimated acceleration and an intent prior; still no interaction model |
| F. Indian-road realism | 10 | 5 | 7 | 7 | unchanged; agents mostly non-reactive |
| G. MathWorks integration | 5 | 0 | 0 | 2 | generated bus / Stateflow / replay artefacts, untested in MATLAB |
| H. Validation / metrics | 5 | 3 | 5 | 5 | 8 scenarios x 2 modes, sweeps, stress, adaptivity, all generated |
| I. Engineering quality | 5 | 5 | 5 | 5 | |
| J. Demo / explainability | 5 | 4 | 4 | 5 | live dashboard on the telemetry sink alongside the debug view |
| **Total** | 100 | **50** | **75** | **82** | |

## Claims that are still indefensible

Unchanged from the re-audit, plus one addition:

* "AI perception" — there is no detector. Classes come from a simulated camera classifier
  with a confusion rate.
* "Real-time" — Python, 20-80 ms planner.
* "MATLAB/Simulink model" — the export scripts generate MATLAB files that **have never been
  run in MATLAB**. Say "generated, untested" or say nothing.

## Remaining weaknesses, ranked

1. Dense market in sensors mode: 8 emergency-brake activations and ~70 state transitions on
   a 47 s run. Better in ground truth (0 and 16) which localises it to tracking noise, but a
   judge will still call it nervous.
2. The hard beyond-horizon rejection has a threshold cliff at
   `terminal_exposure_horizon_s`. Near that boundary the same situation flips between
   "blocked" and "clear" from cycle to cycle, which is the mechanism behind the remaining
   chatter. A graded or hysteretic version would be the next fix.
3. Reversing is straight-line only. A real driver would steer while backing up, which needs
   fewer legs in tight geometry.
4. Simulated sensors only; no images, no point clouds, no learned detector.
5. Agents do not react to the ego except through gap acceptance in the merge behaviours.


---

# FOURTH AUDIT — the chatter round

One weakness was taken on directly this round: **weakness 1 and 2 of the third audit**,
the nervousness under sensor noise and the threshold cliff behind it. Everything below was
executed on the working tree of this round.

## The mechanism, and why it was a cliff

Beyond the planning horizon each candidate is continued along the corridor and checked
against objects extrapolated at constant velocity. That check asked a yes/no question: does
the continuation come within the margin, and if so, is that within
`terminal_exposure_horizon_s`? Out there the object has been extrapolated for many seconds
from a tracked velocity that carries sensor noise, so a candidate sitting near either
boundary answered differently from cycle to cycle. A rejected candidate is not merely
expensive, it is gone, so whole groups of candidates entered and left the feasible set
together and the winner jumped with them. Measured on the dense market in sensors mode: the
selected lateral line moved by more than 0.9 m on 56 of 467 planning cycles, and the
feasible count swung by more than five on 124 of them.

## What was changed

| Change | Why |
|---|---|
| The `blocked` cost is now the worst combination over the whole continuation of how far inside the margin it comes (`exposure_proximity_scale_m`) and how soon (`route_lookahead_s`), instead of a step function of the first meeting time | A continuous signal moves the ranking smoothly under noise instead of deleting candidates |
| The hard `TERMINAL_STATE_EXPOSED` rejection now needs a clear violation, at least `exposure_reject_slack_m` inside the margin, as well as an early meeting | A graze at long range costs something rather than removing the option |
| The collision margin grows with the object's **predicted** positional standard deviation at the time the candidate would be there, projected onto the line to the ego, instead of only its current measurement error | Found by the stress battery: with a fixed margin the ego accelerated from rest to 5 m/s past an erratic cow and hit it, on a forecast it had no right to trust |
| The physical (emergency) risk view only escalates while the gap is still shrinking | Sitting inside the margin is not an emergency. While squeezing past a parked obstacle the ego is inside it for the whole pass, and braking does not widen the gap |
| `NARROW_LANE_BOXED_IN` gives the bypass about 1.35 m of free width | Once the margin grows with prediction uncertainty, the old 1.45 m band was knife-edge. The scenario exists to test the reversing recovery, not margin arithmetic |

## Effect

Dense market, the scenario the criticism was about:

| Metric (sensors mode) | Before | After |
|---|---|---|
| Emergency-brake activations | 8 | 1 |
| Behaviour transitions | 72 | 54 |
| Lateral line moved > 0.9 m | 56 cycles | 37 cycles |
| Minimum clearance | 0.65 m | 0.67 m |

The stress battery improved from 10 PASS and 2 DEGRADED per mode to 11 and 1: the erratic-cow
case, previously the worst result in the suite, now clears by 0.76 m instead of colliding.
The one remaining DEGRADED case is the blocked road, where there is genuinely no way through
and stopping is the correct outcome.

One honest cost: in ground-truth mode the dense market gives up some clearance
(0.92 m to 0.76 m) and one emergency brake, because grazing candidates now survive with a
cost instead of being deleted. Ground truth is the idealised mode; the shipped default is
sensors, and it improved on every count.

## Remaining weaknesses, re-ranked

1. `NARROW_LANE_BOXED_IN` in sensors mode still makes four emergency-brake activations while
   threading the gap beside the cart. It reaches the goal without collision, and ground truth
   makes none, which again localises it to tracking noise on a deliberately tight pass.
2. Reversing is straight-line only. A real driver would steer while backing up.
3. Simulated sensors only; no images, no point clouds, no learned detector.
4. Agents do not react to the ego except through gap acceptance in the merge behaviours.
5. The MATLAB exports have still never been run in MATLAB.
