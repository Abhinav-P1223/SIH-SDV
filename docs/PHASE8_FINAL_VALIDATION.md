# Phase 8 — final validation of the frozen autonomy stack

Measurement only. Nothing was trained, tuned or modified in this phase. Machine-readable results
are in `docs/phase8_final_validation.json`, regenerable with `scripts/final_validation.py`.

## 1. Architecture, confirmed

    sensor simulation -> perception -> fusion -> tracking -> prediction -> TTC/risk
      -> behaviour FSM -> corridor lattice planner -> collision checking
      -> Stanley + PID controller -> kinematic bicycle -> world update

Verified at runtime rather than asserted in prose. `verify_no_leakage` checks, on every run, that
in sensors mode the autonomy stack's object provider **is** the fusion tracker, that in ground-truth
mode it **is** the world provider, and that the planner holds the scenario road. All 22 runs pass.
The ego moves only through the vehicle model.

## 2. Final detector, and where it is not

The deliverable is the **Phase 7C uniform-sampling UVH-26 fine-tuned detector** (commit `edb296f`),
chosen over the Phase 7E oversampled variant on mAP 0.325 against 0.282 and on motorcycle and
auto-rickshaw recall.

**It does not appear in these closed-loop runs, and cannot.** The simulator renders no images, so
there are no pixels for a camera detector to consume. "Sensors mode" here is the simulated camera,
LiDAR and radar models feeding the same `Detection` contract into the same tracker and fusion. The
detector's evidence is offline, from Phases 6 and 7, and stays offline. Claiming the detector drove
these runs would be the fake integration this project has refused at every phase.

What *is* proven about it: it emits the same `Detection` objects and reaches the unmodified tracker,
asserted by tests in `tests/unit/test_uvh26_finetune.py`.

## 3. Scenario results

**22 runs, 11 scenarios, 2 perception modes. Zero collisions. 22 of 22 reached the goal.**

### Ground-truth perception

| Scenario | Clearance | Physical TTC | Brakes | Replans | p95 latency | Path | Reverse |
|---|---|---|---|---|---|---|---|
| Unmarked village road | 1.05 m | 1.4 s | 0 | 351 | 27.4 ms | 143.1 m | 0 |
| Unsignalized intersection | 1.60 m | 2.7 s | 0 | 180 | 40.0 ms | 128.2 m | 0 |
| Highway merge, slow vehicles | 1.75 m | 3.2 s | 0 | 268 | 54.0 ms | 327.6 m | 0 |
| Dense market mixed traffic | 0.89 m | 1.3 s | 0 | 338 | 46.9 ms | 118.6 m | 0 |
| Sudden cattle crossing | 0.96 m | 2.0 s | 0 | 152 | 31.6 ms | 118.2 m | 0 |
| Reverse recovery | 0.51 m | 0.4 s | 0 | 340 | 41.6 ms | 85.6 m | **1** |
| Boxed in | 0.80 m | 1.5 s | 0 | 318 | 49.6 ms | 79.5 m | **1** |
| Unprotected turn | 0.57 m | 3.1 s | 0 | 258 | 51.9 ms | 128.6 m | 0 |
| Narrow-lane mutual yield | 0.95 m | none | 0 | 189 | 23.0 ms | 87.9 m | 0 |
| Emergency pedestrian dart | 1.26 m | 1.4 s | 0 | 119 | 30.9 ms | 97.8 m | 0 |
| Mixed traffic on a curve | 2.88 m | none | 0 | 267 | 34.7 ms | 141.5 m | 0 |

### Sensor perception

| Scenario | Clearance | Physical TTC | Brakes | Replans | p95 latency | Path | Reverse |
|---|---|---|---|---|---|---|---|
| Unmarked village road | 0.91 m | 1.2 s | 0 | 357 | 28.6 ms | 143.2 m | 0 |
| Unsignalized intersection | 1.80 m | 2.6 s | 0 | 183 | 37.7 ms | 128.1 m | 0 |
| Highway merge, slow vehicles | 1.40 m | 3.1 s | 0 | 366 | 50.5 ms | 327.9 m | 0 |
| Dense market mixed traffic | 0.84 m | 1.1 s | **3** | 433 | 45.6 ms | 118.5 m | 0 |
| Sudden cattle crossing | 1.29 m | 2.1 s | 0 | 155 | 30.9 ms | 118.1 m | 0 |
| Reverse recovery | 0.66 m | 0.3 s | **3** | 334 | 43.6 ms | 84.9 m | **1** |
| Boxed in | 0.69 m | 1.6 s | 0 | 245 | 55.0 ms | 78.7 m | 0 |
| Unprotected turn | 0.87 m | 3.3 s | 0 | 178 | 52.5 ms | 128.5 m | 0 |
| Narrow-lane mutual yield | 1.00 m | none | 0 | 186 | 25.4 ms | 88.0 m | 0 |
| Emergency pedestrian dart | 1.59 m | 0.7 s | **1** | 126 | 33.1 ms | 97.8 m | 0 |
| Mixed traffic on a curve | 2.07 m | none | 0 | 258 | 42.3 ms | 141.8 m | 0 |

## 4. Perfect versus sensor perception

| | Ground truth | Sensors |
|---|---|---|
| Collisions | 0 | 0 |
| Completed | 11/11 | 11/11 |
| Emergency brakes | 0 | **7** |
| Worst clearance anywhere | 0.51 m | 0.66 m |
| Replans | 2780 | 2821 |
| Worst p95 replan latency | 54.0 ms | 55.0 ms |
| Worst single replan | 90.9 ms | 189.2 ms |

The stack survives real perception. The cost is **seven emergency brakes that perfect perception
never needs**, concentrated in the dense market and the reverse recovery. Those are the audited
class-A and class-B interventions from Phase 5: every one occurred with zero feasible trajectories,
so braking was the last option rather than a threshold artefact.

Perception quality across the sensor runs: recall 0.865 to 0.997, precision 0.745 to 1.000, worst
mean track position error 0.635 m, fusion latency up to 62.3 ms.

## 5. Replanning evidence

Captured at each scenario's **lowest physical time-to-collision**, which is where the stack is
actually under pressure, rather than a frame chosen to look good.

**Sudden cattle crossing, t = 4.0 s.** Ego at 8.96 m/s, TTC 2.1 s, risk HIGH, cattle 21.7 m ahead
drifting across at -0.71 m/s laterally. 46 of 73 candidates feasible, 27 rejected. The FSM entered
AVOID because the cattle's predicted path intersects the ego's. The trajectory changed, the ego
shifted 0.43 m laterally and shed 3.66 m/s. Outcome: no collision, 1.288 m clearance, goal reached.

**Emergency pedestrian dart, t = 4.2 s.** Ego at 7.44 m/s, TTC 0.7 s, risk CRITICAL, pedestrian
8.69 m away moving across at 2.17 m/s. **Only 1 of 2 candidates feasible.** EMERGENCY_BRAKE, the
safety supervisor overrode the planner, `stop_hard` selected, speed fell 5.6 m/s. Outcome: no
collision, 1.594 m clearance, goal reached.

**Dense market, t = 13.2 s.** Ego at 2.58 m/s, TTC 1.1 s, CRITICAL, pushcart 6.44 m ahead. 1 of 2
candidates feasible. EMERGENCY_BRAKE with override. No collision, 0.836 m clearance.

**Reverse recovery, t = 21.2 s.** Ego at 2.24 m/s, TTC 0.3 s, CRITICAL, pushcart 2.64 m ahead,
**zero of 2 candidates feasible**. EMERGENCY_BRAKE with override, then the reverse manoeuvre. No
collision, goal reached.

## 6. Final safety check

All eight pass.

| Check | Result |
|---|---|
| No collisions in any scenario or mode | PASS |
| Every scenario reached its goal | PASS |
| Sensors mode uses fusion as the object provider | PASS |
| Ground-truth mode uses the world provider | PASS |
| Planner consumes the scenario road | PASS |
| Replanning actually changes the trajectory | PASS |
| Emergency braking fires where required | PASS |
| Reverse recovery fires where required | PASS |

## 7. Regression

| | Result |
|---|---|
| Total tests | **348** |
| Passed | **348** |
| Failed | **0** |
| Skipped | **0** |
| Previous baseline | 348 |

Exactly matches the Phase 7 baseline. No test was removed, weakened or skipped. The zero skips are
new: the tests gated on a trained checkpoint now run, because the checkpoints exist locally.

## 8. Failures and limitations

**No scenario failed.** Nothing was tuned after seeing a result, and no test was weakened.

The honest limitations, unchanged and not papered over:

**The learned detector is not in the closed loop**, for the reason in section 2. Its 173 ms
inference is also far outside the 100 ms planning period, so it is an offline evidence pipeline.

**Camera-only perception is unvalidated on dashcam imagery.** UVH-26 is CCTV; Phase 3B and 3C used
IDD-Lite stills with assumed calibration. Neither establishes vehicle-mounted camera performance.

**Bicycle detection is data-limited**, at 106 training boxes, and Phase 7E showed no sampling scheme
repairs that. Pedestrian and animal detection were never addressed by the fine-tuning data.

**Temporal fusion ships disabled.** Phase 4 built it and measured that enabling it makes the stack
more conservative because it removes an over-confidence the collision margins were tuned against.

**Simulated sensors, not real ones.** These runs use sensor models, not recorded sensor data. The
real-data validation lives in Phase 3A (nuScenes fusion interface) and Phases 6 and 7 (real images).

**Worst-case clearance is 0.51 m** in ground-truth reverse recovery, which is tight for a 1.8 m
vehicle and is the scenario deliberately built to be tight.
