# Technical architecture

System frozen at `2ccd232a8b`. Numbers generated from
`docs/FINAL_SYSTEM_RESULTS.json`.

> **What was validated, and how.** The learned detector was validated independently on held-out
> UVH-26 Indian-scene imagery, and its output contract was verified against the unmodified tracking
> pipeline. The autonomy stack was validated independently, end to end, through simulated
> multimodal sensors. These are two separate validations. The simulator renders no image frames, so
> **no detector consumed camera pixels during the closed-loop runs**, and this package does not
> claim end-to-end learned camera perception in closed loop.

## Runtime data flow, one 50 Hz simulation step

1. **World update.** Agents advance. Interactive agents see only the ego's pose, speed and heading,
   never its future plan.
2. **Sensing.** Camera, LiDAR and radar models emit `Detection` objects in the world frame, each
   carrying its own timestamp, a positional covariance derived from the *measured* range and
   bearing, and per-sensor latency.
3. **Fusion and tracking.** A constant-velocity Kalman filter with an extended-Kalman radial-speed
   update from radar. Mahalanobis gating, greedy nearest-neighbour association, time-based track
   deletion, duplicate merging.
4. **Prediction.** Constant-acceleration with per-class intent priors, producing time-indexed
   positions with growing covariance.
5. **Risk.** Two time-to-collision views: a route view along the corridor at desired speed, and a
   physical view at the ego's actual heading and speed.
6. **Behaviour.** A declarative state machine over 7 states with dual hysteresis, asymmetric dwell
   times and split entry and exit thresholds.
7. **Planning.** A corridor-frame lattice: quintic lateral profiles across feasible offsets,
   jerk-limited longitudinal profiles, plus reverse candidates when the vehicle is boxed in.
8. **Collision checking.** Swept oriented-box footprints against predicted object footprints, with
   margins that grow with each object's *predicted* position uncertainty at the time of encounter.
9. **Control.** Stanley lateral with a look-ahead and curvature feed-forward, PID longitudinal.
10. **Safety supervisor.** Can override the planner outright and command a hard stop.
11. **Vehicle.** Kinematic bicycle with signed speed, a reverse gear, actuator angle and rate
    saturation.

## Interfaces that make the parts swappable

**`Detection`** is the perception boundary. The simulated sensors emit it, and so does the learned
camera detector. The tracker cannot tell them apart, which is why the detector needed no change
anywhere downstream to be integrated.

**`RoadModel` / `DrivableSpace`** is the map boundary. Scenario YAML produces one; so does the
learned drivable-space segmentation. The planner cannot tell them apart.

**`ObjectStateProvider`** is the fusion boundary. Ground-truth mode supplies the world provider,
sensor mode supplies the fusion tracker. Verified at runtime on every validation run.

## Measured behaviour

| | Value |
|---|---|
| Scenarios | 11 |
| Runs | 22 |
| Completion | 22/22 |
| Collisions | 0 |
| Worst p95 replanning latency | 55.0 ms against a 100 ms budget |
| Worst single replan | 189.2 ms |
| Total replans across all runs | 5601 |

## Design decisions worth defending

**Covariance is computed from what the sensor measured**, never from the true range. Deriving it
from ground truth handed the filter an oracle it could never have in reality.

**Collision margin grows with predicted uncertainty**, not current measurement error, evaluated at
the time the candidate would actually be there.

**Reverse is a planned trajectory**, collision-checked with the same swept footprint as forward
motion, not a scripted escape.

**The safety supervisor is authoritative.** When it overrides, the command the vehicle receives is
its own, and the reason is recorded so the event is auditable.
