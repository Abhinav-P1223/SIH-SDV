# Phase 4 — timestamp-aware multi-sensor fusion

    measurement(t_sensor) -> predict the track FORWARD to t_sensor -> update there
                          -> predict to the output time when a state is asked for

Only `autonomy/perception/tracker.py` changed. The Kalman filter, the extended-Kalman radial
update, the state vector, the association and the gating are all untouched. Planner, risk,
controller, behaviour and Phase 2, 3B and 3C are untouched.

## 1. The old assumption, stated exactly

`ingest(detections, now)` predicted every track by `now - last_time` and then applied **every**
detection at `now`. `Detection.timestamp` was read exactly once, to append to `latency_samples`.
The filter therefore asserted that camera, LiDAR and radar fired simultaneously.

There was no per-track state time either. The state's validity time was a single global
`self.last_time`, while `Track.last_update` recorded measurement recency, and
`get_object_states` extrapolated from `last_update`. For a coasting track those two differ, so
the published position was advanced twice over the coasting interval. That bug is fixed here as a
consequence of giving each track an explicit `state_time`.

## 2. What the real data says

Measured on nuScenes-mini in Phase 3A and re-measured here across all 10 scenes and 404 keyframes:

| Quantity | Value |
|---|---|
| Camera rate | about 10 Hz |
| LiDAR rate | about 20 Hz |
| Radar rate | about 13.4 Hz |
| **Sensor separation within one keyframe, median** | **66.8 ms** |
| Sensor separation within one keyframe, max | 82.8 ms |

At 10 m/s, 66.8 ms is 0.67 m of real motion, which is comparable to the tracker's own gating
radius. Told the sensors are simultaneous, the filter has to explain that displacement as noise.

## 3. What changed

**`Track.state_time`.** Each track now carries the time its state is valid at, separately from
`last_update` (measurement recency, which drives ageing) and from `created` (birth order, which
breaks ties when duplicate tracks merge). Conflating `created` with the measurement time was a
real bug found during this work: it made the camera, the sensor with the largest latency, always
look like the oldest track and always win the merge.

**`_predict_to(track, t)`.** Forward only, and idempotent. A track already at `t` costs nothing,
which is the whole of the no-double-prediction requirement. Backward prediction is deliberately
not attempted: running a constant-velocity model with a negative interval shrinks the covariance,
which is not a smoother, it is a lie.

**Sensor groups.** A batch is split per sensor, each group carrying the median of its detections'
timestamps. Grouping per sensor rather than per detection preserves the existing guarantee that
one object gets at most one update per sensor per batch, and keeps association well posed by
giving every track a common time while a group is gated.

**Out-of-order handling.** A measurement behind the track's state cannot be rewound into the
filter, so its measurement noise is inflated by `lag^2 * P_vv`: the object moved during the lag,
and how far is exactly the velocity uncertainty integrated over it. Beyond
`max_out_of_order_s` the measurement is dropped and counted, and it is not allowed to sneak back
in as a new track.

**Rejection.** A timestamp after `now` is a clock fault and is rejected. A timestamp older than
`max_measurement_age_s` is rejected. Both are counted, not clamped, so a fault stays visible.

**Output prediction.** `get_object_states(t)` now propagates **both** the state and the covariance
to `t` without mutating the track. Extrapolating the position while reporting an older covariance
publishes a position from one instant beside an uncertainty from another, and the collision margin
downstream is sized from exactly that matrix.

## 4. Update order: measured, not assumed

Strict ascending-time ordering is the textbook sequencing and it is implemented
(`update_order: time`). It is **not** the default, because it was measured and it is worse here.

In this sensor suite the camera has both the largest latency (80 ms against LiDAR's 50 ms) and the
loosest covariance, so strict time order puts the least precise sensor first on every single
cycle. Association degrades: camera-shaped tracks are established before LiDAR is seen.

| Configuration, NARROW_LANE_BOXED_IN | Collisions | Emergency brakes |
|---|---|---|
| Legacy synchronised | 0 | 0 |
| Temporal, strict time order | **1** | 31 |
| Temporal, time order, covariance not propagated | **1** | 7 |
| **Temporal, precision order (shipped)** | **0** | **6** |

Both orderings use the true measurement times and both pay for a late measurement by inflating its
noise. They differ only in which sensor shapes association first. Precision order (LiDAR, radar,
camera) is also the pre-Phase-4 order, so a genuinely synchronised batch behaves exactly as it
always did, which a test asserts to 1e-12.

## 5. nuScenes validation

`scripts/validate_temporal_fusion.py`, all 10 mini scenes, 404 keyframes, 18,538 annotations.
Both modes see identical detections in identical order.

### 5.1 The first version of this experiment was worthless, and why

`detections_for_sample` derives one detection per modality from the **same annotation box**. Every
modality therefore reports the identical position no matter what its timestamp says. Replaying
that compares two fusion strategies on data with the effect under test removed, and it duly found
no difference.

The fix is `motion_consistent`: shift each observation by the annotated velocity times its own
offset from the keyframe, so a camera firing 66.8 ms after the LiDAR sees the object 66.8 ms
further along, which is what a real camera would see. The annotation remains only the source of a
controlled observation; it is not scored against.

### 5.2 Results, motion-consistent observations

| | A synchronised | B temporal |
|---|---|---|
| Association rate | 0.987 | 0.986 |
| Track position error, median | 0.004 m | 0.004 m |
| Track velocity error, median | 0.012 m/s | 0.011 m/s |
| **Track velocity error, p90** | **0.249 m/s** | **0.239 m/s** |
| Reported covariance trace, median | 0.0897 | 0.0953 |
| Sensor-to-track residual, median | 0.005 m | 0.005 m |
| Ingest latency, mean | 260.0 ms | 263.7 ms |

**Read this honestly.** The improvement is real but small: 4% off the p90 velocity error, nothing
measurable on position, and association is fractionally worse. It is not the result the mechanism
deserves, and the reason is the data rather than the code. The replayed observations are
noise-free, because they are annotations, and most nuScenes-mini objects are slow or parked. A
66.8 ms offset on a stationary car carries no information either way. The reported covariance is
6% higher in temporal mode, which is the honest cost of no longer publishing an uncertainty from
an earlier instant than the position beside it.

## 6. Why it ships disabled

`temporal_fusion: false` in `config/sensors.yaml`. This is a deliberate, evidenced decision, not
timidity.

With it enabled, the simulator's autonomy becomes measurably more conservative. Across six seeds
of NARROW_LANE_BOXED_IN neither mode collides, but emergency brakes go from 1, 2, 4, 2, 2, 0 to
7, 9, 5, 22, 4, 12. On the default seed the ego reverses twice to recover rather than threading
the gap, and in SUDDEN_PEDESTRIAN_DART the safety override fires earlier through the risk path
instead of the time-to-collision path. Every one of those runs still reaches the goal without a
collision.

Across the full eight-scenario suite the picture is better than that paragraph alone suggests:
with temporal fusion on, all eight scenarios still complete with zero collisions, and the worst
obstacle clearance anywhere in the suite improves from 0.69 m to 0.85 m. The evidence supports
enabling it. What blocks the flip is narrower: two existing tests pin behavioural details that
change, and this phase is required to leave every existing test passing.

The cause is not a tracking regression. Measured against ground truth on the same scenario,
position error is 0.065 m against 0.067 m and all three sensors are covariance-consistent
(normalised innovation squared 1.19 to 1.55 against an expected 1.39). The cause is that
propagating covariance to the output time removes an over-confidence that the existing collision
margins were tuned against. Retuning those margins is planner work, which this phase is explicitly
forbidden from touching.

So the mechanism ships complete, tested and validated, and the flip is left as a downstream
decision with the evidence attached. Enabling it is one line of configuration.

## 7. Tests

22 new tests in `tests/unit/test_temporal_fusion.py`, driving the tracker directly so the timing
is exactly what each test says it is: camera before LiDAR and the reverse, radar arriving between
camera frames, a 66.8 ms offset, out-of-order arrival, a measurement too stale to fuse, a
timestamp from the future, three sensors in one batch, a LiDAR-only stream, an empty batch,
determinism, covariance growth and shrink, symmetry and positive-definiteness under mixed timing,
and the guarantee that synchronised input is bit-identical between the two paths.

All 268 pre-existing tests still pass, unchanged.

## 8. Simulator rates: benchmarked, and deliberately left alone

All eight scenarios, sensors mode, four configurations. Rates are **not** changed by this phase.

| Rates | Temporal | Completed | Collisions | Brakes | Min clearance |
|---|---|---|---|---|---|
| current 20/10/20 | off | 8/8 | 0 | 4 | 0.69 m |
| current 20/10/20 | **on** | 8/8 | 0 | 8 | **0.85 m** |
| nuScenes-like 10/20/13 | off | 8/8 | 0 | 4 | 0.56 m |
| nuScenes-like 10/20/13 | **on** | 8/8 | 0 | **29** | **0.21 m** |

**Conclusion: do not change the rates.** Temporal fusion is stable at the current rates, and in
fact improves the worst-case obstacle clearance across the whole suite from 0.69 m to 0.85 m, at a
cost of four extra emergency brakes. Under nuScenes-like rates the same mechanism becomes nervous,
with 29 brakes and the worst clearance falling to 0.21 m.

The reason is that the nuScenes-like profile drops the camera from 20 Hz to 10 Hz while this
suite's camera also carries the largest latency, so the oldest and least precise measurement
arrives half as often and each one is trusted across a longer gap. Matching a real sensor suite's
rates is only worth doing alongside matching its latencies and its per-sensor accuracy, and that
is a sensor-model change rather than a fusion change.

## 9. Remaining limitations

**No smoother.** A measurement that arrives behind the filter state is fused with inflated noise
rather than being replayed in its correct place. A proper out-of-sequence-measurement filter would
keep a short state history and re-run it. That is a redesign, which this phase forbids.

**The radar update mixes two instants.** `_update_radial` uses the sensor pose at measurement time
but the ego velocity at `now`. That inconsistency predates Phase 4 and is unchanged; fixing it
needs either an ego history buffer or two more fields on `Detection`.

**Group time is a median, not per detection.** Detections from one sensor in one batch are treated
as simultaneous with each other. That is true for this simulator and for nuScenes keyframes, and
it is what keeps association well posed, but it is an approximation.

**The nuScenes evidence is weak**, for the reasons in section 5.2. A dataset with noisy
detections and faster-moving objects would exercise this properly.

**It is off by default**, so nothing in the shipped simulator currently benefits from it.
