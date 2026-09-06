# Phase 3A — nuScenes multimodal fusion validation

Goal: prove that real camera, LiDAR and radar observations, with real timestamps and real
calibration, enter the existing fusion and tracking interfaces unchanged. Nothing in `autonomy/`
was modified. Every number below comes from a run over the actual `v1.0-mini` files and is
reproducible with:

```
python scripts/validate_nuscenes_fusion.py --root Datasets/v1.0-mini
```

## 1. Audit of the existing fusion interface

Answering the seven questions asked, from the code rather than from documentation.

**How camera observations enter.** As a `Detection` (`autonomy/core/types.py:165`) with `x`, `y`,
a 2x2 positional `covariance`, plus `object_type` and `class_confidence`. Detections arrive
**already in the world frame**; the sensor model does the transform, not the tracker.

**How LiDAR observations enter.** The same `Detection`, with a tight covariance plus `length`,
`width` and `heading`. `_absorb` (`tracker.py:140-157`) keeps a running mean of the extents and
stores the LiDAR heading.

**How radar relative velocity enters.** `radial_speed` and `radial_speed_std` on the `Detection`,
consumed by `_update_radial` (`tracker.py:116-132`). That is a genuine EKF update: it linearises
`h(x) = (v_track - v_ego) . u` about the sensor-to-track unit vector and subtracts the ego's own
velocity. It needs `sensor_x` / `sensor_y`, the sensor's world position at measurement time, to
form `u`.

**How timestamps are currently handled.** `ingest(detections, now, ego)` (`tracker.py:159`)
predicts **every** track by `dt = now - last_time` once, before applying any detection.
`Detection.timestamp` is used only for the latency statistic at line 168. It never reaches the
filter.

**Does fusion assume synchronised sensors? Yes.** Within one `ingest` call, every detection is
applied as if measured at `now`. In the simulator this is harmless: the latency queue releases
detections at their arrival time and `ingest` runs at 50 Hz, so the intra-batch spread is at most
one 20 ms step. On real data the spread is **67 ms median, 83 ms worst**, which at 10 m/s is
0.7 m of position error — several times the LiDAR's own noise.

**Are calibration and extrinsics represented?** Only as `Detection.sensor_x` / `sensor_y`: the
sensor's 2-D world position. There is no rotation, no 3-D extrinsic and no intrinsic anywhere in
the stack. That is sufficient for the radar geometry and nothing else, so the adapter must
perform all frame work itself.

**What needed modification.** Nothing existing. Phase 3A added `dataset_adapters/nuscenes.py`,
`scripts/validate_nuscenes_fusion.py` and `tests/unit/test_nuscenes_adapter.py`.

## 2. nuScenes sensor configuration, measured

| Channel | Modality | Rate | Format |
|---|---|---|---|
| CAM_FRONT and five others | camera | 10.0 Hz | JPEG |
| LIDAR_TOP | lidar | 20.1 Hz | binary point cloud |
| RADAR_FRONT and four others | radar | 13.3–13.4 Hz | PCD |

Twelve channels, 404 annotated keyframes across 10 scenes, 18,538 3D boxes, 120 calibrated sensor
entries of which 60 carry a camera intrinsic, and one ego pose per `sample_data` row.

**Timing offsets.** Sensors inside a keyframe are not simultaneous: **median 66.8 ms, max
82.8 ms** across all 404 keyframes.

**Rate mismatch with our simulator.** Our `config/sensors.yaml` says camera 20 Hz, LiDAR 10 Hz,
radar 20 Hz. A real suite runs LiDAR as the *fast* sensor and the camera as the *slow* one; ours
is inverted. **Left unchanged deliberately**, per the phase instruction to quantify before fixing.

## 3. Adapter design

`dataset_adapters/nuscenes.py` translates metadata into `Detection` objects. It reads only the
JSON tables, never the sensor blobs, so a full run costs megabytes rather than gigabytes, and it
avoids `nuscenes-devkit` so the project keeps its four dependencies.

**Frames.** The nuScenes global frame is used directly as the world frame. Annotations are
already global, so nothing is transformed for them. The sensor's world position is
`ego_translation + R(ego_rotation) @ sensor_translation`, evaluated at that sensor's own
timestamp, from the real `calibrated_sensor` and `ego_pose` records. No transform is hard-coded.

**Visibility** uses the real calibration: a camera observation exists only if the box projects
inside 1600x900 with the actual 3x3 intrinsic and lies in front of the lens; radar respects each
unit's field of view and range; LiDAR is omnidirectional to 80 m.

**Timestamp strategy, stated explicitly.** Each `Detection` keeps the timestamp of the
`sample_data` row it came from. The tracker requires one fusion time per `ingest`, so the
validation ingests at the **latest** timestamp in the batch. This never asks the filter to run
backwards, and the residual spread is reported rather than hidden. No interpolation is performed
and no detection is back-dated: doing either would need a change inside the tracker, which this
phase forbids.

**Sensor arrays.** nuScenes has six cameras and five radars; our `Detection` contract and the
tracker's per-sensor association assume one unit per modality. The adapter therefore keeps one
observation per object per modality, from the nearest unit. Measured effect: camera observations
fall from 7,378 to 6,759 and radar from 6,214 to 4,816 over two scenes, while the live track count
is essentially unchanged (77.3 to 76.9 per sample). The duplicates were **not** inflating the
track count; the tracker's gating already merged them. Pass `one_per_modality=False` to compare.

**Velocity is derived, never invented.** Range rate comes from the finite difference between the
neighbouring annotated boxes of the same instance, projected onto the sensor-to-object direction.
An object with no usable neighbour yields `None`, and `radial_speed_std` stays 0, so no
confidence is claimed for a value we do not have. 23 of 13,225 radar observations fall in that
category.

## 4. Validation results, all 10 scenes

| Metric | Value |
|---|---|
| Scenes processed | 10 |
| Keyframe samples processed | 404 |
| Samples with no observation | 0 |
| Camera observations | 18,452 |
| LiDAR observations | 18,034 |
| Radar observations | 13,225 |
| Radar carrying a range rate | 13,202 (99.8%) |
| Radar with no derivable velocity | 23 |
| Timestamp associations in spec | 404 |
| Timestamp associations out of spec | 0 |
| Intra-keyframe sensor spread | median 66.8 ms, max 82.8 ms |
| Fused objects, summed over samples | 17,401 |
| Live tracks per sample | mean 43.1, max 136 |
| Fused track vs annotated box | median 0.00 m, p95 0.12 m |
| Ingest latency per sample | mean 132.6 ms, p95 582.3 ms, max 960.8 ms |

Tracking succeeded on every sample. All three modalities reached the tracker, tracks carry finite
position, velocity and a positive-definite covariance, and no dataset identity leaks into a
published track id.

The sub-centimetre median position agreement is expected and is **not** a perception result: the
observations were derived from the same annotated boxes, so this checks that the adapter and
fusion path preserve geometry, nothing more.

## 5. Limitations, stated plainly

**This is not perception.** nuScenes ships images and point clouds; the observations here come
from annotated 3D boxes. Phase 3A validates the interface, the timing path and the calibration
path. It makes no claim about detection quality, and no accuracy number from it should be
presented as one.

**Ingest latency is the real finding.** 132 ms mean and 582 ms p95 per sample, against roughly
2 ms in the simulator. The cause is complexity, not the interface: association is O(detections x
tracks) with a `np.linalg.solve` per pair (`tracker.py:134-138`, applied at line 180). At
simulator scale, about 10 objects, that is 100 evaluations; at nuScenes scale, 86 objects visible
within LiDAR range, it is roughly 20,000 per ingest. The tracker is correct and unmodified; it is
simply not dimensioned for this object density. Fixing it means a spatial index or a vectorised
gate, which is a fusion change and therefore out of scope here.

**Synchronisation is still flattened.** The adapter preserves each sensor's timestamp, but the
tracker applies a batch at a single time. The 67 ms median spread is therefore absorbed as
position error rather than compensated. Compensating it properly means per-detection prediction
inside `ingest`, which this phase forbids. It is quantified above so the decision can be taken
deliberately.

**Two-dimensional only.** The stack is planar. Box heights, roll and pitch are discarded, and
the camera intrinsic is used for visibility but not for any 3-D reasoning.

**Ego velocity is not populated** in the `VehicleState` handed to the tracker, because nuScenes
gives pose rather than speed directly. The radar EKF subtracts an ego velocity of zero, which is
correct in the global frame used here but would need the derived ego speed if the adapter were
moved to an ego-relative frame.

## 6. What this justifies doing next

The interface is proven against real multimodal data. Two follow-ups are now evidence-backed
rather than speculative: correcting the inverted sensor rates in `config/sensors.yaml`, and
deciding whether the tracker should compensate intra-batch time offsets. Neither was done here.
