# Demo script, 5 minutes

Numbers generated from `docs/FINAL_SYSTEM_RESULTS.json`. Say only what is written; every figure
here is measured.

> **What was validated, and how.** The learned detector was validated independently on held-out
> UVH-26 Indian-scene imagery, and its output contract was verified against the unmodified tracking
> pipeline. The autonomy stack was validated independently, end to end, through simulated
> multimodal sensors. These are two separate validations. The simulator renders no image frames, so
> **no detector consumed camera pixels during the closed-loop runs**, and this package does not
> claim end-to-end learned camera perception in closed loop.

## 0:00 to 0:20 — the problem

Indian roads are unstructured. Lane markings are missing or ignored. Traffic is mixed: cars,
motorcycles, auto-rickshaws, pushcarts, cattle and pedestrians share the same space, and behaviour
is negotiated rather than signalled. A planner that assumes lanes and rule-following does not
survive here.

## 0:20 to 0:50 — architecture

Show the block diagram. Camera, LiDAR and radar produce timestamped detections. Kalman fusion and
tracking. Prediction with per-class intent priors. Time-to-collision and risk. A behaviour state
machine. A corridor-frame lattice planner that does not need lane markings, because it plans inside
a drivable corridor. Swept-footprint collision checking. Stanley and PID control into a kinematic
bicycle model with a reverse gear. A safety supervisor that can override the planner.

Say: the vehicle moves only through the vehicle model. There is no scripted trajectory anywhere.

## 0:50 to 1:40 — unstructured road

Run the unmarked village road. Point out that there are no lane markings and the planner is working
in a corridor, choosing lateral offsets rather than following a centre line.

## 1:40 to 2:20 — dense market and pedestrian interaction

Run the dense market. Mixed traffic at low speed, tight clearances. Note that this is one of only
two scenarios where emergency braking occurs under simulated sensors, and that each activation
happened with **zero feasible trajectories**, so braking was the last option rather than a
threshold artefact.

## 2:20 to 2:50 — cattle crossing, prediction and replanning

Run the cattle crossing. At the tightest moment: 2.1 s time-to-collision, 46 of 73 candidates feasible. The behaviour machine enters AVOID
because the predicted path of the cattle intersects the ego's, the trajectory changes, and the
vehicle steers and slows. Show the before, the decision and the after.

## 2:50 to 3:20 — emergency brake

Run the pedestrian dart. At the critical moment: 0.7 s, only 1 of 2 candidates feasible, speed -5.6 m/s. The safety supervisor overrides the planner
and commands a hard stop. Emphasise that the override is authoritative and its reason is recorded.

## 3:20 to 3:50 — reverse recovery

Run the reverse-recovery scenario. A parked car blocks the right, a pushcart blocks the left, and
the gap cannot be lined up from a standstill. At the tightest point: 0.3 s and 0 of 2 candidates feasible. The vehicle backs up,
buys the run-up and then takes the gap. Say clearly: **removing the reverse candidates makes this
scenario fail**, which is how we know the manoeuvre is required rather than decorative.

## 3:50 to 4:20 — measured results

22/22 scenario completions. **0 collisions** across
22 runs covering 11 scenarios in two perception modes. Worst minimum
clearance 0.51 m. Worst p95 replanning latency
55.0 ms against a 100 ms budget. 348 tests
pass, 0 fail.

## 4:20 to 5:00 — Indian-road perception, and its limits

We fine-tuned a camera detector on a small human-verified Indian traffic dataset. Motorcycle recall
improved substantially and auto-rickshaw became detectable at all, since the generic model has no
such class.

Then say the limitations out loud, because a judge will ask:

- That dataset is elevated CCTV, not dashcam, so dashcam transfer is unvalidated.
- Bicycle is data-limited at 106 training boxes and did not recover; an oversampling experiment
  confirmed it.
- Pedestrian and animal detection were not addressed by that data.
- The detector runs at about
  287 ms, outside a 10 Hz budget, so
  it is offline evidence rather than a real-time front end.
- The closed-loop runs used simulated sensors. The detector was validated separately.

## Do not say

Production-ready. Real-world road safety. Real-time learned detection. Validated dashcam
perception. Improved pedestrian or animal detection. End-to-end camera neural perception in closed
loop.
