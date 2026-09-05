# Interfaces and Data Contracts

All types live in `autonomy/core/types.py`. Units: metres, seconds, radians,
m/s, m/s². Every type has a `to_dict()` producing plain JSON for telemetry;
these are the fields a Simulink bus of the same name would carry.

## Data contracts

| Type | Fields | Produced by | Consumed by |
|---|---|---|---|
| `VehicleParameters` | wheelbase, cg_to_front_axle (lf), cg_to_rear_axle (lr), width, length, mass, max_steering_angle, max_steering_rate, max_acceleration, max_deceleration, max_speed; derived `max_curvature`, `footprint_center_offset` | `config/vehicle.yaml` | everything |
| `VehicleState` | timestamp, x, y, yaw, longitudinal_velocity, lateral_velocity, yaw_rate, longitudinal_acceleration, steering_angle | `VehicleModel.step()` only | risk, behaviour, planner, tracker, safety, metrics |
| `ControlCommand` | timestamp, steering_angle (requested road-wheel angle), acceleration ≥ 0, brake ≥ 0, source (`tracker` / `safety`) | `TrajectoryTracker.track()`, `SafetySupervisor.check()` | `VehicleModel.step()` (through `ActuatorModel`) |
| `ObjectState` | id, object_type, timestamp, x, y, vx, vy, heading, length, width, confidence, covariance (2×2) | `ObjectStateProvider.get_object_states()` — ground truth now, sensor fusion later | prediction, risk (lead detection), metrics |
| `ObjectPrediction` | object_id, object_type, times[N], x[N], y[N], heading[N], vx[N], vy[N], covariances[N,2,2], length, width, risk_weight; `sigma()` | `Predictor.predict()` | risk, planner |
| `Trajectory` | t[N] (absolute), x, y, yaw, velocity, curvature, acceleration, id | candidate generator | checker, scorer, tracker, risk (plan view) |
| `TrajectoryPoint` | t, x, y, yaw, velocity, curvature, acceleration | `Trajectory.point(i)` | convenience / MATLAB export |
| `CandidateTrajectory` | id, label, trajectory, lateral_offset_end, target_speed, feasible, rejection_reason, rejection_detail, costs{}, weighted_costs{}, total_cost, min_clearance, min_boundary_clearance, collision_time, fallback, margin_only, degraded | planner pipeline | selection, telemetry, debug view |
| `RiskAssessment` | object_id, object_type, distance, relative_speed, closing_speed, ttc (route), ttc_kinematic, ttc_physical, min_predicted_distance, time_of_min_distance, trajectory_intersection, collision_probability, risk_score, risk_level, uncertainty, risk_weight | `RiskEngine.evaluate()` | behaviour, telemetry |
| `RiskSummary` | assessments[], max_level, max_score, min_ttc, min_predicted_distance, worst_object_id, any_intersection, min_ttc_current_speed, lead_object_id / lead_gap / lead_speed, plan_min_ttc, plan_max_score, plan_max_level, plan_any_intersection, plan_min_predicted_distance | `RiskEngine.evaluate()` | behaviour, safety, metrics |
| `SpeedPolicy` | target_speed, allow_lateral_avoidance, force_stop | behaviour | planner |
| `BehaviorDecision` | timestamp, state, previous_state, reason (text), triggers{} (numbers), speed_policy, time_in_state | `BehaviorStateMachine.decide()` | planner, telemetry, dashboard |
| `PlannerOutput` | timestamp, candidates[], selected, latency_ms, feasible_count, rejected_count, rejection_histogram{}, frame_origin (s0, d0, heading_rel) | `Planner.plan()` | tracker, safety, metrics, telemetry |
| `SafetyStatus` | override_active, reason, activation_count, activated_at | `SafetySupervisor.check()` | metrics, telemetry |
| `SimulationMetrics` | see `autonomy/metrics/collector.py`; all computed from executed states | `MetricsCollector` | results, tests |
| `SimulationState` | time, step, ego, objects, predictions, risk, decision, plan, control, safety, running, termination_reason | `Simulation.step()` | callers |
| `TelemetryFrame` | timestamp, step, planning_cycle, ego, control, control_nominal, objects, predictions, risk, decision, plan, tracker_debug, safety, road, metrics, agents | `Simulation.step()` | sinks: memory, JSONL, CSV, console; Stage 2 WebSocket/REST/MATLAB |

## Module interfaces (replaceable units)

```python
class ObjectStateProvider(ABC):                       # autonomy/core/interfaces.py
    def get_object_states(self, timestamp: float) -> list[ObjectState]

class RoadModel(ABC):                                 # autonomy/core/interfaces.py
    def project(self, x, y) -> (s, d, heading_ref)
    def to_cartesian(self, s[N], d[N]) -> (x[N], y[N], heading_ref[N])
    def footprint_inside(self, corners[N,4,2], margin) -> bool[N]
    def boundary_clearance(self, corners[N,4,2]) -> float[N]
    length: float

class VehicleModel(ABC):                              # autonomy/vehicle/models.py
    def step(self, state: VehicleState, command: ControlCommand, dt: float) -> VehicleState
    def footprint(self, state) -> OrientedBox

class Predictor(ABC):                                 # autonomy/prediction/predictor.py
    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]

class RiskEngine:                                     # autonomy/risk/risk_engine.py
    def evaluate(self, ego, ego_trajectory | None, objects, predictions, road | None) -> RiskSummary

class BehaviorStateMachine:                           # autonomy/behavior/state_machine.py
    def decide(self, risk: RiskSummary, ego: VehicleState, now: float, planner_feasible: bool) -> BehaviorDecision

class Planner:                                        # autonomy/planning/planner.py
    def plan(self, ego, decision: BehaviorDecision, predictions, now) -> PlannerOutput

class LateralController(ABC):                         # autonomy/control/lateral.py
    def compute(self, ego, traj) -> (steering_angle, LateralDebug)
class LongitudinalController(ABC):                    # autonomy/control/longitudinal.py
    def compute(self, ego, traj, now, dt) -> (acceleration, brake, LongitudinalDebug)
class TrajectoryTracker:                              # autonomy/control/tracker.py
    def track(self, ego, traj, now, dt) -> ControlCommand

class SafetySupervisor:                               # autonomy/safety/supervisor.py
    def check(self, command, risk, plan | None, ego, now) -> (ControlCommand, SafetyStatus)

class MetricsCollector:                               # autonomy/metrics/collector.py
    def on_plan(self, plan, decision, risk, predictions, objects, now)
    def update(self, state, objects, safety, inside_corridor, dt)
    def finalize(self, completed, now, reason) -> SimulationMetrics

class TelemetrySink(ABC):                             # autonomy/telemetry/telemetry.py
    def write(self, frame: TelemetryFrame)
```

## Invariants the tests enforce

* Only `VehicleModel.step()` mutates the ego pose; the tracker and planner
  never touch `VehicleState`.
* Every `ControlCommand` passes through `ActuatorModel` (angle, rate,
  acceleration, braking saturation) before dynamics.
* The planner consumes `ObjectPrediction` and `RoadModel` only; it never sees
  agent behaviours, object-type branching, or lane markings.
* Every rejected candidate has a `rejection_reason` and detail; every feasible
  candidate has a full `costs` / `weighted_costs` breakdown whose weighted sum
  equals `total_cost`.
* Every `BehaviorDecision` has a non-empty `reason` and numeric `triggers`.
* Metrics are computed from executed `VehicleState`s and actual agent
  footprints, never from planner intent.

## Stage 2 mapping

| Stage 1 | Stage 2 |
|---|---|
| `GroundTruthObjectProvider` | camera + LiDAR + radar → tracker → fusion implementing `ObjectStateProvider`; `ObjectState.covariance` becomes real |
| `ConstantVelocityPredictor` | acceleration- / intent-aware predictor behind `Predictor` |
| `DrivableSpace` polylines | RoadRunner scene → drivable-area extraction implementing `RoadModel` |
| `KinematicBicycleModel` | dynamic bicycle / Vehicle Dynamics Blockset behind `VehicleModel` |
| `BehaviorStateMachine.TRANSITIONS` | Stateflow chart, one state per `BehaviorState`, one transition per table row |
| `TelemetryFrame` dicts | WebSocket / REST / MATLAB adapter as another `TelemetrySink` |
