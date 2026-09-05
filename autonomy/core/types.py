"""Data contracts shared by every autonomy module.

All quantities are SI: metres, seconds, radians, m/s, m/s^2.
These dataclasses are the Python counterparts of the Simulink buses that will
carry the same signals in Stage 2. Keep them free of behaviour beyond simple
conversions so they can be serialised (telemetry) and re-created (MATLAB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class ObjectType(str, Enum):
    PEDESTRIAN = "PEDESTRIAN"
    BICYCLE = "BICYCLE"
    MOTORCYCLE = "MOTORCYCLE"
    AUTO_RICKSHAW = "AUTO_RICKSHAW"
    CAR = "CAR"
    BUS = "BUS"
    TRUCK = "TRUCK"
    PUSHCART = "PUSHCART"
    CATTLE = "CATTLE"
    UNKNOWN = "UNKNOWN"


class AgentBehaviorType(str, Enum):
    STATIC = "STATIC"
    CONSTANT_VELOCITY = "CONSTANT_VELOCITY"
    CROSSING = "CROSSING"
    MERGING = "MERGING"
    ERRATIC = "ERRATIC"


class RiskLevel(int, Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class BehaviorState(str, Enum):
    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    CAUTION = "CAUTION"
    AVOID = "AVOID"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    STOPPED = "STOPPED"


class RejectionReason(str, Enum):
    NONE = "NONE"
    CURVATURE = "CURVATURE_LIMIT"
    LATERAL_ACCEL = "LATERAL_ACCELERATION_LIMIT"
    BOUNDARY = "OUTSIDE_DRIVABLE_SPACE"
    COLLISION = "PREDICTED_COLLISION"


# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
@dataclass
class VehicleParameters:
    name: str
    wheelbase: float                 # L = lf + lr
    cg_to_front_axle: float          # lf
    cg_to_rear_axle: float           # lr
    width: float
    length: float
    mass: float
    max_steering_angle: float        # rad
    max_steering_rate: float         # rad/s
    max_acceleration: float          # m/s^2 (>0)
    max_deceleration: float          # m/s^2 (>0, braking magnitude)
    max_speed: float                 # m/s

    def __post_init__(self) -> None:
        if abs(self.cg_to_front_axle + self.cg_to_rear_axle - self.wheelbase) > 1e-6:
            raise ValueError("cg_to_front_axle + cg_to_rear_axle must equal wheelbase")
        for name in ("max_steering_angle", "max_steering_rate", "max_acceleration",
                     "max_deceleration", "max_speed", "width", "length", "mass"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def max_curvature(self) -> float:
        """Kinematic curvature limit 1/R_min = tan(delta_max) / L."""
        return math.tan(self.max_steering_angle) / self.wheelbase

    @property
    def footprint_center_offset(self) -> float:
        """Longitudinal offset from CG to geometric footprint centre.

        The footprint is assumed centred at the wheelbase midpoint.
        """
        return self.wheelbase / 2.0 - self.cg_to_rear_axle


@dataclass
class VehicleState:
    timestamp: float
    x: float
    y: float
    yaw: float
    longitudinal_velocity: float
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    longitudinal_acceleration: float = 0.0
    steering_angle: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.longitudinal_velocity, self.lateral_velocity)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ControlCommand:
    """Command requested from the vehicle.

    steering_angle : desired road-wheel angle (rad). The actuator model applies
                     angle and rate saturation.
    acceleration   : desired positive longitudinal acceleration (m/s^2, >= 0)
    brake          : desired deceleration magnitude (m/s^2, >= 0)
    source         : which module produced the command ("tracker", "safety", ...)
    """
    timestamp: float
    steering_angle: float
    acceleration: float
    brake: float
    source: str = "tracker"

    @property
    def net_acceleration(self) -> float:
        return self.acceleration - self.brake

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Objects / prediction
# --------------------------------------------------------------------------- #
@dataclass
class ObjectState:
    """Tracked object as seen by the autonomy stack.

    Stage 1: produced from ground truth. Stage 2: produced by sensor fusion.
    covariance is the 2x2 positional covariance (m^2).
    """
    id: str
    object_type: ObjectType
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    length: float
    width: float
    confidence: float = 1.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.object_type.value,
            "timestamp": self.timestamp,
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "heading": self.heading, "length": self.length, "width": self.width,
            "confidence": self.confidence,
            "covariance": np.asarray(self.covariance).tolist(),
        }


@dataclass
class ObjectPrediction:
    """Time-indexed predicted motion of one object.

    Arrays share length N. covariances has shape (N, 2, 2). risk_weight comes
    from the object profile so downstream modules need not know object types.
    """
    object_id: str
    object_type: ObjectType
    times: np.ndarray            # absolute simulation time (s)
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    covariances: np.ndarray      # (N, 2, 2)
    length: float
    width: float
    risk_weight: float = 1.0

    def sigma(self) -> np.ndarray:
        """Isotropic positional std-dev per time step, sqrt(max eigenvalue)."""
        return np.sqrt(np.maximum(self.covariances[:, 0, 0], self.covariances[:, 1, 1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "type": self.object_type.value,
            "t": self.times.tolist(),
            "x": self.x.tolist(), "y": self.y.tolist(),
            "heading": self.heading.tolist(),
            "sigma": self.sigma().tolist(),
            "length": self.length, "width": self.width,
            "risk_weight": self.risk_weight,
        }


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float
    velocity: float
    curvature: float
    acceleration: float = 0.0


@dataclass
class Trajectory:
    """Time-indexed ego trajectory stored as parallel numpy arrays.

    t is absolute simulation time. All arrays have the same length.
    """
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    curvature: np.ndarray
    acceleration: np.ndarray
    id: str = ""

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def point(self, i: int) -> TrajectoryPoint:
        return TrajectoryPoint(
            float(self.t[i]), float(self.x[i]), float(self.y[i]), float(self.yaw[i]),
            float(self.velocity[i]), float(self.curvature[i]), float(self.acceleration[i]),
        )

    def points(self) -> list[TrajectoryPoint]:
        return [self.point(i) for i in range(len(self))]

    def path_length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    def to_dict(self, stride: int = 1) -> dict[str, Any]:
        s = slice(None, None, stride)
        return {
            "id": self.id,
            "t": self.t[s].tolist(), "x": self.x[s].tolist(), "y": self.y[s].tolist(),
            "yaw": self.yaw[s].tolist(), "v": self.velocity[s].tolist(),
            "kappa": self.curvature[s].tolist(), "a": self.acceleration[s].tolist(),
        }


@dataclass
class CandidateTrajectory:
    id: str
    label: str
    trajectory: Trajectory
    lateral_offset_end: float          # d_end in corridor frame (m)
    target_speed: float                # terminal speed (m/s)
    feasible: bool = True
    rejection_reason: RejectionReason = RejectionReason.NONE
    rejection_detail: str = ""
    costs: dict[str, float] = field(default_factory=dict)      # unweighted components
    weighted_costs: dict[str, float] = field(default_factory=dict)
    total_cost: float = math.inf
    min_clearance: float = math.inf     # to any predicted object footprint (m)
    min_boundary_clearance: float = math.inf
    collision_time: Optional[float] = None
    fallback: bool = False
    margin_only: bool = False          # rejected solely for the boundary *margin*, still on the road
    degraded: bool = False             # selected although margin_only (no fully feasible candidate)

    def to_dict(self, stride: int = 1) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label,
            "lateral_offset_end": self.lateral_offset_end,
            "target_speed": self.target_speed,
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason.value,
            "rejection_detail": self.rejection_detail,
            "costs": self.costs, "weighted_costs": self.weighted_costs,
            "total_cost": None if math.isinf(self.total_cost) else self.total_cost,
            "min_clearance": None if math.isinf(self.min_clearance) else self.min_clearance,
            "min_boundary_clearance": None if math.isinf(self.min_boundary_clearance) else self.min_boundary_clearance,
            "collision_time": self.collision_time,
            "fallback": self.fallback,
            "margin_only": self.margin_only,
            "degraded": self.degraded,
            "trajectory": self.trajectory.to_dict(stride),
        }


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
@dataclass
class RiskAssessment:
    object_id: str
    object_type: ObjectType
    distance: float                    # current footprint-to-footprint distance (m)
    relative_speed: float              # |v_obj - v_ego| (m/s)
    closing_speed: float               # rate at which the gap shrinks (m/s, >0 closing)
    ttc: float                         # earliest predicted overlap time from now (s), inf if none
    ttc_kinematic: float               # distance / closing_speed, inf if not closing
    min_predicted_distance: float
    time_of_min_distance: float        # relative to now (s)
    trajectory_intersection: bool
    collision_probability: float       # max over horizon of p(t)
    risk_score: float                  # weighted, time-discounted
    risk_level: RiskLevel
    uncertainty: float                 # sigma at time_of_min_distance (m)
    risk_weight: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["object_type"] = self.object_type.value
        d["risk_level"] = self.risk_level.name
        for k in ("ttc", "ttc_kinematic", "min_predicted_distance", "distance"):
            if math.isinf(d[k]):
                d[k] = None
        return d


@dataclass
class RiskSummary:
    assessments: list[RiskAssessment]
    max_level: RiskLevel
    max_score: float
    min_ttc: float
    min_predicted_distance: float
    worst_object_id: Optional[str]
    any_intersection: bool
    lead_object_id: Optional[str] = None     # slower object ahead in corridor, if any
    lead_gap: float = math.inf
    lead_speed: float = 0.0
    min_ttc_current_speed: float = math.inf   # physical TTC along the corridor at current speed
    # aggregates along the currently selected plan (route motion is used for the fields above)
    plan_min_ttc: float = math.inf
    plan_max_score: float = 0.0
    plan_max_level: RiskLevel = RiskLevel.NONE
    plan_any_intersection: bool = False
    plan_min_predicted_distance: float = math.inf

    @staticmethod
    def empty() -> "RiskSummary":
        return RiskSummary([], RiskLevel.NONE, 0.0, math.inf, math.inf, None, False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_level": self.max_level.name,
            "max_score": self.max_score,
            "min_ttc": None if math.isinf(self.min_ttc) else self.min_ttc,
            "min_predicted_distance": None if math.isinf(self.min_predicted_distance) else self.min_predicted_distance,
            "worst_object_id": self.worst_object_id,
            "any_intersection": self.any_intersection,
            "lead_object_id": self.lead_object_id,
            "lead_gap": None if math.isinf(self.lead_gap) else self.lead_gap,
            "lead_speed": self.lead_speed,
            "min_ttc_current_speed": None if math.isinf(self.min_ttc_current_speed) else self.min_ttc_current_speed,
            "plan_min_ttc": None if math.isinf(self.plan_min_ttc) else self.plan_min_ttc,
            "plan_max_score": self.plan_max_score,
            "plan_max_level": self.plan_max_level.name,
            "plan_any_intersection": self.plan_any_intersection,
            "plan_min_predicted_distance": None if math.isinf(self.plan_min_predicted_distance) else self.plan_min_predicted_distance,
            "objects": [a.to_dict() for a in self.assessments],
        }


# --------------------------------------------------------------------------- #
# Behavior / planner outputs
# --------------------------------------------------------------------------- #
@dataclass
class SpeedPolicy:
    target_speed: float                # m/s the planner should aim for
    allow_lateral_avoidance: bool      # may the planner leave the desired offset
    force_stop: bool                   # only stop trajectories are acceptable


@dataclass
class BehaviorDecision:
    timestamp: float
    state: BehaviorState
    previous_state: BehaviorState
    reason: str
    triggers: dict[str, Any]
    speed_policy: SpeedPolicy
    time_in_state: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "reason": self.reason,
            "triggers": self.triggers,
            "target_speed": self.speed_policy.target_speed,
            "allow_lateral_avoidance": self.speed_policy.allow_lateral_avoidance,
            "force_stop": self.speed_policy.force_stop,
            "time_in_state": self.time_in_state,
        }


@dataclass
class PlannerOutput:
    timestamp: float
    candidates: list[CandidateTrajectory]
    selected: CandidateTrajectory
    latency_ms: float
    feasible_count: int
    rejected_count: int
    rejection_histogram: dict[str, int]
    frame_origin: tuple[float, float, float]     # (s0, d0, heading_rel) of ego in corridor frame

    def to_dict(self, stride: int = 2) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "feasible_count": self.feasible_count,
            "rejected_count": self.rejected_count,
            "rejection_histogram": self.rejection_histogram,
            "selected_id": self.selected.id,
            "selected": self.selected.to_dict(stride),
            "candidates": [c.to_dict(stride) for c in self.candidates],
            "frame_origin": list(self.frame_origin),
        }


@dataclass
class SafetyStatus:
    override_active: bool
    reason: str
    activation_count: int
    activated_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimulationMetrics:
    minimum_obstacle_clearance: float = math.inf
    collision_count: int = 0
    scenario_completed: bool = False
    time_to_completion: Optional[float] = None
    replanning_count: int = 0
    plan_change_count: int = 0
    planning_latency_mean_ms: float = 0.0
    planning_latency_max_ms: float = 0.0
    max_acceleration: float = 0.0
    max_deceleration: float = 0.0
    max_steering_rate: float = 0.0
    max_curvature: float = 0.0
    mean_abs_curvature: float = 0.0
    path_length: float = 0.0
    average_speed: float = 0.0
    emergency_brake_activations: int = 0
    minimum_ttc: float = math.inf
    behavior_state_durations: dict[str, float] = field(default_factory=dict)
    behavior_transitions: int = 0
    # Stage 2 hooks (computed where possible, None otherwise)
    max_jerk: Optional[float] = None
    path_smoothness: Optional[float] = None
    scenario_success_rate: Optional[float] = None
    perception_latency_ms: Optional[float] = None
    prediction_error_m: Optional[float] = None
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and math.isinf(v):
                d[k] = None
        return d


@dataclass
class SimulationState:
    time: float
    step: int
    ego: VehicleState
    objects: list[ObjectState]
    predictions: list[ObjectPrediction]
    risk: RiskSummary
    decision: Optional[BehaviorDecision]
    plan: Optional[PlannerOutput]
    control: Optional[ControlCommand]
    safety: Optional[SafetyStatus]
    running: bool = True
    termination_reason: str = ""
