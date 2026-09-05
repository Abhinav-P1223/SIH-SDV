"""Typed configuration loaded from YAML.

Degrees in YAML are converted to radians here so the rest of the stack only
ever sees SI radians. Unknown keys raise, so typos in config are caught early.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

from .types import ObjectType, VehicleParameters

T = TypeVar("T")

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _build(cls: Type[T], data: dict[str, Any], where: str) -> T:
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown keys in {where}: {sorted(unknown)}")
    return cls(**data)  # type: ignore[arg-type]


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
@dataclass
class SimulationConfig:
    dt_s: float = 0.02
    max_duration_s: float = 60.0
    stop_on_collision: bool = True
    goal_tolerance_m: float = 2.0


@dataclass
class PredictionConfig:
    horizon_s: float = 4.0
    dt_s: float = 0.1
    road_following_heading_tol_deg: float = 30.0   # a road-following class within this of the corridor (either way)
    lateral_velocity_decay_s: float = 1.0          # ... has its lateral velocity decay with this time constant

    @property
    def steps(self) -> int:
        return int(round(self.horizon_s / self.dt_s)) + 1


@dataclass
class RiskLevelsConfig:
    low: float = 0.05
    medium: float = 0.2
    high: float = 0.45
    critical: float = 0.8


@dataclass
class RiskConfig:
    safety_margin_m: float = 0.5
    uncertainty_margin_gain: float = 1.0
    uncertainty_margin_max_m: float = 0.5
    time_constant_s: float = 2.0
    ttc_high_s: float = 3.0
    ttc_critical_s: float = 1.2
    margin_grace_s: float = 0.3          # inside-margin (no overlap) is not a hit before this: current pose
    levels: RiskLevelsConfig = field(default_factory=RiskLevelsConfig)

    def __post_init__(self) -> None:
        if isinstance(self.levels, dict):
            self.levels = _build(RiskLevelsConfig, self.levels, "risk.levels")


@dataclass
class BehaviorConfig:
    min_dwell_s: float = 0.3
    caution_speed_factor: float = 0.6
    avoid_speed_factor: float = 0.7
    follow_time_gap_s: float = 1.5
    follow_lateral_window_m: float = 1.5
    follow_max_range_m: float = 30.0
    stopped_speed_mps: float = 0.3
    caution_exit_score: float = 0.1
    avoid_exit_score: float = 0.25
    standstill_to_stopped_s: float = 1.0


@dataclass
class CostWeights:
    collision: float = 6.0
    clearance: float = 2.0
    smoothness: float = 0.8
    curvature: float = 0.5
    progress: float = 2.5
    boundary: float = 0.6
    speed: float = 1.5
    uncertainty: float = 1.0
    lateral: float = 1.2
    blocked: float = 3.0
    consistency: float = 0.3
    front_pass: float = 1.5

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class PlanningConfig:
    frequency_hz: float = 10.0
    horizon_s: float = 4.0
    dt_s: float = 0.1
    lateral_offsets_m: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])  # legacy (unused when lateral_step_m > 0)
    lateral_step_m: float = 0.5               # lattice of end offsets desired + k*step spanning the whole corridor
    lateral_cost_scale_m: float = 2.0         # normalisation of the lateral-deviation and consistency costs
    crossing_lateral_speed_mps: float = 0.3   # an object moving across the corridor faster than this is 'crossing'
    exposure_sigma_cap_m: float = 1.5         # beyond-horizon checks inflate objects by min(meas sigma, cap)
    speed_fractions: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0])
    lateral_transition_time_s: float = 2.0
    min_lateral_transition_length_m: float = 10.0
    comfortable_deceleration_mps2: float = 3.0
    max_lateral_acceleration_mps2: float = 3.5
    safety_margin_m: float = 0.5
    uncertainty_margin_gain: float = 1.0      # margin += gain * measured position std-dev of the object
    uncertainty_margin_max_m: float = 0.5     # cap: a far, bearing-uncertain track must not block all lateral options
    tracking_margin_speed_gain_s: float = 0.02  # margin += gain * ego speed: anticipated controller tracking error
    tracking_margin_error_cap_m: float = 0.3    # margin += min(|current cross-track error|, cap)
    boundary_margin_m: float = 0.25
    boundary_margin_grace_s: float = 0.5
    clearance_scale_m: float = 2.0
    clearance_saturation_m: float = 1.5       # clearance beyond margin+this costs nothing more
    standstill_speed_mps: float = 0.5
    standstill_progress_gain: float = 0.5     # progress weight grows by this per second at standstill
    standstill_progress_max_factor: float = 6.0
    terminal_exposure_horizon_s: float = 7.0   # end pose must not be hit by CV-extrapolated objects before this
    route_lookahead_s: float = 15.0            # continuation checked this far for the graded 'blocked' cost
    stop_standoff_m: float = 5.0               # a near-stop end state behind a blocked route keeps this gap
    stop_standoff_speed_mps: float = 1.0
    collision_margin_grace_s: float = 0.3      # inside-margin (not overlap) tolerated this long: current pose
    weights: CostWeights = field(default_factory=CostWeights)

    def __post_init__(self) -> None:
        if isinstance(self.weights, dict):
            self.weights = _build(CostWeights, self.weights, "planning.weights")

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def steps(self) -> int:
        return int(round(self.horizon_s / self.dt_s)) + 1


@dataclass
class StanleyConfig:
    k_gain: float = 1.2
    k_soft: float = 1.0
    heading_gain: float = 1.0
    lookahead_time_s: float = 0.4
    min_lookahead_m: float = 1.5
    max_lateral_acceleration_mps2: float = 4.0   # |delta| <= atan(L * a_max / v^2): the tracker cannot exceed comfort


@dataclass
class PIDConfig:
    kp: float = 1.5
    ki: float = 0.15
    kd: float = 0.05
    integral_limit: float = 2.0
    use_feedforward: bool = True


@dataclass
class ControlConfig:
    stanley: StanleyConfig = field(default_factory=StanleyConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    stop_speed_mps: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.stanley, dict):
            self.stanley = _build(StanleyConfig, self.stanley, "control.stanley")
        if isinstance(self.pid, dict):
            self.pid = _build(PIDConfig, self.pid, "control.pid")


@dataclass
class SafetyConfig:
    ttc_critical_s: float = 1.0
    hold_time_s: float = 0.5
    brake_mps2: float = 8.0


@dataclass
class AutonomyConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AutonomyConfig":
        sections = {
            "simulation": SimulationConfig, "prediction": PredictionConfig,
            "risk": RiskConfig, "behavior": BehaviorConfig, "planning": PlanningConfig,
            "control": ControlConfig, "safety": SafetyConfig,
        }
        unknown = set(data) - set(sections)
        if unknown:
            raise KeyError(f"Unknown top-level config sections: {sorted(unknown)}")
        kwargs = {name: _build(cls, data.get(name, {}), name) for name, cls in sections.items()}
        return AutonomyConfig(**kwargs)

    @staticmethod
    def load(path: Path | str | None = None) -> "AutonomyConfig":
        path = Path(path) if path else DEFAULT_CONFIG_DIR / "autonomy.yaml"
        return AutonomyConfig.from_dict(load_yaml(path))


# --------------------------------------------------------------------------- #
@dataclass
class PerceptionConfig:
    """Loaded from config/sensors.yaml. mode: ground_truth | sensors."""
    mode: str = "ground_truth"
    tracker: dict[str, Any] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: Path | str | None = None) -> "PerceptionConfig":
        path = Path(path) if path else DEFAULT_CONFIG_DIR / "sensors.yaml"
        data = load_yaml(path)
        per = data.get("perception", {})
        mode = str(per.get("mode", "ground_truth"))
        if mode not in ("ground_truth", "sensors"):
            raise ValueError(f"perception.mode must be ground_truth or sensors, got {mode}")
        return PerceptionConfig(mode=mode, tracker=dict(per.get("tracker", {})), sensors=dict(data.get("sensors", {})))

    def with_mode(self, mode: str) -> "PerceptionConfig":
        return PerceptionConfig(mode, dict(self.tracker), {k: dict(v) for k, v in self.sensors.items()})

    def without(self, *sensor_names: str) -> "PerceptionConfig":
        """Ablation helper: same config with the named sensors disabled."""
        sensors = {k: dict(v) for k, v in self.sensors.items()}
        for n in sensor_names:
            if n in sensors:
                sensors[n]["enabled"] = False
        return PerceptionConfig(self.mode, dict(self.tracker), sensors)


@dataclass
class ObjectProfile:
    length_m: float
    width_m: float
    sigma_pos0_m: float
    sigma_vel_mps: float
    sigma_acc_mps2: float
    risk_weight: float
    lateral_factor: float = 1.0
    static_factor: float = 1.0
    follows_road: bool = False


class ObjectProfiles:
    """Per-type behaviour profiles with a default fallback."""

    def __init__(self, default: ObjectProfile, profiles: dict[ObjectType, ObjectProfile]):
        self.default = default
        self.profiles = profiles

    def get(self, object_type: ObjectType) -> ObjectProfile:
        return self.profiles.get(object_type, self.default)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ObjectProfiles":
        default = _build(ObjectProfile, data.get("defaults", {}), "object_profiles.defaults")
        profiles: dict[ObjectType, ObjectProfile] = {}
        for name, raw in (data.get("profiles") or {}).items():
            merged = {**data.get("defaults", {}), **raw}
            profiles[ObjectType(name)] = _build(ObjectProfile, merged, f"object_profiles.{name}")
        return ObjectProfiles(default, profiles)

    @staticmethod
    def load(path: Path | str | None = None) -> "ObjectProfiles":
        path = Path(path) if path else DEFAULT_CONFIG_DIR / "object_profiles.yaml"
        return ObjectProfiles.from_dict(load_yaml(path))


# --------------------------------------------------------------------------- #
def vehicle_parameters_from_dict(data: dict[str, Any]) -> VehicleParameters:
    v = data.get("vehicle", data)
    return VehicleParameters(
        name=v.get("name", "ego"),
        wheelbase=float(v["wheelbase_m"]),
        cg_to_front_axle=float(v["cg_to_front_axle_m"]),
        cg_to_rear_axle=float(v["cg_to_rear_axle_m"]),
        width=float(v["width_m"]),
        length=float(v["length_m"]),
        mass=float(v["mass_kg"]),
        max_steering_angle=math.radians(float(v["max_steering_angle_deg"])),
        max_steering_rate=math.radians(float(v["max_steering_rate_deg_s"])),
        max_acceleration=float(v["max_acceleration_mps2"]),
        max_deceleration=float(v["max_deceleration_mps2"]),
        max_speed=float(v["max_speed_mps"]),
    )


def load_vehicle_parameters(path: Path | str | None = None) -> VehicleParameters:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "vehicle.yaml"
    return vehicle_parameters_from_dict(load_yaml(path))
