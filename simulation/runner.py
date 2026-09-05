"""Closed-loop simulation runner.

    while running:
        objects     = world.object_provider.get_object_states(t)
        if planning cycle due:
            predictions = predictor.predict(objects, t)
            risk        = risk_engine.evaluate(ego, last_selected_trajectory, objects, predictions, road)
            decision    = behavior.decide(risk, ego, t, planner_feasible=last_plan_had_feasible)
            plan        = planner.plan(ego, decision, predictions, t)
        control = tracker.track(ego, plan.selected.trajectory, t, dt)
        control = safety.check(control, risk, plan, ego, t)
        ego     = vehicle.step(ego, control, dt)          # actuator limits inside
        world.update(dt, ego position)
        metrics.update(...); telemetry.publish(frame)

The ego pose is changed only by `vehicle.step`. Nothing else writes to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from autonomy.behavior.state_machine import BehaviorStateMachine
from autonomy.control.tracker import TrajectoryTracker
from autonomy.core.config import AutonomyConfig, ObjectProfiles, PerceptionConfig, load_vehicle_parameters
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig
from autonomy.core.geometry import box_corners
from autonomy.core.types import (BehaviorDecision, ObjectPrediction, PlannerOutput, RiskSummary,
                                 SafetyStatus, SimulationMetrics, SimulationState, VehicleParameters,
                                 VehicleState)
from autonomy.metrics.collector import MetricsCollector
from autonomy.planning.planner import Planner
from autonomy.prediction.predictor import ConstantVelocityPredictor
from autonomy.risk.risk_engine import RiskEngine
from autonomy.safety.supervisor import SafetySupervisor
from autonomy.telemetry.telemetry import (ConsoleSink, CsvSummarySink, InMemorySink, JsonLinesSink,
                                          TelemetryFrame, TelemetryPublisher)
from autonomy.vehicle.models import KinematicBicycleModel, VehicleModel
from simulation.scenarios.loader import Scenario, load_scenario
from simulation.sensors.models import SensorSuite


@dataclass
class RunResult:
    metrics: SimulationMetrics
    frames: list[TelemetryFrame]
    scenario_name: str


class Simulation:
    def __init__(self, scenario: Scenario, cfg: AutonomyConfig, params: VehicleParameters,
                 profiles: ObjectProfiles, publisher: Optional[TelemetryPublisher] = None,
                 vehicle: Optional[VehicleModel] = None, perception: Optional[PerceptionConfig] = None):
        self.sc = scenario
        self.cfg = cfg
        self.params = params
        self.world = scenario.world
        self.road = scenario.world.road
        self.dt = cfg.simulation.dt_s
        self.publisher = publisher or TelemetryPublisher()

        # perception: ground truth (Stage 1 baseline) or simulated sensors -> tracker/fusion
        self.perception = perception or PerceptionConfig(mode="ground_truth")
        self.sensors: Optional[SensorSuite] = None
        self.fusion: Optional[SensorFusionTracker] = None
        if self.perception.mode == "sensors":
            self.sensors = SensorSuite.from_config(self.perception.sensors, scenario.seed)
            self.fusion = SensorFusionTracker(TrackerConfig(**self.perception.tracker), profiles)
        self.object_provider = self.fusion if self.fusion is not None else self.world.object_provider
        self.last_detections = []

        self.ego: VehicleState = scenario.ego.initial_state
        self.vehicle = vehicle or KinematicBicycleModel(params)
        self.predictor = ConstantVelocityPredictor(cfg.prediction, profiles)
        self.risk_engine = RiskEngine(cfg.risk, cfg.behavior, params, scenario.ego.desired_speed)
        self.behavior = BehaviorStateMachine(cfg.behavior, scenario.ego.desired_speed)
        self.planner = Planner(cfg.planning, cfg.risk, params, self.road,
                               scenario.ego.desired_speed, scenario.ego.desired_lateral_offset)
        self.tracker = TrajectoryTracker(cfg.control, params)
        self.safety = SafetySupervisor(cfg.safety)
        self.metrics = MetricsCollector(params, cfg.planning.safety_margin_m)

        self.time = 0.0
        self.step_index = 0
        self.next_plan_time = 0.0
        self.plan: Optional[PlannerOutput] = None
        self.risk: RiskSummary = RiskSummary.empty()
        self.decision: Optional[BehaviorDecision] = None
        self.predictions: list[ObjectPrediction] = []
        self.safety_status = SafetyStatus(False, "", 0)
        self.running = True
        self.termination_reason = ""

    # ------------------------------------------------------------------ #
    def step(self) -> SimulationState:
        t = self.time
        dt = self.dt
        truth = self.world.object_provider.get_object_states(t)
        if self.fusion is not None:
            self.last_detections = self.sensors.sense(self.world.agents, self.ego, t)
            self.fusion.ingest(self.last_detections, t, self.ego)
            self.metrics.m.detections_total += len(self.last_detections)
        objects = self.object_provider.get_object_states(t)      # what the autonomy stack sees

        planning_cycle = self.plan is None or t >= self.next_plan_time - 1e-9
        if planning_cycle:
            self.predictions = self.predictor.predict(objects, t)
            ref = self.plan.selected.trajectory if self.plan else None
            self.risk = self.risk_engine.evaluate(self.ego, ref, objects, self.predictions, self.road)
            had_feasible = self.plan.feasible_count > 0 if self.plan else True
            standstill = self.plan.standstill_s if self.plan else 0.0
            self.decision = self.behavior.decide(self.risk, self.ego, t, planner_feasible=had_feasible,
                                                 standstill_s=standstill)
            self.plan = self.planner.plan(self.ego, self.decision, self.predictions, t)
            self.next_plan_time = t + self.cfg.planning.period_s
            self.metrics.on_plan(self.plan, self.decision, self.risk, self.predictions, objects, t)
            if self.fusion is not None:
                self.metrics.on_perception(objects, truth, self.fusion.mean_latency_s(), 0)

        nominal = self.tracker.track(self.ego, self.plan.selected.trajectory, t, dt)
        control, self.safety_status = self.safety.check(nominal, self.risk, self.plan, self.ego, t)

        self.ego = self.vehicle.step(self.ego, control, dt)          # the only place the ego moves
        self.world.update(dt, (self.ego.x, self.ego.y))
        self.time += dt
        self.step_index += 1

        corners = box_corners(np.array([self.ego.x + self.params.footprint_center_offset * math.cos(self.ego.yaw)]),
                              np.array([self.ego.y + self.params.footprint_center_offset * math.sin(self.ego.yaw)]),
                              np.array([self.ego.yaw]), self.params.length, self.params.width)
        inside = bool(self.road.footprint_inside(corners, 0.0)[0])
        self.metrics.update(self.ego, truth, self.safety_status, inside, dt)   # collisions/clearance vs TRUTH

        self._check_termination()
        frame = TelemetryFrame(
            timestamp=t, step=self.step_index, planning_cycle=planning_cycle,
            ego=self.ego, control=control, control_nominal=nominal,
            objects=objects, predictions=self.predictions, risk=self.risk,
            decision=self.decision, plan=self.plan,
            tracker_debug=self.tracker.last_debug.to_dict() if self.tracker.last_debug else None,
            safety=self.safety_status, road=self.road.to_dict(), metrics=self.metrics.m,
            agents=self.world.to_dict()["agents"],
            detections=self.last_detections, perception_mode=self.perception.mode,
        )
        self.publisher.publish(frame)
        return SimulationState(self.time, self.step_index, self.ego, objects, self.predictions,
                               self.risk, self.decision, self.plan, control, self.safety_status,
                               self.running, self.termination_reason)

    def _check_termination(self) -> None:
        s, _, _ = self.road.project(self.ego.x, self.ego.y)
        if s >= self.world.goal_s - self.cfg.simulation.goal_tolerance_m:
            self.running, self.termination_reason = False, "GOAL_REACHED"
        elif self.cfg.simulation.stop_on_collision and self.metrics.m.collision_count > 0:
            self.running, self.termination_reason = False, "COLLISION"
        elif self.time >= self.cfg.simulation.max_duration_s:
            self.running, self.termination_reason = False, "TIMEOUT"

    def run(self) -> SimulationMetrics:
        while self.running:
            self.step()
        completed = self.termination_reason == "GOAL_REACHED"
        metrics = self.metrics.finalize(completed, self.time, self.termination_reason)
        self.publisher.close()
        return metrics


# ---------------------------------------------------------------------- #
def run_scenario(name: str, log_dir: Path | str | None = "logs", console: bool = True,
                 keep_frames: bool = True, cfg: AutonomyConfig | None = None,
                 perception: PerceptionConfig | str | None = None) -> RunResult:
    """perception: a PerceptionConfig, a mode string ('ground_truth' / 'sensors'), or None for config/sensors.yaml."""
    cfg = cfg or AutonomyConfig.load()
    params = load_vehicle_parameters()
    profiles = ObjectProfiles.load()
    if perception is None:
        perception = PerceptionConfig.load()
    elif isinstance(perception, str):
        perception = PerceptionConfig.load().with_mode(perception)
    scenario = load_scenario(name, profiles)
    memory = InMemorySink()
    publisher = TelemetryPublisher([memory] if keep_frames else [])
    if log_dir is not None:
        log_dir = Path(log_dir)
        publisher.add(JsonLinesSink(log_dir / f"{scenario.name.lower()}.jsonl"))
        publisher.add(CsvSummarySink(log_dir / f"{scenario.name.lower()}_summary.csv"))
    if console:
        publisher.add(ConsoleSink())
    sim = Simulation(scenario, cfg, params, profiles, publisher, perception=perception)
    metrics = sim.run()
    return RunResult(metrics, memory.frames, scenario.name)
