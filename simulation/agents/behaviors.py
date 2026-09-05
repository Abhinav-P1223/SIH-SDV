"""Agent behaviours.

Each behaviour implements `update(agent, ego_xy, dt)` and mutates the agent's
kinematic state. Behaviours are deliberately simple and parameterised through
`agent.behavior_params`; all randomness comes from `agent.rng` (seeded).

Parameters (all optional, defaults shown):

STATIC             : none
CONSTANT_VELOCITY  : none (uses agent.speed / heading)
CROSSING           : trigger_distance_m (30)  - start when ego is this close (Euclidean)
                     trigger_time_s (None)    - or start at this elapsed time
                     crossing_speed_mps (1.5)
                     acceleration_mps2 (1.5)  - ramp-up to crossing speed
                     crossing_distance_m (6.0)- stop after travelling this far
                     stop_after (True)
MERGING            : trigger_distance_m (25), target_heading_rad (0.0),
                     yaw_rate_rad_s (0.4), target_speed_mps (agent.speed),
                     acceleration_mps2 (1.0)
ERRATIC            : heading_sigma_rad (0.35), speed_sigma_mps (0.5),
                     change_interval_s (1.0), max_speed_mps (3.0)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from autonomy.core.geometry import wrap_angle
from autonomy.core.types import AgentBehaviorType

from .agent import Agent


class AgentBehavior(ABC):
    @abstractmethod
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float) -> None: ...


class StaticBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float) -> None:
        agent.speed = 0.0
        agent.phase = "STATIC"
        agent.elapsed += dt


class ConstantVelocityBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float) -> None:
        agent.phase = "MOVING"
        agent.advance(dt)


class CrossingBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float) -> None:
        p = agent.behavior_params
        trigger_d = p.get("trigger_distance_m", 30.0)
        trigger_t = p.get("trigger_time_s")
        v_target = p.get("crossing_speed_mps", 1.5)
        acc = p.get("acceleration_mps2", 1.5)
        dist_total = p.get("crossing_distance_m", 6.0)
        stop_after = p.get("stop_after", True)

        if agent.phase in ("INIT", "WAITING"):
            agent.phase = "WAITING"
            agent.speed = 0.0
            d_ego = math.hypot(agent.x - ego_xy[0], agent.y - ego_xy[1])
            time_trigger = trigger_t is not None and agent.elapsed >= trigger_t
            if d_ego <= trigger_d or time_trigger:
                agent.phase = "CROSSING"
                agent.memory["travelled"] = 0.0
            agent.elapsed += dt
            return

        if agent.phase == "CROSSING":
            agent.speed = min(v_target, agent.speed + acc * dt)
            before = (agent.x, agent.y)
            agent.advance(dt)
            agent.memory["travelled"] += math.hypot(agent.x - before[0], agent.y - before[1])
            if agent.memory["travelled"] >= dist_total:
                agent.phase = "DONE" if stop_after else "CROSSING"
                if stop_after:
                    agent.speed = 0.0
            return

        agent.speed = 0.0
        agent.elapsed += dt


class MergingBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float) -> None:
        p = agent.behavior_params
        trigger_d = p.get("trigger_distance_m", 25.0)
        target_heading = p.get("target_heading_rad", 0.0)
        yaw_rate = p.get("yaw_rate_rad_s", 0.4)
        v_target = p.get("target_speed_mps", agent.speed)
        acc = p.get("acceleration_mps2", 1.0)

        if agent.phase == "INIT":
            agent.phase = "APPROACH"
        if agent.phase == "APPROACH":
            d_ego = math.hypot(agent.x - ego_xy[0], agent.y - ego_xy[1])
            if d_ego <= trigger_d:
                agent.phase = "MERGING"
        if agent.phase == "MERGING":
            err = float(wrap_angle(target_heading - agent.heading))
            step = max(-yaw_rate * dt, min(yaw_rate * dt, err))
            agent.heading = float(wrap_angle(agent.heading + step))
            if agent.speed < v_target:
                agent.speed = min(v_target, agent.speed + acc * dt)
            elif agent.speed > v_target:
                agent.speed = max(v_target, agent.speed - acc * dt)
            if abs(err) < 1e-3 and abs(agent.speed - v_target) < 1e-3:
                agent.phase = "MERGED"
        agent.advance(dt)


class ErraticBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float) -> None:
        p = agent.behavior_params
        h_sigma = p.get("heading_sigma_rad", 0.35)
        v_sigma = p.get("speed_sigma_mps", 0.5)
        interval = p.get("change_interval_s", 1.0)
        v_max = p.get("max_speed_mps", 3.0)
        agent.phase = "ERRATIC"
        next_change = agent.memory.get("next_change", 0.0)
        if agent.elapsed >= next_change:
            agent.heading = float(wrap_angle(agent.heading + agent.rng.normal(0.0, h_sigma)))
            agent.speed = float(min(v_max, max(0.0, agent.speed + agent.rng.normal(0.0, v_sigma))))
            agent.memory["next_change"] = agent.elapsed + interval
        agent.advance(dt)


BEHAVIORS: dict[AgentBehaviorType, AgentBehavior] = {
    AgentBehaviorType.STATIC: StaticBehavior(),
    AgentBehaviorType.CONSTANT_VELOCITY: ConstantVelocityBehavior(),
    AgentBehaviorType.CROSSING: CrossingBehavior(),
    AgentBehaviorType.MERGING: MergingBehavior(),
    AgentBehaviorType.ERRATIC: ErraticBehavior(),
}


def behavior_for(agent: Agent) -> AgentBehavior:
    return BEHAVIORS[agent.behavior_type]
