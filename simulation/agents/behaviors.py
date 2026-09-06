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
                     acceleration_mps2 (1.0), follow_road (False) - when True and a
                     road model is available the target heading is the corridor
                     heading at the agent's position (the agent joins the flow and
                     then follows the road, also through curves);
                     yield_gap_m (None) - gap acceptance: waits while the ego is
                     closer than this behind it along the corridor
ERRATIC            : heading_sigma_rad (0.35), speed_sigma_mps (0.5),
                     change_interval_s (1.0), max_speed_mps (3.0)
"""
from __future__ import annotations

import math

import numpy as np
from abc import ABC, abstractmethod

from autonomy.core.geometry import wrap_angle
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import AgentBehaviorType

from .agent import Agent
from .interaction import APPROACH, COMMIT, EgoView, GapAcceptance, YIELD, idm_acceleration, time_to_point


class AgentBehavior(ABC):
    @abstractmethod
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float,
               road: RoadModel | None = None) -> None: ...


class StaticBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float,
               road: RoadModel | None = None) -> None:
        agent.speed = 0.0
        agent.phase = "STATIC"
        agent.elapsed += dt


class ConstantVelocityBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float,
               road: RoadModel | None = None) -> None:
        agent.phase = "MOVING"
        agent.advance(dt)


class CrossingBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float,
               road: RoadModel | None = None) -> None:
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
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float,
               road: RoadModel | None = None) -> None:
        p = agent.behavior_params
        trigger_d = p.get("trigger_distance_m", 25.0)
        target_heading = p.get("target_heading_rad", 0.0)
        if p.get("follow_road", False) and road is not None:
            _, _, target_heading = road.project(agent.x, agent.y)
        yaw_rate = p.get("yaw_rate_rad_s", 0.4)
        v_target = p.get("target_speed_mps", agent.speed)
        acc = p.get("acceleration_mps2", 1.0)

        if agent.phase == "INIT":
            agent.phase = "APPROACH"
        if agent.phase == "APPROACH":
            d_ego = math.hypot(agent.x - ego_xy[0], agent.y - ego_xy[1])
            if d_ego <= trigger_d:
                # gap acceptance: with yield_gap_m set, do not pull out while the ego is closer than that
                # behind us along the corridor (a reactive agent; Indian traffic still often does not yield)
                yield_gap = p.get("yield_gap_m")
                if yield_gap is not None and road is not None:
                    s_a, _, _ = road.project(agent.x, agent.y)
                    s_e, _, _ = road.project(ego_xy[0], ego_xy[1])
                    if 0.0 < s_a - s_e < yield_gap:
                        agent.phase = "WAITING"
                    else:
                        agent.phase = "MERGING"
                else:
                    agent.phase = "MERGING"
        if agent.phase == "WAITING":
            agent.speed = max(0.0, agent.speed - acc * dt)
            s_a, _, _ = road.project(agent.x, agent.y)
            s_e, _, _ = road.project(ego_xy[0], ego_xy[1])
            if not (0.0 < s_a - s_e < p.get("yield_gap_m", 0.0)):
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
        if agent.phase == "MERGED" and p.get("follow_road", False) and road is not None:
            err = float(wrap_angle(target_heading - agent.heading))
            agent.heading = float(wrap_angle(agent.heading + max(-yaw_rate * dt, min(yaw_rate * dt, err))))
        agent.advance(dt)


class ErraticBehavior(AgentBehavior):
    def update(self, agent: Agent, ego_xy: tuple[float, float], dt: float,
               road: RoadModel | None = None) -> None:
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


class InteractiveBehavior(AgentBehavior):
    """Ego-aware agent: IDM along its own path, gap acceptance at a conflict point.

    Opt-in. Nothing else in the repository uses it, so the eight existing scenarios are
    untouched. Parameters (all optional):

        desired_speed_mps   free-flow speed                              (agent.speed or 8.0)
        a_max_mps2          IDM acceleration limit                       (1.5)
        b_comf_mps2         IDM comfortable deceleration                 (2.0)
        min_gap_m           IDM standstill gap                           (2.0)
        headway_s           IDM desired time headway                     (1.5)
        conflict_x/y        the point this agent's path crosses the ego's (none -> follow only)
        critical_gap_s      time gap demanded before crossing            (2.5)
        hysteresis_s        extra gap needed to stop yielding            (0.8)
        reaction_time_s     age of the ego state the agent acts on       (0.5)
        follow_ego          treat the ego as a lead vehicle when ahead   (True)

    With `interactive_traffic` disabled the agent falls back to constant velocity, which is the
    Phase-1 baseline, so the ablation is a genuine A/B of the same scenario.
    """

    @staticmethod
    def _keep_side(agent: Agent, p: dict, road: RoadModel | None, dt: float) -> None:
        """Steer toward a corridor offset at a bounded yaw rate. Lane discipline, not interaction:
        it does not read the ego, so it applies in the baseline too."""
        keep = p.get("keep_offset_m")
        if keep is None or road is None:
            return
        _, d_a, h_ref = road.project(agent.x, agent.y)
        reverse = abs(wrap_angle(agent.heading - h_ref)) > math.pi / 2
        want = h_ref + math.pi if reverse else h_ref
        aim = math.atan(max(-1.0, min(1.0, (float(keep) - d_a) / max(agent.speed, 1.0) / 2.0)))
        want += -aim if reverse else aim
        rate = float(p.get("yaw_rate_rad_s", 0.5))
        step = max(-rate * dt, min(rate * dt, float(wrap_angle(want - agent.heading))))
        agent.heading = float(wrap_angle(agent.heading + step))

    def update(self, agent: Agent, ego_xy, dt: float, road: RoadModel | None = None) -> None:
        p = agent.behavior_params
        # Receiving a bare (x, y) tuple means `simulation.interactive_traffic` is off: the runner
        # withholds the ego's speed and heading, so there is nothing to interact with and the
        # agent falls back to the non-reactive Phase-1 baseline. That is what makes the ablation
        # a genuine A/B rather than the same behaviour with a different label.
        interactive = isinstance(ego_xy, EgoView) and p.get("interactive", True)
        ego = ego_xy if isinstance(ego_xy, EgoView) else EgoView(ego_xy[0], ego_xy[1])

        if not interactive:
            # Baseline: keep to your own side and stay on the road -- neither depends on the ego --
            # but do NOT brake for it or judge its gaps. The A/B then isolates ego-awareness alone
            # rather than confounding it with lane discipline.
            self._keep_side(agent, p, road, dt)
            agent.phase = "MOVING"
            agent.advance(dt)
            return

        v0 = float(p.get("desired_speed_mps", agent.speed if agent.speed > 0 else 8.0))
        a_max = float(p.get("a_max_mps2", 1.5))
        b_comf = float(p.get("b_comf_mps2", 2.0))

        gate = agent.memory.get("gap")
        if gate is None:
            gate = GapAcceptance(critical_gap_s=float(p.get("critical_gap_s", 2.5)),
                                 hysteresis_s=float(p.get("hysteresis_s", 0.8)),
                                 reaction_time_s=float(p.get("reaction_time_s", 0.5)))
            agent.memory["gap"] = gate
        seen = gate.perceive(ego, agent.elapsed)          # delayed view: no instant reactions

        # --- conflict handling: how long until each of us reaches the crossing point ----------
        gap_obstruction = math.inf
        cx, cy = p.get("conflict_x"), p.get("conflict_y")
        if cx is not None and cy is not None:
            t_agent = time_to_point(agent.x, agent.y, agent.speed, agent.heading, float(cx), float(cy))
            t_ego = time_to_point(seen.x, seen.y, seen.speed, seen.heading, float(cx), float(cy))
            d_conflict = math.hypot(float(cx) - agent.x, float(cy) - agent.y)
            past = t_agent < 0.0 or d_conflict < 0.5
            if past:
                gate.release()                            # conflict behind us; next one decides afresh
            else:
                state = gate.decide(t_agent if math.isfinite(t_agent) else 1e6,
                                    t_ego if t_ego >= 0.0 else math.inf)
                if state == YIELD:
                    # Hold short of the conflict point: IDM treats it as a stopped obstruction.
                    gap_obstruction = max(d_conflict - float(p.get("min_gap_m", 2.0)), 0.0)
                agent.memory["gap_state"] = state

        # --- following: the ego as a lead vehicle when it is ahead in our own path ------------
        if p.get("follow_ego", True):
            along = (seen.x - agent.x) * math.cos(agent.heading) + (seen.y - agent.y) * math.sin(agent.heading)
            lateral = abs(-(seen.x - agent.x) * math.sin(agent.heading) + (seen.y - agent.y) * math.cos(agent.heading))
            if along > 0.0 and lateral < float(p.get("follow_lateral_m", 2.5)):
                gap_obstruction = min(gap_obstruction, along - 0.5 * (agent.length + 4.2))

        dv = agent.speed - (seen.speed if math.isfinite(gap_obstruction) else 0.0)
        if math.isfinite(gap_obstruction) and agent.memory.get("gap_state") == YIELD:
            dv = agent.speed                              # yielding to a fixed point, not a mover
        acc = idm_acceleration(agent.speed, v0, gap_obstruction, dv, a_max=a_max, b_comf=b_comf,
                               s0=float(p.get("min_gap_m", 2.0)), headway_s=float(p.get("headway_s", 1.5)))

        self._keep_side(agent, p, road, dt)

        # --- road containment: brake to a stop rather than driving off the corridor -------------
        # A motion command, not a position clamp: the agent decelerates as it runs out of road and
        # comes to rest at the edge, which is what a driver does.
        if road is not None and p.get("stay_on_road", True):
            # Look ahead by the distance it would take to stop, not one step: braking a step
            # before the edge cannot physically bring the agent to rest on the road.
            # Measured to the NOSE, not the centre: a driver stops with the whole vehicle on
            # the road, not with the bonnet over the edge.
            look = (agent.speed * dt + agent.speed ** 2 / (2.0 * max(b_comf, 0.1))
                    + 0.5 * agent.length + float(p.get("road_margin_m", 0.3)))
            nx = agent.x + look * math.cos(agent.heading)
            ny = agent.y + look * math.sin(agent.heading)
            if not bool(road.contains_points(np.array([[nx, ny]]))[0]):
                acc = min(acc, -b_comf)
                if agent.speed + acc * dt <= 0.0:
                    agent.speed = 0.0
                    agent.phase = "EDGE"
                    agent.elapsed += dt
                    return

        agent.speed = max(0.0, agent.speed + acc * dt)
        agent.phase = agent.memory.get("gap_state", "MOVING")
        agent.advance(dt)


# Registered after the class body: the dict above is declared before it.
BEHAVIORS[AgentBehaviorType.INTERACTIVE] = InteractiveBehavior()
