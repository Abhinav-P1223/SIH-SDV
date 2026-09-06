"""One authoritative configuration path.

`config/*.yaml` is what ships and what every run uses. The dataclass defaults exist only so a
config object can be constructed in isolation (tests, tooling). When the two disagree, code that
builds `PlanningConfig()` directly behaves differently from the same code under
`AutonomyConfig.load()` — which is how a 2.9x difference in the clearance-cost decay length went
unnoticed. These tests fail the moment they drift apart again.
"""
from dataclasses import fields, is_dataclass

import pytest

from autonomy.core.config import AutonomyConfig, PerceptionConfig, load_vehicle_parameters

# Fields whose dataclass default is deliberately empty/neutral and is populated only from YAML.
# Keep this list short and justified; it is an escape hatch, not a dumping ground.
INTENTIONAL_EMPTY_DEFAULTS = {
    "prediction.max_acceleration_by_type",   # per-class override map; {} means "use the scalar limit"
}


def _drift(obj, path: str) -> list[str]:
    """Every field whose loaded value differs from the dataclass default, recursively."""
    fresh = type(obj)()
    out: list[str] = []
    for f in fields(type(obj)):
        loaded_value = getattr(obj, f.name)
        default_value = getattr(fresh, f.name)
        name = f"{path}{f.name}"
        if is_dataclass(loaded_value) and not isinstance(loaded_value, type):
            out += _drift(loaded_value, name + ".")
        elif loaded_value != default_value and name not in INTENTIONAL_EMPTY_DEFAULTS:
            out.append(f"{name}: yaml={loaded_value!r} but dataclass default={default_value!r}")
    return out


@pytest.mark.parametrize("section", ["simulation", "planning", "prediction", "risk",
                                     "behavior", "control", "safety"])
def test_dataclass_defaults_match_the_shipped_yaml(section):
    loaded = getattr(AutonomyConfig.load(), section)
    problems = _drift(loaded, f"{section}.")
    assert not problems, f"{section} config drift:\n  " + "\n  ".join(problems)


def test_vehicle_parameters_match_the_shipped_yaml():
    p = load_vehicle_parameters()
    assert p.max_speed > 0 and p.max_reverse_speed > 0
    assert p.cg_to_front_axle + p.cg_to_rear_axle == pytest.approx(p.wheelbase)


def test_perception_defaults_to_the_sensor_pipeline():
    """The shipped default must be the honest one: sensors, not ground truth."""
    assert PerceptionConfig.load().mode == "sensors"


def test_risk_levels_have_no_dead_critical_threshold():
    """CRITICAL is raised only by the physical TTC view. A score threshold for it would be dead
    config: `_level` can never return CRITICAL, so any value set here would have no effect."""
    levels = AutonomyConfig.load().risk.levels
    assert not hasattr(levels, "critical")
    assert levels.low < levels.medium < levels.high


def test_score_ladder_tops_out_at_high():
    """No risk score, however large, may produce CRITICAL."""
    from autonomy.core.config import load_vehicle_parameters as _p
    from autonomy.core.types import RiskLevel
    from autonomy.risk.risk_engine import RiskEngine
    cfg = AutonomyConfig.load()
    engine = RiskEngine(cfg.risk, cfg.behavior, _p(), 10.0)
    for score in (0.0, 0.3, 0.5, 0.85, 10.0):
        assert engine._level(score, ttc=float("inf")).value <= RiskLevel.HIGH.value
