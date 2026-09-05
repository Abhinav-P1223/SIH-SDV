"""The MATLAB exporter reflects the Python contracts (the .m files themselves are untested in MATLAB)."""
import csv
import dataclasses

from autonomy.behavior.state_machine import TRANSITIONS
from autonomy.core.types import BehaviorState, RiskLevel, VehicleState
from matlab_export import export
from matlab_export.buses import bus_specs, enum_code


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_export_writes_every_file(tmp_path):
    paths = export(tmp_path)
    assert {p.name for p in paths} == {"buses.m", "behavior_transitions.m", "behavior_transitions.csv",
                                       "behavior_states.csv", "replay_telemetry.m", "README.md"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    assert "UNTESTED IN MATLAB" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_buses_contain_every_vehicle_state_field(tmp_path):
    export(tmp_path)
    text = (tmp_path / "buses.m").read_text(encoding="utf-8")
    for f in dataclasses.fields(VehicleState):
        assert f"el('{f.name}', 'double'" in text
    assert "VehicleState = Simulink.Bus;" in text
    assert "'Bus: SpeedPolicy'" in text                      # nested bus for BehaviorDecision.speed_policy
    assert "skipped field reason" in text                    # strings are reported, not dropped silently
    assert f"RiskLevel: " + ", ".join(f"{m.value}={m.name}" for m in RiskLevel) in text


def test_bus_specs_type_mapping():
    by_name = {s.name: s for s in bus_specs()}
    control = {e.name: e for e in by_name["ControlCommand"].elements}
    assert control["reverse"].data_type == "boolean"
    assert control["steering_angle"].data_type == "double"
    assert dict(by_name["ControlCommand"].skipped)["source"].startswith("type str")
    metrics = {e.name: e for e in by_name["SimulationMetrics"].elements}
    assert metrics["collision_count"].data_type == "int32"
    assert "NaN when None" in metrics["time_to_completion"].description
    covariance = {e.name: e for e in by_name["ObjectState"].elements}["covariance"]
    assert covariance.dims == [2, 2]
    assert {e.name: e for e in by_name["BehaviorDecision"].elements}["state"].data_type == "int32"


def test_transition_and_state_tables(tmp_path):
    export(tmp_path)
    transitions = _rows(tmp_path / "behavior_transitions.csv")
    assert len(transitions) == len(TRANSITIONS)
    for row, (from_states, to_state, guard) in zip(transitions, TRANSITIONS):
        assert row["to_state"] == to_state.value
        assert row["guard"] == guard.__name__
        assert set(row["from_states"].split(";")) == {s.value for s in from_states}
    states = _rows(tmp_path / "behavior_states.csv")
    assert [s["state"] for s in states] == [s.value for s in BehaviorState]
    assert [int(s["code"]) for s in states] == [enum_code(s) for s in BehaviorState]
    m_text = (tmp_path / "behavior_transitions.m").read_text(encoding="utf-8")
    assert m_text.count("'g_") >= len(TRANSITIONS)
