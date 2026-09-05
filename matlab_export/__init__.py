"""MATLAB / Simulink export of the Python contracts (buses, FSM table, replay script).

Everything produced here is UNTESTED IN MATLAB: no MATLAB installation was
available while it was written. The generated files follow documented
Simulink.Bus / table / jsondecode syntax and are meant as the starting point
for the Stage 2 Simulink model.
"""
from matlab_export.exporter import export

__all__ = ["export"]
