"""Learned drivable-space perception from a single camera image.

    IDD-Lite image -> Fast-SCNN -> 7-class mask -> drivable mask -> corridor estimate

Sits outside `autonomy/`, like `simulation/` and `dataset_adapters/`, and is not imported by it.
The planner is untouched: the corridor estimate is handed over through the same `RoadModel`
contract the simulator already satisfies.
"""
