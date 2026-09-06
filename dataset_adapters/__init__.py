"""Adapters that feed real datasets into the autonomy stack's existing interfaces.

Named `dataset_adapters` rather than `datasets` on purpose: the downloaded data lives in
`Datasets/`, and Windows paths are case-insensitive, so a package called `datasets` would land
inside the (gitignored) data folder and vanish from the repository.

This package sits alongside `simulation/`: both are SOURCES of `Detection` objects for
`autonomy/`, and neither is imported by it. Nothing here changes the stack; it exists to prove
that real sensor data satisfies the same contract the simulator does.
"""
