# Team onboarding checklist

Work top to bottom. Each step has the command that proves it, so you never have to guess whether it
worked. Full detail in [TEAM_HANDOFF.md](TEAM_HANDOFF.md), quick version in
[SETUP_GUIDE.md](SETUP_GUIDE.md).

## Access and clone

- [ ] **Git access confirmed** — `git ls-remote https://github.com/Abhinav-P1223/SIH-SDV.git` lists refs
- [ ] **Repository cloned** — `git clone https://github.com/Abhinav-P1223/SIH-SDV.git && cd SIH-SDV`
- [ ] **Correct branch checked out** — `git rev-parse --abbrev-ref HEAD` prints `main`
- [ ] **Up to date with remote** — `git fetch origin && git status -sb` shows no `behind`

## Environment

- [ ] **Python verified** — `python -V` prints 3.11 or newer
- [ ] **Virtual environment active** — the prompt shows `.venv`
- [ ] **Dependencies installed** — one command installs everything:
  ```
  pip install -r requirements.txt
  ```
  On Linux run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  first, so pip does not fetch CUDA wheels this CPU-only project never uses.
- [ ] **Imports work** —
  `python -c "import numpy, yaml, matplotlib, torch, torchvision, scipy, PIL, requests; print('ok')"`
- [ ] **You run commands from the repository root** — that is what every documented command assumes

## Local artifacts

- [ ] **Model checkpoints present** — they come with the clone, no download needed:
  ```
  ls perception_detector/checkpoints/fasterrcnn_uvh26.pt
  ls road_perception/checkpoints/fastscnn_iddlite.pt
  ```
- [ ] **You understand that no dataset is required** to run the simulator, the scenarios or the
  final validation
- [ ] **Datasets obtained only if your task needs them** — download links in
  [SETUP_GUIDE.md](SETUP_GUIDE.md), licences in handoff section 6. `Datasets/` is gitignored and
  must never be committed

## It runs

- [ ] **Basic test passes** — `python -m pytest tests/unit -q -m "not slow"`, about 3 minutes
- [ ] **One scenario runs** — `python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet`,
  about 20 seconds, writes `logs/sudden_cattle_crossing.jsonl`
- [ ] **Full test suite passes** — `python -m pytest tests/ -q`, 15 to 25 minutes.
  **348 passed** with datasets present; roughly 35 skips without them, which is correct
- [ ] **Final validation runs** — `python scripts/final_validation.py`, about 15 minutes,
  expect 22 of 22 completions and 0 collisions

## You understand the rules

- [ ] **Frozen architecture** — you have read handoff section 11 and know that `autonomy/`,
  `simulation/scenarios/`, `tests/`, `config/` and the generated reports need lead approval
- [ ] **Safe areas** — you know your work belongs in `visualization/`, the dashboard page, `docs/`,
  presentation assets or a new script
- [ ] **The UI boundary** — the UI reads telemetry, results and configuration. It is never part of
  the planning or control loop
- [ ] **The primary model** — the Phase 7C `fasterrcnn_uvh26.pt` is the deliverable. The Phase 7E
  oversampled checkpoint is an experiment and must not be presented as the final model
- [ ] **The claim you must not blur** — the detector was validated on held-out imagery; the autonomy
  stack was validated through simulated sensors. The simulator renders no images, so no detector
  consumed camera pixels in the closed-loop runs
- [ ] **Reports are generated** — regenerate with `python scripts/build_final_report.py`, never edit
  the numbers by hand

## Before your first pull request

- [ ] Branch created as `feature/<something>`, not committed to `main`
- [ ] `python -m pytest tests/ -q` run, and the result stated in the PR description
- [ ] No dataset file, no new `.pt` checkpoint and no rewritten history in the diff
- [ ] If you touched anything in handoff section 11, you have said so explicitly and asked for review
