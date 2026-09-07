# Setup guide

Get the project running in about ten minutes. For the full picture read
[TEAM_HANDOFF.md](TEAM_HANDOFF.md).

## What you need

| | |
|---|---|
| Python | **3.11 or newer** (`pyproject.toml` requires it; developed on 3.11) |
| OS | Windows, Linux or macOS. Developed on Windows 11 |
| GPU | **Not needed.** Everything here is CPU-only, including the models |
| Disk | About 1 GB for the repository. Datasets are extra and optional, see below |

## 1. Clone and enter

```bash
git clone https://github.com/Abhinav-P1223/SIH-SDV.git
cd SIH-SDV
git checkout main
```

## 2. Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

## 3. Install

One command:

```bash
pip install -r requirements.txt
```

That covers everything: the simulator, the tests and the perception packages. It pulls PyTorch, so
expect a few hundred megabytes and a few minutes.

**Linux users, do this first.** The default index serves CUDA builds of PyTorch, which are several
gigabytes and useless here because the project is CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run commands from the repository root. `pip install -e .` also works now, but it is not needed.

## 4. Verify

```bash
python -c "import numpy, yaml, matplotlib, torch, torchvision, scipy, PIL, requests; print('ok')"
```

## 5. Run one scenario

From the repository root:

```bash
python scripts/run_scenario.py SUDDEN_CATTLE_CROSSING --quiet
```

Takes about 20 seconds and writes `logs/sudden_cattle_crossing.jsonl` and a summary CSV. Any of the
eleven scenario names works; see [CODEBASE_MAP.md](CODEBASE_MAP.md).

## 6. Run the tests

```bash
python -m pytest tests/ -q
```

**Expect 348 passed with the datasets present.** Without them you will see roughly 35 skips
instead, which is correct and not a failure. Takes 15 to 25 minutes on a laptop CPU.

For a faster smoke check, the unit tests alone take **about 3 minutes** (measured, 292 tests):

```bash
python -m pytest tests/unit -q -m "not slow"
```

## 7. Reproduce the final validation

```bash
python scripts/final_validation.py
```

22 runs, roughly 15 minutes, writes `docs/phase8_final_validation.json`. Expect 22 of 22 completions
and zero collisions.

## Datasets: optional, and none are required to run the simulator

The closed-loop simulator, every scenario and the final validation need **no datasets at all**. The
two model checkpoints you need are **already in the repository**.

| Dataset | Size | Needed for | How to get it |
|---|---|---|---|
| nuScenes v1.0-mini | 5.1 GB | Fusion validation, detector evaluation | Register at nuscenes.org, extract to `Datasets/v1.0-mini/` |
| IDD-Lite | 42 MB | Segmentation training and evaluation | Register at idd.insaan.iiit.ac.in, extract to `Datasets/idd-lite/` |
| UVH-26 subset | 2.5 GB | Detector fine-tuning and its A/B | `python scripts/select_uvh26_subset.py --download`, reproducible from the tracked manifest |

`Datasets/` is gitignored and **must stay that way**. Never commit dataset files.

## If something breaks

See the troubleshooting section of [TEAM_HANDOFF.md](TEAM_HANDOFF.md). The three most common
problems are running from the wrong directory, a Python older than 3.11, and pip pulling CUDA
wheels on Linux.
