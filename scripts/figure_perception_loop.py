"""Phase 3C step 6: four-panel evidence, image to planner trajectory.

    python scripts/figure_perception_loop.py

Produces `docs/img/p3c_{normal,difficult,rejected}.png`, each showing the whole path:

    1. the original IDD-Lite photograph
    2. the network's prediction, with the drivable class picked out
    3. the extracted corridor in metres, with the stretch the gate approved
    4. the trajectory the EXISTING planner actually drove through it

Cases are chosen by measurement, not by eye. `--pick` reads `docs/phase3c_results.json`, sorts the
accepted episodes by how far the predicted corridor's boundaries sit from the ground-truth ones,
and takes the best and the worst of them. The rejected case is a real gate refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from PIL import Image                                                    # noqa: E402

from autonomy.core.config import (AutonomyConfig, ObjectProfiles, PerceptionConfig,  # noqa: E402
                                  load_vehicle_parameters)
from autonomy.telemetry.telemetry import TelemetryPublisher              # noqa: E402
from road_perception.dataset import CLASS_NAMES, DRIVABLE, IDDLite, INPUT_H, INPUT_W  # noqa: E402
from road_perception.drivable import corridor_width_profile              # noqa: E402
from road_perception.planner_bridge import PerceptionPlannerBridge       # noqa: E402
from run_perception_loop import IDENTITY_ANCHOR, build_scenario          # noqa: E402
from simulation.runner import Simulation                                 # noqa: E402

PALETTE = np.array([[107, 142, 35], [220, 20, 60], [255, 200, 0], [0, 130, 200],
                    [145, 30, 180], [128, 128, 128], [135, 206, 235]], dtype=np.uint8)


def colourise(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask < len(PALETTE)
    out[valid] = PALETTE[mask[valid]]
    return out


def drive_path(road, cfg, params, profiles, speed: float) -> tuple[np.ndarray, str]:
    """Run the shipped Simulation and return the path the ego actually took."""
    sim = Simulation(build_scenario(road, "P3C_FIG", speed), cfg, params, profiles,
                     TelemetryPublisher(), perception=PerceptionConfig(mode="ground_truth"))
    xs = []
    while sim.running:
        sim.step()
        xs.append((sim.ego.x, sim.ego.y))
    return np.asarray(xs), sim.termination_reason


def figure(bridge, ds, index: int, label: str, note: str, out: Path,
           cfg, params, profiles, speed: float) -> None:
    p = ds._item(index)
    img_full = np.asarray(Image.open(p.image).convert("RGB"))
    img = np.asarray(Image.open(p.image).convert("RGB").resize((INPUT_W, INPUT_H)))
    res = bridge.from_image(img_full, IDENTITY_ANCHOR)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.0))
    ax[0].imshow(img)
    ax[0].set_title(f"1. IDD-Lite image\n{p.image.name}", fontsize=9)

    ax[1].imshow(colourise(res.prediction))
    frac = float(np.mean(res.prediction == DRIVABLE)) * 100.0
    ax[1].set_title(f"2. prediction, {frac:.0f}% drivable\n{res.inference_ms:.0f} ms inference",
                    fontsize=9)

    c = res.corridor
    if c.valid_rows:
        ax[2].plot(c.left[:, 1], c.left[:, 0], color="0.7", lw=1.2, label="extracted")
        ax[2].plot(c.right[:, 1], c.right[:, 0], color="0.7", lw=1.2)
        ax[2].plot(c.reference[:, 1], c.reference[:, 0], color="0.7", lw=1.0, ls=":")
    if res.gate.ok:
        u = res.gate.usable
        ax[2].plot(u.left[:, 1], u.left[:, 0], color="tab:green", lw=2.0, label="gate approved")
        ax[2].plot(u.right[:, 1], u.right[:, 0], color="tab:green", lw=2.0)
        w = float(np.median(corridor_width_profile(u)))
        sub = f"approved to {res.gate.lookahead_m:.0f} m, median width {w:.1f} m"
    else:
        sub = f"REJECTED [{res.gate.code}]"
    ax[2].set_title(f"3. corridor, vehicle frame\n{sub}", fontsize=9)
    ax[2].set_xlabel("lateral, m"); ax[2].set_ylabel("range, m")
    ax[2].invert_xaxis(); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=7, loc="upper right")

    if res.gate.ok:
        u = res.gate.usable
        path, reason = drive_path(res.road, cfg, params, profiles, speed)
        ax[3].fill(np.concatenate([u.left[:, 1], u.right[::-1, 1]]),
                   np.concatenate([u.left[:, 0], u.right[::-1, 0]]),
                   color="tab:green", alpha=0.15)
        ax[3].plot(path[:, 1], path[:, 0], color="tab:blue", lw=2.2, label="ego path")
        ax[3].plot(path[0, 1], path[0, 0], "o", color="k", ms=5)
        ax[3].set_title(f"4. EXISTING planner drove it\n{reason}, {len(path)} steps", fontsize=9)
        ax[3].legend(fontsize=7, loc="upper right")
    else:
        ax[3].text(0.5, 0.55, "no trajectory", ha="center", fontsize=13, weight="bold")
        ax[3].text(0.5, 0.42, "the gate refused this corridor,\nso the planner never saw it",
                   ha="center", fontsize=9)
        ax[3].text(0.5, 0.24, res.gate.reason, ha="center", fontsize=7.5, color="0.35", wrap=True)
        ax[3].set_title("4. EXISTING planner\nnot invoked", fontsize=9)
        ax[3].set_xlim(0, 1); ax[3].set_ylim(0, 1)
    ax[3].set_xlabel("lateral, m"); ax[3].set_ylabel("range, m")
    if res.gate.ok:
        ax[3].invert_xaxis(); ax[3].grid(alpha=0.3)
    else:
        ax[3].axis("off")

    for a in ax[:2]:
        a.axis("off")
    fig.suptitle(f"{label.upper()}  -  {note}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("Datasets/idd-lite/idd20k_lite"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("road_perception/checkpoints/fastscnn_iddlite.pt"))
    ap.add_argument("--results", type=Path, default=Path("docs/phase3c_results.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    ap.add_argument("--speed", type=float, default=8.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ds = IDDLite(a.root, "val", augment=False)
    names = [ds._item(i).image.name for i in range(len(ds))]
    bridge = PerceptionPlannerBridge.from_checkpoint(a.checkpoint)
    cfg, params, profiles = AutonomyConfig.load(), load_vehicle_parameters(), ObjectProfiles.load()

    data = json.loads(a.results.read_text(encoding="utf-8"))
    geo = [g for g in data["geometry"] if g.get("left_rmse_m") is not None]
    geo.sort(key=lambda g: max(g["left_rmse_m"], g["right_rmse_m"]))
    rejected = [e for e in data["episodes"] if e["mode"] == "B" and not e["accepted"]]

    picks = []
    if geo:
        best = geo[0]
        picks.append((names.index(best["image"]), "normal",
                      f"closest agreement, boundary error "
                      f"{max(best['left_rmse_m'], best['right_rmse_m']):.2f} m"))
        worst = geo[-1]
        picks.append((names.index(worst["image"]), "difficult",
                      f"worst accepted case, boundary error "
                      f"{max(worst['left_rmse_m'], worst['right_rmse_m']):.2f} m"))
    if rejected:
        r = rejected[0]
        picks.append((names.index(r["image"]), "rejected",
                      f"safety gate refused it: {r['gate_code']}"))

    for index, label, note in picks:
        figure(bridge, ds, index, label, note, a.out / f"p3c_{label}.png",
               cfg, params, profiles, a.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
