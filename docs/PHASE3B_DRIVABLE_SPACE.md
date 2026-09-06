# Phase 3B — learned drivable-space perception from IDD-Lite

    Indian-road camera image -> Fast-SCNN -> 7-class mask -> drivable mask -> corridor -> planner

Nothing in `autonomy/` was modified. The corridor reaches the planner through `DrivableSpace`,
the same contract the simulator already satisfies.

## 1. Dataset, from the audited files

Measured by `scripts/audit_datasets.py` on the extracted 28.2 MB archive, not quoted from
documentation.

| Property | Value |
|---|---|
| Images | 2,011: train 1,403, val 204, test 404 |
| Labels | 1,607 semantic masks, train and val only. **Test is unlabelled** |
| Resolution | 320 x 227 RGB, uint8 masks |
| Classes | 7 level-1 ids, 255 = ignore |
| Class balance | drivable 32.4%, far objects 26.0%, sky 18.9%, road-side 11.2%, vehicles 8.1%, **non-drivable 2.2%**, living things 1.3% |

Level 1 folds road and drivable-fallback into **drivable**, and sidewalk, curb and
non-drivable-fallback into **non-drivable**. There is therefore no separate curb class to learn.
For a corridor estimate that abstraction is the right one; for curb-level geometry it is not.

The test split is used for qualitative figures only. It has no masks, so it cannot be trained or
scored on, and the loader returns `255` for it so a mistake would be loud rather than silent.

## 2. Model selection

**Fast-SCNN**, implemented from the published design rather than vendored, so no licence is
inherited. Measured on this machine, CPU only, no GPU:

| Candidate | Params | Inference | Verdict |
|---|---|---|---|
| **Fast-SCNN (chosen)** | 0.76 M | 41 ms, 24.6 FPS | Two-branch: cheap detail path plus a deep context path with pyramid pooling. Designed for exactly this regime |
| Fast-SCNN width 0.5 | 0.18 M | 7 ms, 139 FPS | Kept as an option; unnecessary since width 1.0 already clears the budget |
| BiSeNet / BiSeNetV2 | ~5-13 M | slower | Heavier for no gain at 320 x 224 on 1,403 images |
| MobileNet-backbone | ~2-3 M | moderate | Needs a pretrained-weights download; training from scratch removes that dependency |
| Segmentation transformer | tens of M | far slower | Absurd for this data volume on CPU |

24.6 FPS against a 10 Hz planning budget leaves ample headroom, so the choice was correctness and
simplicity rather than squeezing the last millisecond.

## 3. Loss, and why not plain cross-entropy

Non-drivable is **2.2% of pixels**. Unweighted cross-entropy converges to predicting drivable
almost everywhere: high pixel accuracy, useless corridor. Two terms address that:

**Weighted cross-entropy.** Inverse-frequency class weights computed from the training split and
capped at 12, so rare classes carry gradient without one class dominating a 0.76 M-parameter
model. Computed weights: drivable 0.14, non-drivable 2.05, living things 3.44, vehicles 0.56,
road-side 0.40, far 0.17, sky 0.24.

**Boundary weighting.** A pixel adjacent to a drivable/non-drivable transition counts 3x. This is
the term that matters: the only thing a planner consumes from this model is **where the drivable
region ends**, and an error there costs more than an error deep inside a region.

No Dice or Tversky term. With boundary weighting in place it added complexity without a
measurable gain, and an unnecessary term is a liability.

A test enforces the intent: an error placed **at** the edge must cost more than the identical
error placed in the interior, and under unweighted loss the two must be equal.

## 4. Training configuration

| Setting | Value |
|---|---|
| Dataset | `Datasets/idd-lite/idd20k_lite` |
| Splits | train 1,403 / val 204 / test 404 (test never used for fitting) |
| Model | Fast-SCNN, width 1.0, 0.76 M parameters |
| Input | 320 x 224 (nearest multiple of 32 to the native 227) |
| Batch size | 8 |
| Epochs | 40 |
| Optimiser | AdamW, lr 3e-3, weight decay 1e-4 |
| Schedule | Cosine annealing |
| Augmentation | Horizontal flip only |
| Loss | Weighted CE (cap 12) + 3x boundary weighting |
| Seed | 1234 |
| Selection | Best validation **drivable IoU**, not loss and not pixel accuracy |
| Checkpoint | `road_perception/checkpoints/fastscnn_iddlite.pt` |

Augmentation is deliberately minimal. A flipped Indian road scene is still a plausible scene;
colour or geometric distortion on 1,403 images would misrepresent a small dataset.

## 5. Metrics, defined

Pixel accuracy is reported **last and never alone**. In order of what matters here:

1. **Drivable IoU** — can the road be found at all.
2. **Boundary F1** — is the edge in the right place.
3. **Inference latency** — does it fit the planning budget.

**Boundary F1, exactly.** Both maps are reduced to the binary question "is this pixel drivable".
A boundary pixel is one whose binary label differs from a 4-neighbour. Precision is the fraction
of predicted boundary pixels within 2 px of any ground-truth boundary pixel, by Euclidean
distance transform; recall is the converse; F1 is their harmonic mean. At 320 x 224 a 2 px
tolerance is about 0.6% of image width. Both maps empty scores 1.0; one empty scores 0.0.

Also reported: mean IoU, per-class IoU, drivable precision and recall, and the full confusion
matrix.

## 6. Drivable-space extraction

Deterministic post-processing, four inspectable steps in `road_perception/drivable.py`:

1. **Binary mask** from predicted class 0.
2. **Clean.** Keep the largest connected component, then binary-close small holes. Justification:
   detached patches of road cannot be reached without crossing non-drivable ground, and holes are
   usually objects *on* the road, which the planner already receives as tracked objects rather
   than as missing road.
3. **Row scan.** From the bottom upward, take the widest contiguous drivable run nearest the
   previous row's centre, so a side road does not capture the corridor. The scan **stops where
   the evidence stops** rather than extrapolating.
4. **Ground projection.** Flat-road pinhole projection to metres, then emit reference, left and
   right polylines.

## 7. Integration boundary — stated precisely

**Real:** the image is a real Indian-road photograph, the segmentation is a real learned model
running real inference, and the corridor is built from that prediction and handed over as
`DrivableSpace`, the planner's own type.

**Not real:** IDD-Lite is a set of still photographs with no ego motion, no calibration and no
dynamic objects. It cannot be driven through. Closing the simulator loop on it would require
inventing a vehicle trajectory and a camera pose, which is exactly the fake integration this
phase forbids. The demonstration therefore ends at a planner-compatible corridor built from a
real prediction, compared against a ground-truth corridor on identical terms.

**The scale caveat.** IDD-Lite ships no calibration, so the image-to-ground step uses explicit,
documented assumptions (camera height 1.4 m, 60 degree horizontal field of view, flat road). The
**shape** of the corridor is learned; the **scale** comes from those assumptions. The corridor is
metrically plausible, not metrically calibrated.

## 8. Results

Trained 40 epochs in 77 minutes on CPU. Selected epoch 34 by best validation drivable IoU.
Everything below is the **held-out validation split, 204 images**, never trained on, produced by
`scripts/eval_segmentation.py` and stored in `docs/phase3b_results.json`.

### 8.1 Segmentation

| Metric | Value |
|---|---|
| **Drivable IoU** | **0.877** |
| Drivable precision / recall | 0.981 / 0.891 |
| Mean IoU (7 classes) | 0.566 |
| **Boundary F1 (2 px)** | **0.408** (P 0.483 / R 0.361) |
| Pixel accuracy | 0.808 |

Per-class IoU: drivable 0.877, sky 0.921, far objects 0.650, vehicles 0.572, road-side objects
0.361, non-drivable 0.292, living things 0.287.

The shape of this result is what the class balance predicted. Drivable is found reliably.
Precision 0.981 means almost nothing called road is not road, which is the direction that matters
for safety: the model under-claims rather than over-claims. Recall 0.891 means about a tenth of
the true road is missed, which shows up later as a corridor that is consistently too narrow.

Non-drivable sits at 0.292 IoU on 2.2% of pixels, and its errors are informative rather than
random: 13.2% of non-drivable pixels are called road-side objects, and 6.8% are called drivable.
The first confusion is benign, since both are non-drivable for planning. The second is the one
that matters and it is the smaller of the two.

### 8.2 Latency

| Measure | Value |
|---|---|
| Inference | 27.5 ms mean, 33.3 ms p95 |
| Inference + corridor extraction | 44.1 ms mean per frame |
| Throughput | 36.4 FPS |

CPU only, machine idle. A first measurement taken while training was still running gave 106 ms
and was discarded; it measured contention, not the model. Against a 10 Hz (100 ms) planning
budget the full perception path uses 44%.

### 8.3 Mode A (ground-truth corridor) vs Mode B (predicted corridor)

Both modes run the identical extraction code. The only difference is whether the drivable mask
comes from the ground-truth label or from the network.

| Measure | Value |
|---|---|
| Images | 204 |
| Mode A produced a usable corridor | 204 / 204 |
| **Mode B produced a usable corridor** | **204 / 204** |
| Comparable pairs | 204 |
| Left boundary RMSE | 1.04 m median |
| Right boundary RMSE | 1.12 m median |
| **Corridor width bias** | **-0.75 m median** (p10 -2.36, p90 -0.03) |
| Overlapping range | 33.4 m median |
| Corridor length | 25 rows median, both modes |

Every predicted corridor became a `DrivableSpace`, the planner's own type, with no planner change.

**The width bias is systematic and it is negative in every decile.** The predicted corridor is
narrower than the true one essentially always, by 0.75 m at the median. That follows directly
from drivable recall 0.891: missed road pixels can only shrink a corridor. For a planner this is
the conservative direction, but it is a bias, not noise, and a corridor 0.75 m too narrow on an
already narrow Indian road is a real constraint, not a free safety margin.

### 8.4 Where it fails

Reported because the aggregate numbers hide it:

| Failure | Count |
|---|---|
| Boundary F1 below 0.20 | 17 / 204 images |
| Worse-side boundary RMSE above 3 m | **46 / 204 images** |
| Worst-side RMSE, worst image | 11.6 m |

**Roughly one image in four has a boundary off by more than 3 m on one side.** Boundary F1 has a
median of 0.403 and a p10 of 0.223. This is the honest headline: the model finds the road, and is
much less reliable about exactly where its edge is. Drivable IoU alone would have hidden that,
which is why boundary F1 is reported beside it.

The worst case, `frame2503_image.jpg` at boundary F1 0.003, predicts 16.8% drivable against 23.0%
in the label. It is in `docs/img/seg_worst.png` alongside the median and best cases, chosen by
score rather than by eye so a failure is always in the figure set.

### 8.5 Two defects found by running the evaluation, both fixed

Recorded because both would have produced a confident and wrong report.

**Every corridor was empty, ground truth included.** The row scan started at the bottom image row
and stopped at the first row without drivable pixels. IDD-Lite masks stop two rows short of the
frame bottom, so the scan terminated on its first iteration and returned an empty corridor for
all 204 images. Rows below the first evidence are now skipped rather than treated as the end of
the road; rows after it still terminate the corridor. Two tests cover both halves.

**Boundary F1 was inflated from 0.408 to 0.925.** Training scored the metric on batched
`(B, H, W)` arrays, and the Euclidean distance transform then treated the batch axis as spatial,
letting a boundary pixel in one image satisfy a boundary pixel in the next. The confusion matrix
works on flattened pixels and was never affected, and checkpoint selection uses drivable IoU, so
**the selected weights are exactly the ones the correct criterion would have picked** and no
retraining was needed. `binary_boundary` now rejects a non-2-D array, and `SegmentationMetrics`
splits a batch and scores image by image. The per-epoch boundary column in
`road_perception/checkpoints/history.json` is from the buggy run and is not comparable; the
checkpoint's own `val` block was recomputed, with the original kept as `val_as_trained`.

## 9. Limitations

**Resolution.** IDD-Lite is 320 x 227. The predicted road boundary is **not precise enough for
curb-level geometry**, and no amount of post-processing changes that. A 2 px boundary tolerance is
roughly 0.6% of image width, which at 10 m range is tens of centimetres on the ground.

**No curb class exists.** Level 1 folds sidewalk and curb into non-drivable, so the model cannot
distinguish a kerb from a verge from a wall.

**No calibration.** See the scale caveat above.

**Small dataset, single domain.** 1,403 training images from Hyderabad and Bangalore drives,
trained from scratch. Expect degradation on any other city, weather or camera.

**Boundary placement is the weak result.** 46 of 204 validation images have a corridor boundary
off by more than 3 m on one side, and the predicted corridor is narrower than the true one in
every decile. Section 8.4 gives the distribution. Drivable IoU 0.877 is not the whole story and
should not be quoted alone.

**Still images only.** No temporal smoothing, no ego motion, no closed loop. Temporal filtering
across frames is the obvious next lever on the boundary noise, and this phase does not attempt
it.

**CPU only.** All timings are CPU. A GPU would change the latency figures but nothing else.
