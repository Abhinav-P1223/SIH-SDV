# Phase 7 — small-data Indian-scene fine-tuning of the Phase 6 detector

    UVH-26 (750 images) -> fine-tune the EXISTING Faster R-CNN MobileNetV3-320-FPN
                        -> Detection -> the EXISTING tracker, fusion, prediction, risk, planner

The architecture is unchanged. The only structural edit is the final box predictor, which had to go
from COCO's 91 outputs to our 7, because a head that predicts 91 COCO classes cannot predict an
auto-rickshaw. Nothing under `autonomy/` or `simulation/` was touched.

## 1. What this experiment can and cannot claim

**It can claim:** fine-tuning on a small human-verified Indian traffic dataset improved detection of
selected Indian traffic participants on held-out UVH-26 scenes.

**It cannot claim** improved Indian dashcam perception, or that autonomous-vehicle camera
performance is proven. UVH-26 is elevated fixed-camera CCTV imagery. The held-out test split is
CCTV too, so **CCTV-to-dashcam transfer remains entirely unvalidated.**

**It does not address** pedestrian or animal detection. UVH-26 has neither class, so Phase 6's
pedestrian recall of 0.16 and its unsupported animal class are untouched.

## 2. Setup

| | |
|---|---|
| Data | UVH-26 (AIM @ IISc), CC BY 4.0, human-verified consensus labels |
| Subset | 500 train / 100 validation / 150 test, disjoint, Phase 7B manifest, seed 20260907 |
| Model | Faster R-CNN MobileNetV3-Large-320-FPN, COCO weights, **backbone frozen** |
| Trainable | 16.01 M of 18.96 M parameters |
| Schedule | 3 epochs, SGD, lr 0.002, momentum 0.9, weight decay 5e-4, batch 4 |
| Augmentation | Horizontal flip only |
| Training time | **21.7 minutes on CPU** |

The backbone is frozen deliberately. With 500 images, unfreezing a feature extractor trained on
118,000 is the fastest route to overfitting, and the question here is narrow: does a small amount of
Indian-scene supervision move the head without wrecking what the backbone knows.

## 3. Training behaviour, and the overfitting check

| Epoch | Train loss | Val mAP | Val precision | Val recall |
|---|---|---|---|---|
| 1 | 1.064 | 0.190 | 0.612 | 0.270 |
| 2 | 0.868 | 0.253 | 0.647 | 0.311 |
| 3 | 0.836 | **0.281** | 0.615 | 0.341 |

**No overfitting signal.** Validation mAP rose at every epoch and was still rising when the run
stopped, while training loss fell smoothly. The classic warning sign, a collapsing training loss
beside a stalling validation metric, did not appear. Epoch 3 was selected on validation mAP.

Validation was still improving, which means three epochs was probably too few rather than too many.
That is a finding, not a licence to run more: no further training was launched.

## 4. A/B on the locked 150-image test set

The test split was never loaded by the training script, never used for epoch selection, and never
used to tune a threshold. Both arms were scored by the same function with the same IoU of 0.5, the
same score threshold of 0.35, and the same class-aware greedy matching.

| | COCO baseline | UVH-26 fine-tuned | Delta |
|---|---|---|---|
| Overall precision | 0.536 | **0.621** | **+0.084** |
| Overall recall | 0.226 | **0.379** | **+0.153** |
| mAP at 0.5 | 0.238 | **0.325** | **+0.087** |
| True positives | 353 | 591 | +238 |
| False positives | 305 | 361 | +56 |
| False negatives | 1207 | 969 | -238 |
| Latency | 238 ms | 250 ms | +12 ms |

### Per class

| Class | GT | COCO recall | FT recall | Delta | Verdict |
|---|---|---|---|---|---|
| **Motorcycle** | 754 | 0.151 | **0.379** | **+0.228** | Improved |
| **Auto-rickshaw** | 222 | 0.000 | **0.347** | **+0.347** | Improved |
| Car | 325 | 0.480 | 0.523 | +0.043 | Improved |
| Bus | 76 | 0.447 | 0.355 | -0.092 | **Regressed** |
| Truck | 151 | 0.305 | 0.205 | -0.099 | **Regressed** |
| **Bicycle** | 32 | 0.094 | **0.000** | -0.094 | **Collapsed** |

One asymmetry, stated rather than buried: COCO has no auto-rickshaw class, so the baseline's recall
there is zero **by construction, not by failure**. It made no auto-rickshaw predictions because it
cannot. That gap is precisely what this phase exists to close, and the +0.347 should be read as
"a class became possible", not "a class got better".

## 5. The regressions, which are real

Three classes got worse and one disappeared entirely. This is not a footnote.

**Bicycle collapsed to zero.** The fine-tuned model made **no bicycle predictions at all** on the
test set. With 106 training boxes against 2,419 for motorcycles, a 23-to-1 imbalance, the head
learned to spend its capacity elsewhere. This is exactly the rare-class collapse the phase brief
warned about, and it happened.

**Bus and truck lost about 0.1 recall each.** Both are also comparatively rare, at 247 and 509
training boxes. The fine-tuned model became more conservative on them: truck predictions fell from
126 to 64, so it is missing them rather than confusing them.

**Overall this is still a net gain**, with recall up 68% in relative terms and mAP up 37%, and
precision improved rather than traded away. But a system that detects more motorcycles and fewer
bicycles is not uniformly better, and anyone quoting the headline mAP should be shown this table
alongside it.

## 6. Integration

The fine-tuned model is a drop-in replacement. It is loaded through the same
`CameraObjectDetector` by setting one config field, emits the same `RawDetection` objects, and
those become the same `Detection` objects through the same monocular geometry. Tests drive the
whole path into the unmodified `SensorFusionTracker` and assert finite track states.

Setting no checkpoint leaves the Phase 6 behaviour byte for byte identical, which is what the A/B
baseline arm depends on.

## 7. Reproducibility

The checkpoint records its config, seed, class names, epoch, git commit and manifest path. The
dataset is gitignored and is not redistributed, per the CC BY 4.0 terms which permit publishing a
model trained on it as an abstract derivative work. Attribution: UVH-26, AIM @ IISc.

## 8. Known limitations

**CCTV, not dashcam**, as stated in section 1. This is the single largest caveat and no result here
escapes it.

**No pedestrian, no animal.** Phase 6's two other weaknesses are untouched.

**Bicycle is now worse than useless** in the fine-tuned model, at exactly zero predictions.

**Three epochs on 500 images** is a small experiment by construction. Validation was still improving
when it stopped.

**Latency is unchanged and still far outside the planning budget**, at 250 ms against a 100 ms
planning period. Fine-tuning changed what the detector sees, not how fast it sees it.


---

# Phase 7D/7E — diagnosing the regression, and testing class-aware oversampling

## 9. Why bicycle, bus and truck got worse (7D)

**Bicycle was never absent, it was never confident.** The uniform model emits 73 bicycle boxes
above 0.01, but its **maximum bicycle score across the whole test set is 0.181** against a 0.35
cut. Every other class reaches 0.90 or more: car 0.997, motorcycle 0.990, bus 0.982,
auto-rickshaw 0.943, truck 0.902. Lowering the threshold to 0.10 still only reaches 0.12 recall on
validation, so it is genuinely under-learned rather than merely mis-calibrated.

**Where the missed objects went**, for the uniform model:

| | Missed | Largest cause | Confused with |
|---|---|---|---|
| Bicycle | 32 | overlaps something else 41% | motorcycle, all 9 wrong-class cases |
| Bus | 49 | wrong class 39% | car 8, auto-rickshaw 7, truck 4 |
| Truck | 120 | low confidence 30% | auto-rickshaw 14, car 10 |

Missed objects are overwhelmingly **large** (bicycle 19 of 32, bus 39 of 49, truck 99 of 120), so
this is not a small-object failure.

**This is not catastrophic forgetting**, and the evidence is specific. The backbone was frozen, so
the shared representation could not drift, and the box predictor was replaced with a randomly
initialised seven-class head, so there was nothing in it to forget. The mechanism is under-training
of rare classes in a fresh head under a 23-to-1 imbalance.

## 10. The oversampling experiment (7E)

One change from Phase 7C: a deterministic weighted image sampler. Same 750 images, same splits,
same manifest, same architecture, same frozen backbone, same optimiser, learning rate, epochs,
augmentation and seed.

**Formula.** Class weight is inverse box frequency normalised to the most common class, capped.
An image takes the **maximum** weight over the classes it contains, because the rare class is what
makes the image worth drawing. The sampling unit is the image, never the box.

**The cap exposed a ceiling that bounds the whole experiment.** Measured before training:

| Cap | Bicycle box exposure | Bicycle-image share of an epoch |
|---|---|---|
| 8 | 1.33x | 26.9% |
| **16 (used)** | **1.95x** | **39.4%** |
| 25 (uncapped) | 2.38x | 48.2% |

**Even uncapped, bicycle boxes only become 2.4x more frequent.** The 101 bicycle images hold 106
bicycles between them, about one each, alongside crowds of motorcycles. Image-level oversampling
can raise how often a bicycle is seen but cannot change what arrives with it. That is a property of
the data, not of the sampler.

Realised exposure at cap 16: motorcycle 1.03x, car 1.02x, auto-rickshaw 1.12x, bus 1.25x,
bicycle 1.95x, and truck **0.92x**, which dips because bicycle and bus frames crowd it out.

### Checkpoint selection was changed, and it mattered

Ranking epochs on mAP alone selected epoch 2, where **bicycle validation recall is still exactly
0.000**. That optimises the metric this phase cares least about. The rule is now: keep every epoch
within 0.02 mAP of the best, then take the highest bicycle recall, breaking ties on mAP. It selects
epoch 3, trading 0.010 mAP for bicycle recall 0.000 to 0.120. Both arms are epoch 3, so the
comparison below differs only in the sampler.

| Epoch | mAP | Bicycle val recall | |
|---|---|---|---|
| 1 | 0.187 | 0.000 | |
| 2 | 0.267 | 0.000 | best mAP |
| 3 | 0.257 | **0.120** | **selected** |

## 11. Three-way result on the locked 150-image test set

| | COCO | FT-Uniform | FT-Oversampled |
|---|---|---|---|
| Precision | 0.536 | 0.621 | **0.630** |
| Recall | 0.226 | **0.379** | 0.371 |
| mAP | 0.238 | **0.325** | 0.282 |
| Motorcycle recall | 0.151 | **0.379** | 0.359 |
| Auto-rickshaw recall | 0.000 | **0.347** | 0.320 |
| Car recall | 0.480 | **0.523** | 0.498 |
| Bus recall | **0.447** | 0.355 | 0.368 |
| Truck recall | **0.305** | 0.205 | 0.291 |
| Bicycle recall | **0.094** | 0.000 | 0.062 |
| Latency | 255 ms | 287 ms | 270 ms |

True positives, which is what the small classes actually turn on:

| Class | GT | COCO TP | Uniform TP | Oversampled TP |
|---|---|---|---|---|
| Motorcycle | 754 | 114 | 286 | 271 |
| Auto-rickshaw | 222 | 0 | 77 | 71 |
| Car | 325 | 156 | 170 | 162 |
| Truck | 151 | 46 | 31 | **44** |
| Bus | 76 | 34 | 27 | 28 |
| Bicycle | 32 | 3 | 0 | **2** |

## 12. What the evidence actually supports

**Strong evidence.** Motorcycle and auto-rickshaw gains over COCO are real and large, on 754 and
222 ground-truth boxes. Both survive oversampling with a small cost.

**Directional evidence.** Truck genuinely recovers under oversampling, 31 to 44 true positives on
151 boxes, back to roughly the COCO level. That is the clearest win of this experiment.

**Inconclusive.** Bicycle moves from 0 to 2 true positives out of 32. Two detections cannot support
a claim either way, and the oversampled model is **still below the COCO baseline's 3**. Bus moves
27 to 28 on 76 boxes, which is noise.

**The honest verdict: oversampling did not solve bicycle.** It recovered truck, improved precision
slightly, and cost 0.043 mAP along with small amounts of motorcycle, auto-rickshaw and car. The
ceiling analysis explains why: 106 training boxes cannot be repaired by drawing the same 101 images
more often.

## 13. Recommendation

**Keep the Phase 7C uniform-sampling model as the deliverable.** It has the better mAP (0.325
against 0.282) and the better motorcycle and auto-rickshaw recall, which are the classes this
project actually needs.

Prefer the oversampled model only if truck recall matters more than mAP, where it is clearly
better, 0.291 against 0.205.

**Bicycle is data-limited, not method-limited.** No sampling scheme fixes 106 boxes. Fixing it
needs more bicycle images, which means more data, which is outside this phase's scope and was
explicitly ruled out.
