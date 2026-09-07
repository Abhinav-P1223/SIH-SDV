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
