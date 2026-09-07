# Phase 6 — real camera object detection into the existing pipeline

    camera image -> COCO-pretrained detector -> boxes -> monocular geometry
                 -> Detection -> EXISTING tracker -> EXISTING fusion / prediction / risk / planner

**Nothing under `autonomy/` or `simulation/` was modified.** The detector emits `Detection`
objects, which is exactly what the simulated camera already emits, so the rest of the stack cannot
tell where they came from.

## 1. What this is, and what it is not

**It is** a genuine learned detector running real inference on real photographs, feeding the real
tracker.

**It is not our detector.** The weights are torchvision's COCO-trained Faster R-CNN
MobileNetV3-Large-320-FPN. We did not train them, and the network has never seen an Indian road.
Every accuracy number below is a number about a COCO detector evaluated out of domain.

**It cannot see auto-rickshaws or pushcarts.** COCO has no equivalent class. Those are two of the
most characteristic objects on an Indian road and the detector is blind to both. No threshold
change fixes that; it needs Indian training data.

## 2. Why we did not train our own

Measured in the Phase 6 audit rather than assumed.

**IDD-Lite cannot supply detection labels at all.** It ships `_inst_label.png` files, which look
like instance masks and are not: they repeat the same seven semantic ids, one blob per class. A
derived box has a median size of 310x107 px on a 320x227 image, in other words most of the frame.
Falling back to connected components gives 5.3 vehicle and 3.4 living-thing blobs per image, but
they merge touching objects, which in dense traffic is constant, and they carry only two coarse
classes.

**nuScenes-mini has the volume but not the diversity.** Its 18,538 annotations are only **911
distinct physical objects** across ten Boston and Singapore scenes, re-annotated at 2 Hz.

| ObjectType | Unique instances in nuScenes-mini |
|---|---|
| Car | 390 |
| Pedestrian | 213 |
| Truck | 28, marginal |
| Motorcycle | 20, too few |
| Bicycle / Bus | 15 each, too few |
| Auto-rickshaw / Cattle | **none** |
| Pushcart | 3 |

**And the cost is real.** Training measured 10.3 minutes per epoch on this CPU, so 40 epochs is
6.9 hours, to fit two usable classes from the wrong continent. A COCO detector reaches seven of our
nine object types today and can be measured against nuScenes projections it also never trained on.
That is the more honest result.

## 3. Which detector, and why the audit's recommendation was overridden

The audit recommended SSDLite320 on size and speed. Measured on 8 real nuScenes front-camera images
carrying 147 ground-truth boxes in our classes, it was not usable:

| Model | COCO mAP | CPU ms | Detections at 0.35 | at 0.2 |
|---|---|---|---|---|
| SSDLite320 MobileNetV3 | 21.3 | 226 | **4** | 18 |
| **Faster R-CNN MobileNetV3-320-FPN** | 22.8 | 263 | **31** | 72 |

SSDLite's score distribution was so flat its 90th percentile sat at 0.14. Eight times the recall
for 16% more time is not a close decision, so the recommendation was overridden by measurement.
Both are torchvision, BSD-3-Clause, and neither adds a dependency. `model_name` still accepts
SSDLite so the comparison can be rerun.

## 4. The hard part is range, not classification

A monocular box gives **bearing** almost for free: the horizontal box centre maps through the
intrinsics to an angle good to a fraction of a degree.

Range has no such answer. A single image contains no depth, so two estimators are used and both are
assumptions:

- **Ground contact.** Assume the box bottom is where the object meets a flat road, and invert the
  depression angle. Accurate when it holds, badly wrong when the feet are occluded or the road
  slopes.
- **Size prior.** Assume the object is typical for its class and invert the projection. Robust to
  occluded feet, but it inherits the real spread of object sizes, and a misclassification becomes a
  range error.

Both run, and **their disagreement is folded into the reported variance**. Two methods that agree
produce a tight estimate; two that argue produce a wide one. That matters more than the estimate
itself, because a Kalman filter is far more sensitive to a confidently wrong covariance than to a
noisy measurement. The result is range-dominated and large, which is the same shape the simulated
camera already produces, which is why nothing downstream needed changing.

## 5. Results

`scripts/eval_detector.py`, all 404 CAM_FRONT keyframes, IoU 0.5, score threshold 0.35,
class-aware matching. Stored in `docs/phase6_results.json`.

### 5.1 Detection quality on nuScenes-mini

| Measure | Value |
|---|---|
| Images | 404 |
| True positives / false positives / misses | 844 / 988 / 2119 |
| Precision | 0.461 |
| Recall | 0.285 |
| mAP at IoU 0.5 | 0.148 |
| Inference | 173 ms mean, 228 ms p95 |

| Class | Ground truth | Predictions | TP | Precision | Recall | AP |
|---|---|---|---|---|---|---|
| Car | 1685 | 1342 | 629 | 0.47 | 0.37 | 0.31 |
| Pedestrian | 801 | 249 | 129 | 0.52 | 0.16 | 0.13 |
| Bus | 174 | 76 | 38 | 0.50 | 0.22 | 0.18 |
| Truck | 134 | 142 | 44 | 0.31 | 0.33 | 0.23 |
| Motorcycle | 107 | 13 | 2 | 0.15 | 0.02 | 0.01 |
| Bicycle | 62 | 10 | 2 | 0.20 | 0.03 | 0.02 |

### 5.2 The finding that matters: recall collapses with range

| Ground-truth range | Recall | Boxes |
|---|---|---|
| 0 to 15 m | **0.580** | 383 |
| 15 to 30 m | 0.422 | 1044 |
| 30 to 50 m | 0.184 | 804 |
| beyond 50 m | **0.045** | 732 |

The aggregate recall of 0.285 is not a useful number on its own, and this table is why. The
detector finds most of what is close enough to matter for an emergency stop and almost nothing
beyond 50 m, because a 320 px input leaves a distant car a few pixels wide. Half the ground-truth
boxes in this split sit beyond 30 m, so the aggregate is dominated by the range band where the
detector is weakest. **Any claim about this detector has to be a claim about a range band.**

Two-wheelers are the other clear failure: motorcycle recall 0.02 and bicycle 0.03. On an Indian
road that is the worst possible class to miss.

### 5.3 Integration with the shipped tracker

Detections go into the unmodified `SensorFusionTracker`. Nothing in `autonomy/` was changed, which
`git status` confirms: every Phase 6 file is new.

| Measure | Value |
|---|---|
| Frames | 50 |
| Detections produced | 227, 4.54 per frame |
| Confirmed tracks, mean | 3.14 |
| Box to `Detection` conversion | 0.39 ms per frame |

The conversion cost is negligible next to the 173 ms of inference. The track count is well below
the detection count, partly through correct association across frames and partly through the
monocular merging described in section 6.

### 5.4 Real Indian-road images

Qualitative only. IDD-Lite has no detection ground truth, so these are counts and not accuracy.

| Measure | Value |
|---|---|
| Images | 100 |
| Detections per image | 4.07 |
| Images with no detection | 17 |
| By class | car 151, pedestrian 112, motorcycle 65, truck 46, bus 29, bicycle 4 |
| Inference | 162 ms mean |

The detector does fire on real Indian streets, and notably finds 65 motorcycles here despite
scoring 0.02 recall on nuScenes motorcycles, which says more about the size of nuScenes motorcycle
boxes than about the detector. Seventeen images in a hundred yield nothing at all.

## 6. Known limitations

**Recall collapses with range**, which is the headline limitation and is quantified in section 5.
A 320 px input destroys distant objects.

**Monocular detections merge in the tracker.** Range uncertainty is metres, so two genuinely
distinct objects at similar bearing and similar apparent size land inside each other's gate and
become one track. That is the tracker behaving correctly given what one camera can tell it, and it
is why this phase does not claim camera-only object tracking. A test pins the behaviour rather than
hiding it.

**Two Indian classes are undetectable**, as stated in section 1.

**Latency is far outside the planning budget.** Inference is roughly 260 ms per image against a
100 ms planning period, and Phase 3B's segmentation already costs 27.5 ms. A camera-only perception
front end at this speed is an offline evidence pipeline, not a real-time one.

**The IDD-Lite numbers are qualitative and are labelled as such.** That dataset has no detection
ground truth, so detection counts there are counts, never accuracy.

**The evaluation is out of domain in both directions.** The detector trained on COCO, was scored on
nuScenes, and was demonstrated on IDD-Lite. No number here describes a detector trained on the data
it was tested on, which is the honest position but also a weak one.
