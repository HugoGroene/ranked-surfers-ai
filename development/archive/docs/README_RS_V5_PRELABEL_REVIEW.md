# V5: Model-assisted polygon labeling with LabelMe

**Goal:** Manually label a smaller diverse seed set, train YOLO-seg, let the model propose polygons on remaining native-resolution screenshots, correct proposals in LabelMe, then retrain from the entire corrected dataset.

This workflow **does not touch your working V4 controller** and avoids training on unchecked machine-generated labels.

## 1. Seed data: hand-label a first set

Aim to hand-label roughly 80–120 **diverse** frames across at least two independent native capture sessions, keeping an additional independent session for genuine validation. Start with 20 if you want to test annotation/export first. Include coins, front/side/roof/ramp train surfaces, each confirmed barrier type, and distant/partly occluded objects. Label **all relevant visible instances** in each chosen frame; don't leave coin chains half annotated. If a rare class isn't present enough, collect targeted frames; a given number of screenshots alone doesn't guarantee coverage.

Use `rs_v5_export_labelme.py` to convert the *manually reviewed* LabelMe `.json` files to `segment` and `detect` datasets. Its validation splitter keeps recording sessions separate. Only train the segment model once the exported data is sensible.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

.\.venv\Scripts\python.exe -u rs_v5_export_labelme.py `
  --output ai_data\v5_hazards\export_seed

.\.venv\Scripts\yolo.exe segment train `
  model=yolo11n-seg.pt `
  data=ai_data/v5_hazards/export_seed/segment/data.yaml `
  imgsz=960 batch=2 epochs=40 device=0 workers=2 `
  project=ai_data/v5_hazards/runs name=segment_seed
```

The checkpoint should be at `ai_data\v5_hazards\runs\segment_seed\weights\best.pt`. Check that it exists with `Test-Path`. Different Ultralytics versions may generate a numbered suffix if an existing run name is reused.

## 2. Have YOLO propose labels for 40 unlabeled frames

`rs_v5_prelabel_review.py` runs the trained **segmentation** model and converts each predicted mask to editable LabelMe polygon points. Small/degenerate masks become rectangles. It hardlinks or copies original images to a **separate review batch**; it does not change the raw images or manual annotations.

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review.py propose `
  --session v5_capture_01 `
  --batch capture01_batch01 `
  --model ai_data\v5_hazards\runs\segment_seed\weights\best.pt `
  --max-images 40 `
  --conf 0.20 `
  --imgsz 960 `
  --device 0
```

Use a **new** `--batch` name every time. The script skips images that already have manual annotations in `ai_data\v5_hazards\raw\v5_capture_01`. If the first 40 unlabeled images are too repetitive, use `--start-index 100` to skip ahead. Decreasing `--conf` tends to include more weak detections, which helps catch small/distant coins but creates more false positives to delete.

## 3. Edit polygons in LabelMe

Open the generated **review folder**, not the raw folder:

```powershell
.\.venv-labelme\Scripts\labelme.exe `
  ai_data\v5_hazards\review_batches\capture01_batch01 `
  --labels rs_v5_labels.txt
```

Inspect **every image** in the batch. Fix outlines, correct class names, delete false positives, **add all missed coins/barriers/trains**, and save the JSON. For confirmed barrier movement classes, use `barrier_jump`, `barrier_roll`, or `barrier_either`; don't guess. You can use rectangles/circles when faster. Don't assume YOLO detected every instance or that an empty proposal means an empty scene.

A trained segmentation model does not know safe contacts or time-to-impact. PPO and later motion modeling must learn those.

## 4. Explicitly import your corrected annotations

**Only after reviewing all 40 images**, run:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review.py accept `
  --batch capture01_batch01 `
  --confirm-reviewed
```

The script checks labels and image dimensions, refuses to overwrite existing manually labeled images, and copies reviewed JSON files to `ai_data\v5_hazards\raw\v5_capture_01` beside the original photos. **Do not use `--confirm-reviewed` until each frame has been checked**: this is a safety acknowledgement, not an automatic audit. The raw folder stays unchanged before acceptance; unreviewed model guesses are never included by `rs_v5_export_labelme.py`.

Create another review batch for the next 40 unlabeled frames, ideally drawing from other capture sessions and rare situations.

## 5. Retrain incrementally

Re-export the WHOLE reviewed dataset to a **new** export folder (the exporter refuses overwrites):

```powershell
.\.venv\Scripts\python.exe -u rs_v5_export_labelme.py `
  --output ai_data\v5_hazards\export_round02

.\.venv\Scripts\yolo.exe segment train `
  model=ai_data/v5_hazards/runs/segment_seed/weights/best.pt `
  data=ai_data/v5_hazards/export_round02/segment/data.yaml `
  imgsz=960 batch=2 epochs=30 device=0 workers=2 `
  project=ai_data/v5_hazards/runs name=segment_round02
```

This fine-tunes from the seed checkpoint **on old + newly reviewed screenshots together**, reducing forgetting. Next round, pass `segment_round02/weights/best.pt` to `propose`.

**Validation:** Hold out whole recording sessions, not random adjacent frames. For rigorous comparisons, keep a dedicated fully manual validation set unchanged across rounds; don't accept model-generated annotations into it or repeatedly tune based on its frames.

**Notes:** The source script expects a YOLO **segmentation** checkpoint using exactly the nine labels in `rs_v5_labels.txt`. The old 3-class V4 hazard model is not compatible with these labels. Automated predictions are suggestions, not ground truth. The V4 PPO hazard feature encoder also still needs a dedicated V5 update before a V5 checkpoint can be used live.
