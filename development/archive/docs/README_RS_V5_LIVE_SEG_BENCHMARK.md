# V5.4 segmentation: screenshots, live latency, and spatial recall

This package **does not overwrite** the working V4 controller or models. It uses
`ai_data/v5_hazards/runs/segment_round04/weights/best.pt` (nine-class `segment_round04`)
by default and your existing `rs_v4_screen_states.py` classifier for MENU / LOADING /
GAMEPLAY detection.

**Important distinction:** This is a **shadow** segmentation benchmark: the nine-class
YOLO model runs on the same high-resolution native game crops while PPO still runs its
original 512×288 four-frame observation and update. The nine-class YOLO output is logged
and overlaid, but is **not** passed into the old three-class PPO auxiliary vector. That
auxiliary hazard subvector is deliberately zeroed rather than incorrectly classifying a
train roof as an impassable train. V5-to-PPO observation integration comes *after* this
benchmark. `--benchmark-action jump` forces jump-only gameplay so PPO does not drive
into trains beside the starting lane. It is **NOT meaningful PPO training**; no policy
checkpoint is saved.

## 0 — Installation / prerequisites

Extract the ZIP in the repository root (Windows PowerShell):

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
Expand-Archive "$HOME\Downloads\rs_v5_live_segmentation_package.zip" -DestinationPath . -Force
Test-Path "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
Test-Path "ai_data\v4_screen_states\screen_state_best.pt"
```

Both `Test-Path` values should be `True`. Keep the `.venv` with CUDA-enabled
PyTorch, Ultralytics, Playwright, OpenCV and SB3 active. Local game only.

## A — See V5.4 predictions on 12 saved images (no live game)

```powershell
.\.venv\Scripts\python.exe -u rs_v5_preview_predictions.py `
  --model ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --source ai_data\v5_hazards\raw\v5_capture_04 `
  --sample 12 --seed 42 --imgsz 960 --conf 0.20 --device 0 `
  --output ai_data\v5_hazards\preview_round04

Start-Process "ai_data\v5_hazards\preview_round04"
```

Outputs `_original.jpg` + `_masks.jpg` pairs and `preview_report.json`. The
12 images are sampled across the recording in **different time blocks**, not just
the first 12 frames. It never edits source screenshots or labels.

To inspect predictions on exactly the 20 **validation** images (not the final test):

```powershell
.\.venv\Scripts\python.exe -u rs_v5_preview_predictions.py `
  --source ai_data\v5_hazards\export_round04_fixed\segment\images\val `
  --sample 20 --output ai_data\v5_hazards\preview_validation_v54
```

**Avoid looking at the 32-image `test` set repeatedly to choose checkpoints**. Leave
that test holdout until you have made your choice.

## B — Optional measured recall by image region on annotated validation

The visual preview alone cannot tell how many hazards were missed. This separate
script uses the polygon-derived **ground truth boxes** to count detections by
image-height region with one-to-one same-class box IoU matching. Default IoU=0.50.
It is **not mask mAP**, physical distance, or a calibrated danger estimate.

```powershell
.\.venv\Scripts\python.exe -u rs_v5_validation_by_region.py `
  --dataset ai_data\v5_hazards\export_round04_fixed\segment `
  --split val `
  --model ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --iou 0.50 --device 0
```

It writes `ai_data/v5_hazards/v54_val_by_region.json` and prints coin recall in
upper / middle / lower image regions. **Lower image region does not automatically
mean an imminent collision**; perspective, ramps, roof and camera position matter.
Only 20 val images are available and many barrier categories are rare.

## C — Live full pipeline benchmark, V5.4 asynchronous segmentation

In another PowerShell window, start local Practice server:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory .\rankedsurfers-downloaded
```

Then, in the project's other window:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_segmentation_live_benchmark.py live `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --crop 0 0.055 1 0.965 `
  --steps 256 --max-minutes 5 --n-steps 128 --batch-size 32 --n-epochs 2 `
  --capture-mode fast --resize-method linear --torch-threads 1 `
  --benchmark-action jump --press-delay-ms 5 `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --state-confidence 0.90 --menu-ready-timeout 30 `
  --startup-timeout 8 --startup-grace 0.10 --restart-grace 0.10 `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 `
  --save-overlays 8 --preview-delay 0.20 `
  --run-name v54_shadow_seg_jump5 `
  2>&1 | Tee-Object -FilePath ai_data\v54_shadow_seg_jump5.log
```

The script uses the known working V4 state classifier and restart controller.
It measures real PPO inference and PPO updates, but **all actions are overridden
to JUMP**. Jump is not guaranteed to prevent every death: it merely prevents PPO
from arbitrarily switching into trains beside the start.

Mask overlays (native source *paired with its own predictions*):

```powershell
Start-Process "ai_data\rl_v2_runs\v54_shadow_seg_jump5\matched_detection_previews"
```

Saved report:
`ai_data/rl_v2_runs/v54_shadow_seg_jump5/v4_low_latency_report.json`

## D — Summarize the result in PowerShell

```powershell
$r = Get-Content "ai_data\rl_v2_runs\v54_shadow_seg_jump5\v4_low_latency_report.json" -Raw | ConvertFrom-Json
$g = $r.gameplay_only_decision_intervals_ms
$v = $r.async_yolo.v5_segmentation

[pscustomobject]@{
  Complete       = $r.complete
  GameplayHz     = if ($g.median_ms) { [math]::Round(1000/$g.median_ms,1) } else { $null }
  GameplayP95ms  = $g.p95_ms
  GameplayP99ms  = $g.p99_ms
  YOLOms         = $r.async_yolo.inference_ms.median
  YOLOAgeMs      = $r.async_yolo.age_at_decision_ms.median
  YOLOFresh      = $r.async_yolo.fresh_fraction
  ModelPredictMs = $v.model_predict_wall_ms.median_ms
  RGBtoBGRms     = $v.input_rgb_to_bgr_ms.median_ms
  MaskDecodeMs   = $v.mask_and_box_extract_ms.median_ms
  GuardFailures  = $r.startup_guard_failures
  SavedOverlays  = $r.matched_overlays
} | Format-List

$r.async_yolo.detected_objects
$v.detections_by_vertical_image_region | Format-List *
```

Note: `YOLOms` is a **separate asynchronous worker**, not an amount added
serially to each PPO step. Frame age includes detector submission frequency,
queue timing and screenshot age. The original V4 speed result with old three-
class YOLO was 31.4 Hz on the user's Windows PC; **compare measured rates**, do
not assume V5.4 has the same performance.

For an apples-to-apples **no-YOLO overhead control** of this new script,
rerun the same live command with `--no-yolo`, `--save-overlays 0`, and a new
`--run-name v54_no_yolo_jump5`. You can then distinguish YOLO contention from
base browser/input/capture timing. Use the same steps, crop, resolution, and
keypress duration.

## Accuracy / interpretation limits

* Live upper/middle/lower class **counts are NOT accuracy metrics**: without human
  ground truth, a detector may miss objects completely. Use the independent
  labeled validation metric and qualitative overlays.
* Segmentation mask extraction and RGB→BGR conversion are timed separately.
* Only benchmark in the local Practice copy; don't run the tool on live ranked servers.
* The 5 ms keypress's actual Unity event registration still needs verification.
* If `segment_round04` is not a nine-class segmentation checkpoint, the live
  benchmark fails early rather than silently throwing away train roofs/ramps.
* Do not resume or continue PPO training from a fixed-JUMP performance run.
