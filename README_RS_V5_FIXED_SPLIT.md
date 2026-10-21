# V5.4: explicit train / validation / test split

The original `rs_v5_export_labelme.py` automatically chooses validation sessions. This caused the `export_round04` dataset to include the intended test session `v5_capture_05` in training. **Do not use that export for V5.4 training.** Your source LabelMe annotations are unchanged and safe.

## Installation

Copy `rs_v5_export_labelme_v2.py` to the project root beside the existing exporter; do not replace the original.

## Export a clean dataset

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

.\.venv\Scripts\python.exe -u rs_v5_export_labelme_v2.py `
  --output ai_data\v5_hazards\export_round04_fixed `
  --val-sessions v5_capture_04 `
  --test-sessions v5_capture_05
```

Expected split based on most recent logs:

- Train: `v5_capture_02` (59) + `v5_capture_03` (28) = **87 images**
- Validation: `v5_capture_04` = **20 images**
- Held-out test: `v5_capture_05` = **32 images**

The `segment/data.yaml` and `detect/data.yaml` include `train`, `val`, and `test` paths. Read the output and `export_manifest.json` to confirm.

## Fine-tune from V5.3 (validation on 04 only)

```powershell
$previousModel = (Resolve-Path "ai_data\v5_hazards\runs\segment_round03\weights\best.pt").Path
$yaml = (Resolve-Path "ai_data\v5_hazards\export_round04_fixed\segment\data.yaml").Path
$projectDir = Join-Path (Get-Location).Path "ai_data\v5_hazards\runs"

.\.venv\Scripts\yolo.exe segment train `
  model="$previousModel" `
  data="$yaml" `
  imgsz=960 `
  batch=2 `
  epochs=35 `
  device=0 `
  workers=2 `
  project="$projectDir" `
  name=segment_round04
```

Ultralytics validates on session 04; session 05 is *not* used for model fitting or selection. To preserve its holdout value, evaluate on session 05 only after deciding on a final checkpoint, rather than repeatedly tuning to it.

Example final test after finishing development:

```powershell
$trained = Join-Path $projectDir "segment_round04\weights\best.pt"
.\.venv\Scripts\yolo.exe segment val model="$trained" data="$yaml" split=test imgsz=960 device=0
```

**Note:** 20 validation frames are small and session 04 has limited tunnel examples. A future independent recording could provide a larger validation set. Avoid moving frames from test session 05 into training just to increase counts. The test session was pre-labeled by V5.3 then manually corrected; check for missed objects independently before interpreting performance metrics.
