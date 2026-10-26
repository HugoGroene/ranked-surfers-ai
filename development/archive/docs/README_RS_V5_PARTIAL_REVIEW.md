# Ranked Surfers V5: Accept only reviewed images, retrain, then sample spread-out frames

This patch extends (and does not delete or replace) your `rs_v5_prelabel_review.py` workflow.
New file: `rs_v5_prelabel_review_v2.py`. It operates on the **same** existing review batches and manifest files.
Your V4 controller, labeled screenshots and existing checkpoints are left alone.

## 0. Install file

Extract ZIP into `C:\Users\Hugo\Documents\ranked-surfers-ai`, alongside the existing `rs_v5_prelabel_review.py`. Keep the original script as a backup.

The commands below assume the batch you partially reviewed is **`capture02_batch02`**, and you reviewed the **first 16 filenames** in the order the batch was generated. If the actual name differs, update `--batch`. If you skipped around rather than reviewing the first 16, use the `--images-file` option described below.

Finish your LabelMe edits, press Ctrl+S, and close LabelMe before importing.

## 1. Preview which 16 images will be accepted (saves nothing)

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py accept `
  --batch capture02_batch02 `
  --start 0 --count 16 --dry-run
```

The tool prints **all 16 exact filenames**. Verify they are the 16 you manually checked. It also validates the image/JSON pairs and classes. If you reviewed different files, **do not run the next command**; use `--images-file` instead.

## 2. Accept only these 16

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py accept `
  --batch capture02_batch02 `
  --start 0 --count 16 `
  --confirm-reviewed
```

Expected: `[IMPORTED] 16` and `[LEFT UNIMPORTED] 24`. These 16 annotation JSONs are copied next to their original screenshots in `ai_data\v5_hazards\raw\v5_capture_02`. The other 24 stay **only** in the review folder and will **not** be used for training. Do not run the original all-batch `accept` command after this!

If the first 10-review batch (`first_auto_test`) was **not yet accepted**, review it fully and use the older or newer script with `--batch first_auto_test --confirm-reviewed`. If already accepted, leave it alone: trying to import again should refuse overwrite.

### If your 16 reviewed images weren't the first 16

Create a text file `reviewed16.txt` with one **exact image filename** (`frame_XXXXX.jpg` or `.png`) per line. Pick these from `ai_data\v5_hazards\review_batches\capture02_batch02\_review_manifest.json` or Windows Explorer. Then run:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py accept `
  --batch capture02_batch02 --images-file reviewed16.txt --dry-run

.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py accept `
  --batch capture02_batch02 --images-file reviewed16.txt --confirm-reviewed
```

## 3. Export ALL accepted/manual images, including your original set and the new 10+16

Use a new export folder because the exporter refuses to overwrite an existing export:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_export_labelme.py `
  --output ai_data\v5_hazards\export_round02
```

Inspect the printed per-session image counts. Only images with paired LabelMe JSON in the *raw* capture sessions are exported. All unreviewed batch guesses are excluded.

## 4. Fine-tune YOLO with the expanded dataset

Your first model was saved to the **nested** folder shown in your Windows training output:

```powershell
$seedModel = (Resolve-Path 'runs\segment\ai_data\v5_hazards\runs\segment_seed\weights\best.pt').Path
$projectDir = Join-Path (Get-Location).Path 'ai_data\v5_hazards\runs'
$yaml = (Resolve-Path 'ai_data\v5_hazards\export_round02\segment\data.yaml').Path

.\.venv\Scripts\yolo.exe segment train `
  model="$seedModel" `
  data="$yaml" `
  imgsz=960 batch=2 epochs=35 device=0 workers=2 `
  project="$projectDir" name=segment_round02
```

Wait for training to finish. Confirm the new checkpoint path in the final training message; it should normally be:

```powershell
$newModel = Join-Path $projectDir 'segment_round02\weights\best.pt'
Test-Path $newModel
```

If that prints False, use `Get-ChildItem $projectDir -Filter best.pt -Recurse | Select FullName` to find the actual round02 directory, which may have an appended suffix.

**Important:** The round02 run must train on both the old manually corrected set and the newly accepted corrections, not just the latest 16.

## 5. Generate 20 **spread-out** images, rather than 20 consecutive ones

The updated tool supports `--selection spread` (one randomly chosen image in each of 20 equally sized spans across the remaining capture) or `--selection random` (20 uniform random images). Both modes support a reproducible `--seed`.

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py propose `
  --session v5_capture_02 `
  --batch capture02_spread20_round02 `
  --model "$newModel" `
  --max-images 20 `
  --selection spread --seed 42 `
  --conf 0.20 --imgsz 960 --device 0
```

The sampler excludes frames already labeled in the raw session **and** frames reserved in earlier review batches (including the 24 you didn't finish). It therefore won't recreate the previous 40 consecutive candidates.

Open the proposed 20 in LabelMe:

```powershell
.\.venv-labelme\Scripts\labelme.exe `
  ai_data\v5_hazards\review_batches\capture02_spread20_round02 `
  --labels rs_v5_labels.txt
```

Once you have manually checked and corrected the *whole* new 20-image batch, use:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py accept `
  --batch capture02_spread20_round02 --dry-run

.\.venv\Scripts\python.exe -u rs_v5_prelabel_review_v2.py accept `
  --batch capture02_spread20_round02 --confirm-reviewed
```

If you only complete a portion, use the same `--start`/`--count` workflow instead of importing unchecked predictions.

## Safety notes

- Do **not** remove or modify original annotated frames; the `raw` dataset is the source of truth.
- **Saving a JSON is not proof of review.** Every AI-generated suggestion already has a JSON file before you touch it. Only accept files you personally checked, especially for missed coins and wrong barrier classes.
- The original session-separated validation split remains in place. Avoid contaminating a dedicated held-out validation session with unchecked pre-labels.
- `--selection spread` improves time diversity, not semantic diversity: check that the results contain distinct obstacles.
- If 35 epochs are too slow, start with 20 for this assisted-labeling experiment; inference quality is what matters for the next correction batch.
