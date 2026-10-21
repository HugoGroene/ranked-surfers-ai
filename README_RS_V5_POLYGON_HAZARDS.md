# Ranked Surfers V5 — native hazard capture + polygon annotations

Status: data collection/annotation tooling. **Do not replace the current V4 YOLO detector checkpoint with a V5 model yet.** The existing PPO auxiliary feature adapter knows only `coin`, `train`, `barrier`; it would silently discard new class names without a future adapter update. V4 controller remains untouched.

This version targets local no-coin Practice only. It uses the working V4 MENU/LOADING/GAMEPLAY screen classifier **only as a recording filter**. During capture it does not press R, send swipes, click, or navigate the browser. Human should manually play using the same Chromium window framing as the V4 automated controller. No live ranked gameplay.

## Why polygons?

Bounding boxes are rectangles (not necessarily squares). Boxes are quick for simple coins, but they include a lot of irrelevant pixels for train ramps, arches, railings, slanted barriers, and train roofs. A *segmentation polygon* traces the visible contour. We retain both polygon labels and boxes computed from those same labels, letting us benchmark detection-vs-segmentation tradeoffs using exactly the same dataset. Segment models are generally heavier; our 31 Hz gameplay baseline must be re-measured later.

## 1 — Install/prepare

In PowerShell from `C:\Users\Hugo\Documents\ranked-surfers-ai`:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
Test-Path ai_data\v4_screen_states\screen_state_best.pt
```

It must be `True`. The main `.venv` already has numpy, OpenCV, MSS, and torch. The camera must be the **same screen and Chromium layout** as V4, with Chrome window borders/scale matching its screenshot. The recorder applies the same normalized crop `0 .055 1 .965` by default and saves full native-resolution cropped frames (typically around 1920×983, *not* 512×288 and definitely not 256×144).

For polygon labeling, use **open-source LabelMe**. Keep its GUI dependencies out of the PPO virtual environment. One option:

```powershell
py -3.10 -m venv .venv-labelme
.\.venv-labelme\Scripts\python.exe -m pip install labelme
```

Alternatively install a standalone Windows LabelMe release from https://github.com/wkentaro/labelme/releases. Official LabelMe documentation: https://github.com/wkentaro/labelme and https://labelme.io/docs/annotate-image.

## 2 — Record GAMEPLAY at native resolution

Start local server in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory .\rankedsurfers-downloaded
```

Open local game `http://localhost:8000/` in the SAME kind of Chromium window used for the V4 screenshot benchmark, not a differently zoomed fullscreen layout. Practice and restart manually with `R` and arrows. On a *second* terminal:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_record_native.py `
  --session v5_capture_01 `
  --seconds 120 --fps 8 --save-every 0.30 `
  --max-images 300 --delay 5
```

You have five seconds to focus Chromium. Start at MENU; manually press R, play, die, restart. The recorder sends **NO** inputs. It only saves images when the existing screen-state classifier predicts GAMEPLAY at >=0.90 confidence; images are sampled at least 0.30s apart, with basic near-duplicate filtering. If a class is rare, manually replay until you get several diverse examples.

**Check the first frames immediately**:

```powershell
Start-Process "ai_data\v5_hazards\raw\v5_capture_01"
```

They should show only game content, at native game-crop resolution. **Check that thin coins and distant barriers remain visible**. If the layout differs from the automated V4 controller's source crops, STOP; fix Chromium framing/crop before labeling hundreds of images.

Record at least two additional independent sessions (new names, never overwrite originals):

```powershell
.\.venv\Scripts\python.exe -u rs_v5_record_native.py --session v5_capture_02 --seconds 120 --fps 8 --save-every 0.30
.\.venv\Scripts\python.exe -u rs_v5_record_native.py --session v5_capture_03 --seconds 120 --fps 8 --save-every 0.30
```

You can start with 50–100 distinct useful annotated screenshots; increase toward several hundred after inspecting class coverage. Long repetitive sequences of identical starting trains do less good than variety across speed, roofs, ramps, coins, barriers, tunnels, and lighting. Session names are preserved in the validation split.

## 3 — Label polygons (not only boxes)

The included `rs_v5_labels.txt` contains **exact spelling** of nine classes. Start LabelMe as:

```powershell
.\.venv-labelme\Scripts\labelme.exe ai_data\v5_hazards\raw\v5_capture_01 --labels rs_v5_labels.txt
```

In LabelMe, choose **Polygon** and click around the visible outline; close the shape by clicking the first point again, or press Enter after placing 3+ points. Save the result as the neighboring JSON file. Use Rectangle or Circle for small coins if quicker; the exporter understands all three. Move to the next image; label *all visible instances*, not just the most dangerous or biggest. LabelMe's saved files remain beside originals as `frame_XXXX.jpg` and `frame_XXXX.json`.

| Class | Visible object/region | Notes |
|---|---|---|
| `coin` | Individual coin | Fatal to touch in **this** no-coin challenge; every coin matters. Rectangle/circle works. |
| `train_front` | Front/bumper of a train | Include oncoming and parked fronts. Do not label the entire roof as front. |
| `train_side` | Visible vertical side/body of train | Physically distinct from front and top. |
| `train_roof` | Visible traversable roof surface | Important for roof navigation and landing interpretation. |
| `ramp` | Slanted surface leading onto train | Polygon useful for wedge-shaped red ramps. |
| `barrier_jump` | Obstacle that is safely cleared by jumping, not rolling | Only assign when confirmed in local gameplay. |
| `barrier_roll` | Overhead arch/signal cleared by rolling | Only assign when confirmed in local gameplay. |
| `barrier_either` | Obstacle cleared by either jump or roll | Don't guess: use confirmed examples. |
| `tunnel_entrance` | Tunnel arch/opening boundary | Useful when roof-running into a tunnel. |

Important: the class identifies **visible geometry**, not guaranteed outcome in every movement state. Contacts with roof vs front vs side are determined by the player trajectory, elevation, and timing. Annotate only *visible* surfaces, not imagined hidden parts behind occluders. A single physical train can have separate front, side, and roof polygons. For indistinguishable or uncertain barrier types, skip the ambiguous object's class or collect test footage to verify; avoid teaching contradictory labels. Don't annotate ordinary background track, walls or powerups (none in this mode) unless we later decide they are necessary.

The mechanics such as mid-air lane switching, jump-cancel into roll, input buffering, or wall contact cannot be learned directly from static YOLO object labels. They belong to PPO/action-state modeling with temporal vision.

## 4 — Validate your hand labels and export

After labeling the first batch:

```powershell
.\.venv\Scripts\python.exe rs_v5_preview_annotations.py --max-images 24
Start-Process "ai_data\v5_hazards\annotation_previews"
```

This writes overlay previews without modifying the original images or LabelMe JSON files.

Export the session-separated dataset:

```powershell
.\.venv\Scripts\python.exe -u rs_v5_export_labelme.py `
  --raw ai_data\v5_hazards\raw `
  --output ai_data\v5_hazards\export_01
```

It exports two different YOLO formats from the SAME annotations:

```
ai_data/v5_hazards/export_01/
  segment/
    data.yaml
    images/train/  images/val/
    labels/train/  labels/val/    # class id + polygon coords
  detect/
    data.yaml
    images/train/  images/val/
    labels/train/  labels/val/    # class id + rectangular boxes
  export_manifest.json
```

It **does not** silently put the same session in train and validation. Missing annotations (no .json) are ignored, not considered negative images. Deliberately empty labeled images *with* a LabelMe JSON containing 0 shapes can be used as negative examples. Dataset YAML `path` is an absolute local path; generated on Windows for Windows. It fails if the export folder already exists, so choose `export_02` after relabeling or adding sessions.

A class that appears in only one recording session cannot be honestly tested on a different session. The export script keeps all classes in training and **warns** when some classes have no validation examples. If ramp/roof classes are missing from validation, make a new capture session that includes them before relying on reported per-class metrics.

## 5 — Train (only once dataset looks good)

From the project root in your active PPO `.venv`:

```powershell
# First test polygon segmentation at 640 with a small pretrained model:
yolo segment train model=yolo11n-seg.pt `
  data=ai_data/v5_hazards/export_01/segment/data.yaml `
  imgsz=640 batch=4 epochs=30 device=0 workers=2 `
  project=ai_data/v5_hazards/runs name=segment_640_v1
```

If detection of distant tiny coins is poor and VRAM allows it, compare `imgsz=960 batch=2` in a new run. The segmentation model should be timed in the complete live pipeline before any change to the V4 controller.

For a cheaper bounding-box comparison from the SAME hand labels:

```powershell
yolo detect train model=yolo11n.pt `
  data=ai_data/v5_hazards/export_01/detect/data.yaml `
  imgsz=640 batch=4 epochs=30 device=0 workers=2 `
  project=ai_data/v5_hazards/runs name=detect_640_v1
```

**Important integration note:** `rs_v2_live_hazards_v2.py` has `CLASSES = ('coin','train','barrier')`, and its `hazard_features()` packs a fixed `FEATURE_COUNT` from exactly those classes. The new V5 taxonomy will NOT be propagated correctly through this old adapter. We will create a versioned V5 hazard feature encoder (including masks/boxes, safe/unsafe geometry, distance estimates, freshness) and update the PPO auxiliary dimensionality before live controller integration. Do not just use `--detector path/to/new_V5_weights.pt` with V4 and assume PPO is receiving train fronts or ramps.

## 6 — What to send back before a full labeling marathon

Please record **one 1–2 minute session**, inspect its images, and annotate just **10–20 diverse frames**. Send: screenshot(s) of your polygon outlines, which classes you found, and whether gameplay filtering skipped the menu and kept full-quality coin images. We can adjust labels/crop once, early, instead of correcting hundreds of labels afterward.
