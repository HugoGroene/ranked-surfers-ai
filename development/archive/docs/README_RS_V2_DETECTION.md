# Ranked Surfers V2: whole-image hazard detection (diagnostic)

Run from `~/Documents/rankedsurflocal` with your `.venv` activated. Keep your existing images and `ai_data/v2_hazard_images/labels.json` where they are.

## 1) Convert existing labels (no new manual boxing)

Copy `rs_v2_object_detector.py` to the project folder and run:

```bash
python rs_v2_object_detector.py prepare
xdg-open ai_data/v2_yolo_dataset/ground_truth_preview.jpg
```

The generated dataset is `ai_data/v2_yolo_dataset/`. It contains YOLO `data.yaml`, `images/train`, `images/val`, and corresponding normalized box coordinates in `labels/`. **Only `coin`, `train`, and `barrier` are object classes.** Your old `empty` labels mean background and are not detector targets.

Check that **every visible hazard in the example images is actually boxed**. The previous patch-classification task tolerated incomplete frame annotation; object detection doesn't. An object that's clearly visible but unlabeled is incorrectly taught as background. To add missing labels, use your previous labeling UI with `--review-all` and re-run `prepare --overwrite` before training. CAUTION: `--overwrite` recreates the ENTIRE prepared dataset folder, including detector checkpoints if you have already trained there. Only do this *before* training, or back up the runs folder first.

Keep validation by full gameplay episode, not individual screenshot, to reduce near-duplicate leakage. Check `split_report.json` for class counts.

## 2) Install pretrained YOLO detector package

```bash
python -m pip install "ultralytics>=8.3,<9"
```

It may download pretrained weights (`yolo11n.pt`) on first use. Ultralytics is not used by the old trainer. Ensure your version of PyTorch still detects CUDA: `python -c 'import torch;print(torch.cuda.is_available())'`.

## 3) First detector training run

```bash
python -u rs_v2_object_detector.py train --epochs 40 --imgsz 640 --batch 4 2>&1 | tee ai_data/v2_yolo_train.log
```

The converted 256×144 gameplay images are resized internally by YOLO. `imgsz=640` does NOT invent missing detail; it creates a larger working tensor for detecting small objects.

Check the saved weights at `ai_data/v2_yolo_dataset/runs/hazards/weights/best.pt` and summary chart at `ai_data/v2_yolo_dataset/runs/hazards/results.png`. The script evaluates the held-out episodes automatically.

**Caution:** 49 annotated screenshots, 18 barrier boxes, and ~9 independent episodes is a SMALL object-detection dataset. Even excellent validation numbers may be brittle on genuinely new gameplay. We should not use this detector to shape RL rewards until its errors are reviewed visually.

## 4) Audit detections on held-out gameplay

```bash
python rs_v2_object_detector.py predict --split val --limit 16 --conf 0.20
xdg-open ai_data/v2_yolo_dataset/predictions/val/
```

Look for missed small coins, false coin detections on yellow scenery, missing distant trains, and low confidence barriers. To audit *new, never-labelled* images later:

```bash
python rs_v2_object_detector.py predict --source ai_data/v2_hazard_images --limit 30 --conf 0.20
xdg-open ai_data/v2_yolo_dataset/predictions/external/
```

Note the external folder may also contain training screenshots. For a truly unseen test, capture a new few runs **after** training into a separate folder using `rs_v2_capture_hazards_manual.py --output ai_data/v2_hazard_unseen ...` and then `--source ai_data/v2_hazard_unseen`.

## How to decide what follows

If the detector catches coins/trains/barriers in new scenes reliably, the next phase is to derive lane-relative positions and movement trajectories, and then experiment with reward shaping. Recognition alone is not a safe-lane policy and is insufficient to certify precise movements.
