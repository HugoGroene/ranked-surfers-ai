# Ranked Surfers V2: performance + real-hazard recognition checks

**Purpose:** test the bottlenecks and whether important visual categories
survive the model's 256×144 RGB resize **before** you spend hours recording
human demonstrations or training a final PPO agent.

These scripts operate only on your **local Practice game** and save files
under `ai_data`. They do **not** change your PPO weights or checkpoints.

Put the three `.py` files beside `train_rs_v2_verify.py` and your other scripts
in `~/Documents/rankedsurflocal/`, then activate `.venv` as usual.

## A. Measure the hardware bottleneck (about a minute)

```bash
cd ~/Documents/rankedsurflocal
source .venv/bin/activate
python rs_v2_perf_probe.py --live --samples 80
```

If MSS is blocked because of the display session, you can benchmark only
resizing using an existing saved screenshot:

```bash
python rs_v2_perf_probe.py \
  --image ai_data/rl_v2_runs/v2_r_smoke/startup_after_r.jpg
```

**Look for:** median capture times for `full` versus `windowed` and CPU
preprocessing times for OpenCV AREA versus LINEAR at 1, 2 and 4 threads.
`INTER_LINEAR` is often much faster on large-to-small downscaling but may
alias or lose small coins; do **not** change the trainer's resize method just
because it is faster without checking images and recognition performance.

It saves `ai_data/v2_speed_probe.json`. These numbers *do not* include the
Unity WebGL rendering process, Playwright input latency, or GPU PPO steps.

**Windowed-mode A/B check:** Your V2 trainer already accepts no `--fullscreen`
flag. This retains the **same** four RGB 256×144 frames. For an initial
controller-only test:

```bash
python -u train_rs_v2_verify.py --mode smoke \
  --run-name v2_windowed_test --max-minutes 2 \
  2>&1 | tee ai_data/v2_windowed_test.log
```

Check `actual_model_input.png` afterward. `R` works in fullscreen already;
verify it in windowed mode rather than assuming it does. If the windowed crop
looks good and restarts reliably, compare `[FAST]` logs from the two display
modes under similar conditions. **Do not mix fullscreen and windowed images
within one model's training** until you establish that the scene rendering is
consistent enough for transfer.

## B. Capture real images without demos or PPO (4–6 minutes)

Start your local Unity HTTP server as usual. Ensure Chromium can take
fullscreen, and keep the game visible.

```bash
python -u rs_v2_capture_hazards.py \
  --fullscreen --minutes 5 --interval 0.6 --max-images 300 \
  2>&1 | tee ai_data/v2_hazard_capture.log
```

This runs the previously verified smoke controller with random actions,
restarts using `R`, and saves only **actual AI-input screenshots** to:

```
ai_data/v2_hazard_images/ep0001_frame00000.png
ai_data/v2_hazard_images/ep0001_frame00004.png
...
```

No human actions, behavior-cloning dataset or PPO model are created. The
frames are already **256×144**, the same spatial resolution as V2 observations.

## C. Label a small recognition dataset

```bash
python rs_v2_hazard_probe.py label --images ai_data/v2_hazard_images
```

A GUI displays 4× enlarged frames (to make small coins easier to click):

- Press **C** for COIN, **T** for TRAIN, **B** for BARRIER, or **E** for an
  unobstructed patch of track/background. Choose the type **before** drawing.
- Click-and-drag a tight rectangle around an object or representative clear
  region. Label the *visible object*, not what the ideal player action would be.
- **U** undoes the last rectangle on that image; **N** advances; **Q** exits.
- Labels are saved immediately to `ai_data/v2_hazard_images/labels.json`.
  The tool resumes unlabeled/unreviewed frames when reopened.

**Target:** at least **12 labels per category across ≥5 different episodes**
to run the probe at all. For interpretable results, prefer **30–50 labels
per category** spread across many independent episodes, including small and
large coins/trains/barriers. For 'empty', select genuinely clear track/background
regions. Spend only a little time on the first pass: this is validation, not
recording demonstrations. With a random episode-level train/holdout split,
there must be at least a few examples of each category in both splits.

The test does **not** determine which lane is safe or reward any action. A
single scene can contain several different types of objects and be ambiguous
for planning; we are *only* checking object appearance recognition.

## D. Train and evaluate a *diagnostic* recognizer offline

```bash
python rs_v2_hazard_probe.py train \
  --images ai_data/v2_hazard_images --epochs 30 \
  2>&1 | tee ai_data/v2_hazard_recognition.log
```

The script reports:
- Holdout accuracy and macro-F1 (all four classes weighted equally),
- Majority-class guessing baseline,
- Confusion matrix with actual versus predicted labels,
- Best checkpoint selected by holdout macro-F1,
- A warning if labels/classes/independent episodes are insufficient.

Output:
`ai_data/v2_hazard_images/hazard_probe_report.json` and
`hazard_probe_weights.pt`.

**Critical limitation:** this is a separate *patch classifier trained on
manually marked boxes*, **not** proof that the PPO model autonomously locates
hazards or chooses safe moves. It is an inexpensive test of whether the visual
information is present at 256×144. Full scene localization and temporal
recognition would be separate steps, and a recognizer should NOT be turned
into a reward signal without verifying false positives on live gameplay.

## E. Choosing hardware and display mode

Your prior V2 logs measured about 24 ms MSS, ~22 ms downsampling, and only
3–4 FPS training throughput with 1920×1080 site fullscreen, compared with
~7 ms MSS, ~5.6 ms resizing, and ~11 steps/s from the earlier 1160×644 source
in the older 160×90 model. This is not an exact A/B test because *both*
source size and model input size changed, among other things. The speed probe
isolates capture and preprocessing so we don't overattribute all of it to GPU.

A Ryzen 7800-class desktop and RTX 4060 Ti should have more CPU/GPU headroom,
but the measured bottleneck may still be Unity, screenshot capture, and browser
round trips. A desktop is worth comparing once the recognition experiment
passes. Keep the trained policy, Python versions, and browser resolution
consistent between machines.
