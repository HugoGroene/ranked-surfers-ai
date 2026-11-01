# Isolated Electron GPU capture and render-resolution tests

This is a prototype, not a production backend change. It uses the existing
Electron 45 runtime and Koffi dependency. Nothing is installed. The Windows x64
bridge calls D3D11 through native FFI, imports Electron's NT shared texture handle,
dispatches a bilinear compute shader, waits for its staging copy, and transfers
BGRA pixels through the existing CPU shared-memory protocol. This is **not** a
zero-copy Torch/CUDA implementation. A native C++ addon could reduce FFI overhead
if profiling justifies it; no Chromium rebuild is required.

The borrowed texture is released in `finally` after GPU readback completes. No
handle is sent to another process. D3D libraries remain alive until COM objects
are released. Format, size, row pitch, geometry and increasing frame serials are
checked. Unsupported textures fail closed. Production 50 ms freshness bounds,
bounded stale retries and all screen-state/terminal thresholds remain enabled.
Initial software paints are ignored in GPU mode. A changed DOM overlay can take
several fresh paint callbacks to appear; the calibration measures this explicitly.
Paint callback timestamps are not proof of Unity physics or screen presentation.

## Arms

| Name | Unity internal resolution | CPU readback | Capture |
|---|---:|---:|---|
| bitmap_100 | 100% | 1920 x 945 | Electron NativeImage bitmap |
| bitmap_75 | 75% | 1920 x 945 | Same bitmap path |
| bitmap_50 | 50% | 1920 x 945 | Same bitmap path |
| gpu_100 | 100% | 1920 x 945 | Shared texture + GPU shader |
| gpu_half_readback | 100% | 960 x 473 | Shared texture + GPU downsample |

All use the same 1920 x 945 CSS viewport and 60 FPS renderer target. Resolution
changes are applied to the served arena.html response **in this browser context**;
the local game files and preferences are not written. Dynamic resolution is off.
The actual canvas backing dimensions and loaded WebGL renderer are checked and
recorded before policy collection. Startup is rejected if the HTML anchors change.

The half-readback arm restores the old guard image dimensions with a CPU resize.
It loses detail and may change classification even with unchanged thresholds. It
does not claim pixel equivalence with full-resolution gameplay. This conservative
adapter tests feasibility without replacing all the inherited guard ROI code.
Policy resizing remains the existing canvas_v1 transformation. Review saved
canvas_v1_native.png/canvas_v1_policy.png, electron/review_capture_*.png and terminal evidence for small coins,
text and geometry before considering adoption. A successful short run cannot rule
out all false detections. The full-resolution GPU arm tests transport independently.

## Execution and checkpoint

`temporal_history/run_batch.ps1` now includes this matrix by default:

1. All offline tests for both experiments.
2. All five synthetic compositor checks: colors, orientation, full-frame parity,
   HTML overlay presence/update and increasing serials, without game inputs.
3. Six frozen stages at one worker, then six at three workers: all five arms and
   a repeated bitmap control. One active minute each, plus startup/whole-rollout
   completion. The same saved policy and worker seeds are used in every stage.
4. Temporal-history smoke batch, then all six 30-minute scratch training stages,
   then all six 100-episode evaluations.

The capture stages default to the validated saved checkpoint:
`reports/v612_3workers_menu_recovery_24h_20261101_111341/latest.zip`.
It is identical to checkpoint_000480000.zip, SHA256
`c733ca3fcfd8228b7b1c724968d6d8a7c674128292e3aaf9ddc13b3990b5058a`,
with 3,493,278 cumulative timesteps. The interrupted run report still says RUNNING;
the checkpoint ZIP is intact. No claim is made that cancellation finalized the run.

These stages are **frozen stochastic collection**, not training. No checkpoint is
written by them; source file and learner policy hashes are verified unchanged.
Report `processed_transitions_per_second`, not optimized/s (optimization is off).
The temporal models start fresh; they are not initialized from this checkpoint.
No winning capture backend is automatically applied to the temporal experiment.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\temporal_history\run_batch.ps1
```

Options:
- `-CaptureFollowup`: repeated full-resolution bitmap versus GPU capture at three
  and four workers (two repetitions, reversed order, repeated bitmap controls).
  Adds about 15–25 minutes before temporal smoke/training/evaluation. Uses no new
  weights and does not adopt a backend automatically. Cannot combine with SkipCapture.
- `-CaptureOnly`: stop after capture/resolution diagnostics, approximately 20–30
  minutes including model/game loading (12 active benchmark minutes).
- `-SkipCapture`: run the temporal experiment alone, after reviewing capture tests.
- `-Checkpoint "path\latest.zip"`: choose another preserved compatible detector-off
  canvas_v1 model for the frozen stages only.
- `-TrainMinutes 30`: minutes for each full temporal training arm.
- `-Seeds 123`: full temporal batch seed(s); default one seed. Smoke seed is fixed.

The full default batch is estimated at **4–6 hours** with short episodes: three
hours of training plus diagnostics, startup, and 600 evaluation episodes. Longer
episodes increase evaluation time. Each full evaluation has a 60-minute cap;
incomplete evaluations stop the batch instead of being scored as complete.

The server must already run on localhost:8000. No dashboard is started. Outputs
go to unique reports/capture_matrix_* and temporal_batch_* directories. Stage logs
are written under the printed matrix directory. comparison.json contains results,
per-worker latency, stage-level capture timing and renderer/scale verification;
system_usage.jsonl and pipeline profiles stay in each raw stage directory.
Three existing owned frames per worker are retained for visual review and encoded
only after collection stops (about 21 MiB extra per worker, identical in all arms).

A failed calibration/guard/frozen-policy check stops the batch before long
training. Existing checkpoints and reports are never overwritten. Browser input
and independent simulation-clock parity remain separate validation questions.

## Verification

Offline tests cover response-only scale changes, changed-wrapper rejection,
one-variable arms, frozen-command construction and incomplete checkpoint rejection.
Run with the project interpreter:
`python -B -s -m unittest discover -s development/experiments/capture_pipeline -p 'test_*.py' -v`

The GPU bridge's full and half-readback synthetic calibration passed locally.
These checks do not establish live Unity throughput or scaling. The user-run
matrix repeats calibration before each full batch.

Two brief single-worker frozen-policy smoke tests also passed locally: 75% Unity
render scale and GPU half readback, each collecting 384 transitions with unchanged
source/policy hashes and no guard stop. They exercised pause/resume and resets.
Actual WebGL renderer was NVIDIA ANGLE/D3D11. Reports:
`reports/capture_scale75_frozen_smoke_20261101_01` and
`reports/capture_gpu_half_frozen_smoke_20261101_01`.
These short checks are implementation validation, not a comparative speed win.

References: [Electron offscreen rendering](https://www.electronjs.org/docs/latest/tutorial/offscreen-rendering),
[shared texture lifetime](https://www.electronjs.org/docs/latest/api/structures/offscreen-shared-texture),
[Windows handles](https://www.electronjs.org/docs/latest/api/structures/shared-texture-handle).
