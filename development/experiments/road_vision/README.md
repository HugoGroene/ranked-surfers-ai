# Isolated road-vision experiment

This directory adds optional policy crops, smaller inputs, a spatial encoder and
end-to-end behavior cloning (BC). It does not change production or temporal-history
source files. The complete live comparison remains pending.

Update: `reports/road_preflight_20261101_01` completed a one-minute road128/spatial
smoke (768 optimized transitions, PASS_SETTLED_LEARNING, no guard stop or
finalization errors). This verifies that one path runs, not learning quality or
all-crop coverage. Its step-30 preview was a startup camera view, so later gameplay
crop evidence is explicitly collected by the short suite below.

## Short checks before longer learning comparisons

Run after stopping other training/benchmarks, with the usual local server on port
8000. Allow approximately 45-65 minutes, depending on startup/reset times:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\road_vision\run_short_checks.ps1
```

This is a separate 40-stage suite. It never starts the overnight learning batch:

1. Offline road, temporal and capture regression tests.
2. Eight one-minute road/encoder training smokes using full-resolution GPU shared
   textures (including the optional 96-pixel and tighter central crop arms), plus
   a paired bitmap road128/spatial smoke. Each is followed later by a matching
   three-episode evaluation. Four frames / 500 ms, target capture 60 Hz, decision
   cap 30 Hz throughout; no BC and no checkpoint migration.
3. Six episodes of cached full-frame crop evidence, then learned/NONE/JUMP/ROLL
   controls (50 episodes each, eight-minute cap). All use the same completed
   `wide4_60_seed123` model, including its actual model-forward workload. Constant
   controls override the selected action after inference, not the input guards.
   The evidence stage buffers at most 12 full images and writes PNGs after closing
   the game; it is excluded from the four behavioral controls.
4. Bitmap/GPU-full compositor calibrations, then repeated four/six-worker frozen
   capture benchmarks, with order reversal and bitmap controls. These use the
   preserved mature production-compatible checkpoint from the earlier capture
   matrix. They do not use the new scratch crop models, and do not measure PPO
   update throughput. No reduced render resolution or half-readback sweep repeats.

Outputs are indexed in the printed `reports/short_checks_.../short_checks.json`.
Individual stage logs and report/checkpoint paths are retained. The crop evidence
folder contains native images, candidate crop outlines and actual small policy
images. Startup/death frames may occur; inspect moving-road examples and lane
coverage instead of treating file existence as a passed crop check.

To resume with unchanged source, retaining verified successful stages:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\road_vision\run_short_checks.ps1 `
    -Resume "reports\short_checks_REPLACE_WITH_PRINTED_FOLDER"
```

Failed stages get new paths, and failed training's evaluation is skipped. A cleanly
finalized failure allows independent stages to continue; missing reports or
unverified cleanup stop the suite. Resume rechecks source and retained artifact
hashes. Changes to code require a new suite, preserving the old evidence separately.
`-PlanOnly` prints a plan without launching a browser. No stage installs anything.

The menu-confirmation experiment adds only eligibility for bounded fresh checks
when six confident MENU readings include a strong template hit and the final three
have moderate template agreement. This covers the saved episode-61 failure. The
unchanged production confirmation still requires fresh supporting evidence, never
returns PLAY, aborts on conflict, and expires after six further captures. Production
source is unchanged; the process-local extension is active in road trials and
control evaluations, not the separate capture matrix. Offline replay tests prove
the branching and bounds; they do not prove how often live recovery is needed.

The failed eight-frame/30-Hz learning trial is deferred because this suite fixes
history at four frames. A short suite cannot rank final learning ability. It can
eliminate broken paths, expose bad crops, estimate reaction/scaling costs and
measure simple behavior controls before allocating hours to repeated training.

The road trainer and long-batch runner now accept `--capture-backend gpu_100`
(PowerShell wrapper: `-CaptureBackend gpu_100`). `bitmap_100` is the paired
full-resolution experimental bitmap control; `native_bitmap` preserves the old
launcher and remains the generic runner default until these checks are reviewed.
This connects the existing shared-texture implementation to actual PPO training,
not just frozen capture tests. Backend choice is recorded in trial metadata and
held constant between training and evaluation. Both experimental full-pixel paths
force the same internal render scale; half-readback/lower-resolution variants are
not adopted. Full pixels keep the observation contract unchanged across capture
backends; source/backend provenance still distinguishes the experiments.

Worker-count checks at this stage screen capture latency and resource contention
on the mature frozen model. They do not prove optimized throughput for every new
crop/encoder, whose computation and memory costs differ. Hold worker count at one
for the architecture learning comparison, then verify the selected model under
the candidate multiworker configuration before a long production continuation.

## What is borrowed from the video

[Niklas Petersen's subwAI](https://github.com/nikp06/subwAI/tree/9dad0af80869979b26075bc2cf98dcb887d5c27d)
uses supervised imitation of human actions, a compact cropped image, horizontal
mirroring with LEFT/RIGHT relabeling, and a CNN with a flattened spatial feature
map. Its demonstration is not evidence that these changes improve our no-coin
task. We retain PPO and our existing action, reward, pause and safety behavior.

The new spatial encoder keeps the existing three convolution layers and ReLU,
but preserves an 8x8 feature grid instead of pooling to 3x6. It is an experiment
in preserving obstacle positions, not an exact copy of the video's architecture.
No dropout, hand-written action suppression or new action rewards are added.

## Controlled default comparison

| Arm | View | Policy image | Encoder |
|---|---|---|---|
| full512_tiny | Full canvas | 512x288 | Existing |
| full128_tiny | Full canvas | 128x128 | Existing |
| road128_tiny | Central 70% | 128x128 | Existing |
| road192_tiny | Central 70% | 192x192 | Existing |
| road128_spatial | Central 70% | 128x128 | 8x8 spatial |
| road192_spatial | Central 70% | 192x192 | 8x8 spatial |
| road256_spatial | Central 70% | 256x256 | 8x8 spatial |
| road384_spatial | Central 70% | 384x384 | 8x8 spatial |

New plans default to `gpu_100` for every arm, including existing-encoder controls;
production defaults are unchanged. The 256/384 candidates have offline shape and
gradient coverage but still require their live smoke gates. Previously completed
results remain evidence under their original source hashes; do not resume an old
plan across this change or rerun the whole short suite just to test these two arms.

The default four frames span 500 ms (nominal targets approximately 0, 167, 333 and
500 ms old). Actual frame ages and validity are recorded. This is an unproven
movement-context hypothesis, not a measured animation duration or a mandatory
500-ms wait before each action. Keep history constant for crop/encoder comparisons;
later compare shorter history on the selected candidate if useful. A frame stack
is not recurrent memory and does not guarantee learning button combinations.

Optional arms are `road96_spatial` and `center128_spatial` (central 50%). All
crops retain the full vertical canvas extent: the distant horizon and nearby
obstacles remain in view. The central 70% is a conservative candidate, not a
verified guarantee that all lane changes and camera poses remain visible. Inspect
the smoke run's `road_bounds_step*.png` and `road_input_step*.png` before selecting
a crop for longer training. Missing screenshots are not a passed visibility test.

All arms use aspect-preserving area downsampling and letterboxing. Thus the
full512 control is the control for this experiment, not an exact reproduction of
the production linear-resize path. Compare resolution, crop and encoder using
matched pairs in the table. The 192-pixel arm checks whether small coins disappear
at 128 pixels. Squashing the entire frame to a square is deliberately avoided.

Full-resolution canvas images still feed guards and pause checks. Only policy
images and temporal history are cropped. Smaller inputs reduce policy/rollout
work; they do **not** reduce Unity rendering or full-frame readback/transport yet.
For four RGB frames and 128 steps, image storage alone is 216 MiB at 512x288,
54 MiB at 192x192 and 24 MiB at 128x128. These are buffer sizes, not total RAM use
or measured speedups. More retained spatial features can cost extra computation.

Temporal settings are shared across arms and configurable: default four frames
over 500 ms, capture target 60 Hz, decision cap 30 Hz. Choose these after reviewing
the running temporal batch. History selection, age/validity inputs, segment reset
isolation, freshness limits and settled-learning audits reuse that implementation.
The existing action history remains present. This CNN is not recurrent and does
not promise to learn combos; the tests ask whether clearer geometry and temporal
context improve learning. Capture targets are not promises of achieved frame rate.

## Run after the current batch finishes

Default mode writes a plan only. No browser or model starts:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\road_vision\run_batch.ps1
```

For the first live check, start the usual localhost game server, then:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\road_vision\run_batch.ps1 -Execute -SmokeOnly
```

This runs offline tests, eight one-minute training smokes and eight three-episode
evaluations. Allow roughly 10-20 minutes, depending on startup/reset costs. Review
crop screenshots. Then reuse passed smoke stages and run the full comparison:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\road_vision\run_batch.ps1 `
    -Resume "reports\road_batch_REPLACE_WITH_PRINTED_FOLDER" -Execute
```

Alternatively `-Execute` without `-SmokeOnly` runs everything in one invocation.
The phases are offline gates, all smoke checks, all eight 30-minute training runs,
then all eight 100-episode deterministic evaluations. Budget about 5-6 hours for
planning, not a guaranteed completion time; the configured evaluation timeout is
60 minutes per model, so long-surviving models can take longer. Each trial uses
one worker. Do not overlap other training while measuring performance.

All default models start **from scratch**; no latest production checkpoint is
loaded or overwritten. Evaluation loads each trial's own checkpoint with exact
crop/history/encoder compatibility checks. Every completed stage, checkpoint and
comparison is retained. `-Resume` requires unchanged source hashes and verifies
retained artifacts; failed stages restart into new folders, not mid-episode.
Changed-source results remain separate evidence. Five-minute checkpoints also
remain available for inspection after interrupted longer trials.

`road_batch.py` exposes `--arms`, `--seeds`, `--eval-episodes` and
`--train-minutes`; the wrapper exposes `-Frames`, `-SpanMs`, `-CaptureHz` and
`-DecisionHz`. Resume uses the prior plan's settings. To change configuration,
create a new plan rather than resume.

## Optional end-to-end BC

`--bc-root <approved-demo-root> --allow-downsampled-replays` on `road_batch.py`
adds BC-initialized copies of the spatial arms alongside their scratch controls.
BC trains the visual encoder and actor, with uniform action sampling and
train-only horizontal mirroring. It mirrors LEFT/RIGHT labels, action history,
lane context and action-age features together. Critic-head weights stay fixed
during BC, although shared encoder changes can change value predictions.

Whole recording runs are separated into training and validation before sampling
or mirroring. Reports include per-action recall and cross-entropy; natural
holdout class proportions are preserved. BC time is reported separately from the
equal PPO wall budgets and included in total stage time. Classification accuracy
alone is not a gameplay result.

Only explicitly approved, schema-compatible `canvas_v1` recordings are accepted.
The currently inspected approved recordings are legacy framing, so **BC is not
enabled in the default batch**. They are not silently relabeled or converted.
Recorded 512x288 images have already lost detail; the explicit replay flag
acknowledges that cropping those pixels differs from cropping the native canvas.
No new demonstrations, approvals or recorder changes are made here.

## Interpreting results

Root `comparison.json` combines training throughput, capture/latency reports,
matching evaluation survival statistics and action counts; BC arms also record
initialization cost. Each run retains source/input contracts and learning audits.
Compare matched evaluation performance as well as latency and optimized samples
per second. Equal training time does not mean equal sample counts. Game track
seeds are not controlled, and episode bootstrap intervals do not measure variation
between independent training seeds. Thirty minutes is screening, not convergence.

After screening, repeat promising arms with multiple seeds and benchmark the best
selected models with three and four workers. Recent six-worker GPU tests failed;
do not include six again without a targeted reason. Concurrency gains are not
established by this single-worker batch. No configuration is automatically promoted
to production, and the safety thresholds are not relaxed to improve throughput.

Offline tests (CPU only):

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -m unittest discover `
    -s development/experiments/road_vision -p 'test_*.py' -q
```
