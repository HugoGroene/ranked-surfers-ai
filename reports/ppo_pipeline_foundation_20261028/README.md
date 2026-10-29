# Autonomous PPO pipeline foundation: implementation and validation

## Delivered changes

- Default episode limit 300 seconds; explicit 15-second historical benchmarks
  still supported. Default training step count 4,096 now matches rollout divisibility.
- Explicit death/time-limit evaluation counts and a local reload after a
  legitimate timeout, followed by the existing guarded restart. No fake death
  penalty is introduced for a timeout; SB3 receives proper truncation metadata.
- A single owned browser and worker thread persist across sequential stages.
  Each stage gets a freshly initialized environment/detector and reloaded game.
- An additional bounded startup focus attempt temporarily joins input queues
  and always detaches. OS foreground and browser focus remain mandatory. No
  synthetic key or global foreground-lock setting is used.
- Opt-in exact-observation gameplay reading reuse, with a single-use identity /
  timestamp check and 20 ms age checks before use and before key dispatch. Old,
  changed or conflicting observations, suspicions and menu handoffs use the
  original independent capture/recheck path. No confidence threshold was lowered.
- Template scores are reused only for their exact capture. Evaluation samples
  probabilities from its existing forward, preserving the selected action and
  stochastic RNG sequence while removing a redundant CNN forward.
- Explicit saved-PPO-setting overrides, with effective settings written to disk
  and rollout buffers reconstructed consistently. No KL/entropy/rate change was
  adopted by default. A 512-step rollout was tested explicitly.
- Training debug step-timing dictionaries are bounded to the newest 10,000;
  complete reward audits still stream to disk.
- Reusable batch manifests support automatic unique run prefixes and remap
  generated checkpoint references between stages.

No demonstration update/loss, recurrent model, movement cooldown, detector
removal, architecture change or action-specific reward was added. All original
checkpoint/recording files remain intact. No long run was launched.

## Offline verification

126 selected tests passed. This includes the existing 48 core tests and 36 clean
runtime/pause regressions plus session ownership/cleanup, Windows focus checks,
time-limit versus real death, SB3 terminal-observation/bootstrap metadata,
actual controller guard fallback, pre-dispatch deadline expiry, saved-setting
and buffer consistency, single-forward prediction/RNG equivalence and audit
cleanup between persistent stages. See offline_checks.json.

The same-frame path is optional. It preserves the gate's evidence/threshold
rules, but moves the initial confidence check to observation preparation. This
changes the capture-to-action timing window; it is not a claim of identical
failure probability or a guarantee that every input uses the newest rendered
frame. A screenshot guard retains a finite capture-to-key race.

## Live browser/timeout validation

An initial smoke attempt stopped before gameplay when Windows refused focus.
Its artifacts are preserved in live_smoke. The subsequent two-stage smoke
completed in one window, recording two time-limit endings and a guarded
reload between episodes. The final timeout was followed by the normal fresh
stage reload. The second stage used a 300-second cap and ended through deaths.

Two four-stage validation batches then completed. Each recorded **one browser
launch and three reuses**, including evaluation-to-training and training-to-
evaluation transitions. The final batch exercised the new native queue-handoff
activation path successfully; later stages retained foreground without another
taskbar click. Windows may still refuse startup on other occasions/machines.
This is not support for training while another application obscures MSS.

The actual controller was also tested synthetically at 15.1, 60 and 240 seconds
without truncation, and at 300.1 seconds with correct timeout bootstrapping.
These tests establish support for longer episodes, not a model capable of
surviving multiple minutes.

## Final latency comparison

The refined comparison used the same preserved pilot, settings and owned window,
ten episodes per condition. Configurations differed in run name and the optional
observation-guard flag. Both used the single-forward evaluation implementation.

| Metric | Normal guard | Same-observation guard |
| --- | ---: | ---: |
| Movement commands measured | 659 | 739 |
| Median policy RGB age at dispatch | 28.22 ms | 13.77 ms |
| p95 policy RGB age at dispatch | 33.88 ms | 20.93 ms |
| p99 policy RGB age at dispatch | 39.92 ms | 42.52 ms |
| Median gameplay observation interval | 51.92 ms | 40.19 ms |
| Mean gameplay observation interval | 53.57 ms | 39.43 ms |
| p95 gameplay observation interval | 65.15 ms | 54.12 ms |
| p99 gameplay observation interval | 66.90 ms | 57.71 ms |

The shorter median interval corresponds to roughly 29% higher reciprocal-median
observation frequency. It is **not** a 29% improvement in learning, independent
experience collection, or end-to-end training throughput. The game still advances
in real time; additional frames can be correlated. These sequential, unpaired
small samples do not establish better survival or long-run reliability.

Pixel ages run from desktop grab completion to Playwright request start. They
do not measure when Unity applies an input or displays its response. The 20 ms
reuse check uses the existing post-processing capture timestamp, so actual
pixel age can be slightly older. Legacy fallback decisions can have older policy
images; the limit applies to prepared readings, not every fallback action.
Rare fallback tails remain an optimization/validation concern, so the path is
still opt-in. Raw audits include startup and recovery outliers.

An earlier iteration improved median latency but worsened p95. That result and
its raw records were retained in validation_batch. The redundant forward and
template work were then removed; final results are in validation_v2_batch and
latency_comparison.json. Do not select only a favorable median from the first
iteration and hide its tail regression.

## Training smoke tests

The first four-stage batch completed 2,048 PPO steps at an effective 512-step
rollout; the refined batch completed another independent 1,024-step smoke run
from the unchanged source at the same rollout size. Saved checkpoint data and
effective-setting metadata both record n_steps=512. Image shape remains
12x288x512 and auxiliary dimension 557.

Both passed PASS_STABILITY_CANDIDATE and PASS_REWARD_REPLAY. Optimizer pauses
were longer (roughly four seconds) and less frequent, with active clocks
corrected and fresh four-frame resynchronization before continuation. The first
test verified three continuations; the refined test verified one. Subsequent
evaluation loaded each smoke output in the same browser. No guard abort/hard
stop occurred in completed validation stages.

Observed smoke throughput was 15.81 and 16.32 steps/s including resets. These
are not controlled learning-efficiency comparisons against the old overnight
runs: policies, maps, episode samples and rollout sizes differ. The smoke models
are diagnostic outputs, not new best checkpoints.

Pause-clock drift is an internal consistency measure, not an independent Unity
timer. Reward replay verifies arithmetic, not unbiased reward caps or discounts.
Full audit warnings remain in training_validation.json and per-batch audit files.

## Use and next step

See [the pipeline guide](../../README_PPO_PIPELINE.md) for the ready-to-review
65,536-step monitored manifest, automatic run prefixes, source preservation and
explicit hyperparameter overrides. It was not launched as part of validation.
The original pilot remains the survival reference. Use longer monitored PPO
budgets and periodic evaluations before claiming better learning or promoting
the faster path for unattended overnight use.

[Remote and parallel training](SCALING.md) distinguishes offloading one worker,
parallel sampling and engine-controlled acceleration. No cloud resources were
provisioned, external files uploaded or packages installed.

All source checkpoint hashes were verified; see checkpoint_hash_verification.json.
The single-forward and instrumentation-detachment housekeeping also has focused
unit coverage. No commit or push was performed. Existing historical result
reports/checkpoints were not rewritten.
