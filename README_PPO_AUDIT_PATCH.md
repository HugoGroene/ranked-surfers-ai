# Audit follow-up: detector-free operation and experiment integrity

Prepared in isolation while the 31 October two-hour run was active. This document
describes the staged changes; they become active only when the patch is applied.

## Defaults and input compatibility

New training, evaluation, BC and recorder CLI defaults are **detector off** and
**canvas_v1**. The 557-value auxiliary shape stays compatible: detector geometry
is zero/unavailable, detection freshness is false, and unavailable age is encoded
as before. Ordinary action/timing context is retained. The separate screen-state
safety classifier remains required. No YOLO checkpoint is required in off mode.

`--detector-mode on` and `--framing-profile legacy` remain explicit historical
reproduction options. Do not use them for the new detector-free continuation.
Other speed settings, including tap duration and capture/transport selection,
retain their previous defaults: continue using the explicit optimized command.

Training/evaluation inspect checkpoint JSON before starting a browser. A new
`observation_contract` records framing, detector mode, image shape/order,
resize/crop, auxiliary version/dimension, action mapping and lane-context mode.
New checkpoints must match that contract. Older checkpoints validate the
framing/detector metadata they contain and report the remaining uncertainty.
Missing historical metadata means legacy framing and detector on.

| Flag | Where | Meaning |
| --- | --- | --- |
| `--allow-input-migration` | Resumed single/shared training | Explicitly allow and record a deliberate input-contract change. Never silently applied. Rejected for evaluation/frozen comparison. |
| `--worker-seed-base N` | Shared training | Derive distinct reproducible worker sampling seeds. Default inherits saved seed, or 123 if absent. Does not seed Unity tracks. |
| `--legacy-eval-metrics` | Standalone evaluation | Historical wall-time survival and reward reporting. Default is verified gameplay survival/rewards, consistent with settled training. |
| `--framing-profile canvas_v1` | Training/eval/BC/recorder | Current default; clean calibrated canvas and fixed preprocessing. |
| `--detector-mode off` | Training/eval/BC/recorder | Current default; object detector disabled, safety classifier enabled. |

The current detector-off/canvas checkpoint can continue without an input migration
flag. Migrating old detector-on checkpoints is a separate experiment, not an
automatic prerequisite for continuing the current model.

## Evaluation

The default `durations_seconds` and `mean_survival_seconds` now use verified
gameplay time. Reports additionally retain `wall_durations_seconds`,
`verified_durations_seconds`, `survival_metric` and `return_metric`. Legacy mode
reports wall durations as the primary metric and null verified durations.

Evaluation records `status`, `stop_reason`, requested/completed episode counts,
`completed_requested_episodes`, errors and terminal evidence counts. CLI exit code
2 means incomplete evaluation, including wall-time cutoff, user interruption or
UI safety stop. Successful completion returns 0. Frozen policies are warmed before
the initial reset. Live validation of the new evaluation reporting remains due;
do not directly compare new verified-time means with old wall-time means.

Training and evaluation also tag terminal evidence as inherited template,
pre-action guard, boundary guard or time limit. The patch changes no death
acceptance threshold, recovery bound or movement authorization rule.

## Shared training and saving

Workers are reseeded after loading/warmup and before collection. Seeds are saved
in configuration and per-worker JSON. The optional shared inference service has
its own reproducible sampling stream; client seeds do not imply independent
per-client RNG streams inside that service.

Startup provenance includes source/native-helper hashes, installed package
versions, selected asset hashes and the observation contract. Hashing is startup
work, not a per-action operation.

The final committed learner checkpoint and an atomic status report are attempted
before cleanup. Cleanup stages are isolated so one failure does not skip the
others or the final report. Errors are recorded and preserve an existing primary
failure. If interruption occurs **inside an optimizer update**, that partial
model is not saved as a committed checkpoint; the prior periodic save remains the
recovery point. Hard process kills and power loss still rely on periodic saves.

## BC and recordings

Both BC entry points select matching framing and validate the checkpoint input
contract. Detector-off BC copies auxiliary samples and converts only detector
fields to their unavailable representation; recordings remain unchanged. Cached
feature reuse now requires matching contract metadata, so old feature caches
must be regenerated. No old frames are recropped or reapproved automatically.

The original BC entry point also allows epoch zero (the original actor) to win
when every trained epoch worsens holdout loss. The recorder skips YOLO loading
and inference by default and labels the capture's detector mode. Recorded
demonstrations still require review and whole-run holdout. This patch does not
run BC, change its loss, or train the visual extractor during actor-only BC.

## Deferred until evidence or a separate controlled check

Remaining action/terminal evidence can be collected in one separate-terminal
command: `python -B -s -u -m tools.run_ppo_live_checks`. Start `server.py` first.
The default runs two repetitions of LEFT, RIGHT, JUMP, ROLL and LEFT_TWICE using
5 ms taps in local headless Electron, with detector off and existing safety guards.
It saves exact returned observation sequences, step timestamps, terminal evidence
sources and browser event counts in a unique `ai_data/rl_v2_runs/live_action_checks_*`
folder. No checkpoint is loaded or trained. `--trials` controls repetitions and
`--max-minutes` bounds collection (default five minutes, checked between steps).
Success means evidence collected, not verified Unity movement. Inspect the images
afterward. Image writes add diagnostic overhead; this is not a speed benchmark.
Physical-browser parity, independent simulation-clock verification, human BC
recording and long reliability runs are not automatically certified by this script.

The diagnostic now uses `--startup-wait-seconds 1.5`: after each guarded reset,
it advances with NONE actions before testing movement. Lead-in frames and a fresh
`pre_action` image are saved; this fixed delay is not proof the opening animation
has finished. A terminal during the lead-in stops the test without movement.
This is diagnostic-only and does not change training or classifier behavior.

For frozen local headless evaluation, use `python -B -s -m
tools.evaluate_ppo_headless --resume CHECKPOINT --run-name UNIQUE_NAME
--eval-episodes 100 --max-minutes 30`. This entry point uses Electron shared-memory
capture, canvas_v1, detector-off inputs and the existing evaluation CLI. It never
runs optimizer updates; incomplete evaluations exit nonzero. Start `server.py`
first. See `reports/AUDIT_PATCH_LIVE_20261031.md` for the short live checks and
remaining validation. A single-worker evaluation has different contention from
three-worker training; it is not a matched reaction-latency benchmark.

- Consolidating inherited template-only terminal acceptance with the newer guard.
- Re-inferring or suppressing stale policy decisions after guard recovery.
- Verifying Unity action consumption and simulation-time progression.
- Changing PPO settings, exploration, rewards, network architecture or image size.
- Hot-path logging, aggregation and memory optimizations without a performance check.
- A new independent GAE audit trace; existing return/terminal tests remain intact.

These are not silently included in an infrastructure patch. First analyze the
completed two-hour run, then choose the smallest diagnostic/change supported by it.
