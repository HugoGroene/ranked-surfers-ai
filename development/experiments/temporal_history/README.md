# Isolated temporal-history experiment

This directory is not imported by the running trainer. It now includes an opt-in
single-worker Electron adapter, scratch PPO trainer and sequential batch evaluator,
as well as the original offline buffer harness. Installation happens only in each
experiment subprocess's memory. No production source, existing checkpoint or
experiment result is modified. Do not run live trials concurrently with the long
training run: resource contention would disturb it and invalidate comparisons.

## Proposed first batch

| Preset | Frames | Span | Capture target | Decision limit |
|---|---:|---:|---:|---:|
| short4_60 | 4 | 100 ms | 60 Hz | 30 Hz |
| wide4_60 | 4 | 500 ms | 60 Hz | 30 Hz |
| wide6_60 | 6 | 500 ms | 60 Hz | 30 Hz |
| wide8_60 | 8 | 500 ms | 60 Hz | 30 Hz |
| wide12_60 | 12 | 500 ms | 60 Hz | 30 Hz |
| wide8_30 | 8 | 500 ms | 30 Hz | 30 Hz |

Each starts from fresh weights, trains for 30 active budget minutes, then receives
100 deterministic evaluation episodes with identical framing/history/rate settings.
All training stages run first, then all evaluations. Order is shuffled reproducibly.
The first arm is an experimental short-history control, NOT an exact reproduction
of the production four-frame pipeline. All arms use the new capture adapter and
temporal metadata. Compare wide4 with short4 for span, wide4/6/8/12 for count,
and wide8_30 with wide8_60 for capture rate.

One seed means three hours of training plus evaluations (not three hours total).
Thirty minutes from scratch is a screening budget, not proof of convergence. Use
another seed and more training for promising/close results. Equal wall budgets
intentionally measure progress per time; optimized interaction counts may differ.
Python/Torch seeds do not control Unity track layouts, and different network input
sizes do not yield identical initial weights even with the same seed. Latest-model
adaptation is deliberately not mixed into this scratch comparison.

## Live command, after the current training run stops

1 November follow-up: the first capture matrix completed; temporal startup failed
on inherited parsing of FPS `30.0`. The isolated rate bridge now preserves numeric
runtime rates while keeping the legacy CLI parser unchanged. Real-parser regression
tests and an eight-frame train/load/evaluate smoke check cover this boundary.
The capture pump also uses an explicit stop flag to survive cancellation races.
See [the completed matrix analysis](../../../reports/CAPTURE_MATRIX_REVIEW_20261101.md).

Use `-CaptureFollowup` for the new focused repeated full-resolution bitmap/GPU
comparison at three/four workers, then the temporal batch. Use `-SkipCapture` to
skip all capture benchmarks and run just the repaired temporal batch. Both avoid
repeating the original complete matrix; the follow-up includes fresh controls.

The combined script now first runs the isolated
[GPU capture and render-resolution matrix](../capture_pipeline/README.md).
Those frozen-policy tests use the latest validated saved detector-off checkpoint;
the temporal-history models below still start from scratch. No capture winner is
automatically adopted. Use `-CaptureOnly` for diagnostics alone or `-SkipCapture`
to run only the temporal experiment. Estimate 4–6 hours for the combined batch
with short evaluation episodes; see the capture README for scope and limitations.

Start the local Practice server if needed, in its own terminal:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s server.py
```

In a separate terminal, this runs the offline test suite, then all six one-minute
smoke trials and their three-episode evaluations. Only if every stage passes does it launch the
six 30-minute scratch trials and their 100-episode evaluations:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\temporal_history\run_batch.ps1
```

No dashboard. The temporary PowerShell policy override applies only to that child
process. Smoke models are never reused as full-trial initialization. No live tests
have been performed by the assistant; the smoke gate is still required.

To prepare a plan without starting anything (safe during current training):

```powershell
& $py -B -s development/experiments/temporal_history/batch.py
```

It prints a unique `reports/temporal_batch_.../plan.json`. Execute that exact plan
later with `batch.py --plan <path> --execute`. Sources must still match its manifest.
Alternatively select a smaller batch:

```powershell
& $py -B -s -u development/experiments/temporal_history/batch.py `
    --configs short4_60 wide4_60 wide8_60 --seeds 123 `
    --train-minutes 30 --eval-episodes 100 --execute
```

Batch flags: `--configs` preset names; `--seeds` distinct seeds (default 123);
`--train-minutes` (30); `--eval-episodes` (100); `--eval-max-minutes` (60 per model);
`--order-seed` (42); `--output` new directory under reports; `--plan` saved plan;
`--execute` opt-in launch. A time-capped incomplete evaluation fails the batch,
not a silently smaller evaluation sample. Existing output folders are rejected.

For arbitrary validated settings use `run_trial.py --mode train --output <new reports path>`
with `--frames`, `--span-ms`, `--capture-hz`, `--decision-hz`, `--minutes`, `--seed`.
Evaluation uses `--mode eval --checkpoint <trial/latest.zip> --episodes 100` and
the identical temporal options. Frame counts: 4 or 6..12; span: (0,2000] ms;
capture target: [1,60] Hz; decision limit:
[1,60] Hz. The observation contract rejects mismatched evaluations. Training
does not accept `--checkpoint`: it always initializes from scratch.

All arms preserve rollout 128, batch 32, five epochs, learning rate .0003,
entropy .002, gamma .995, lambda .95, elapsed-time discount reference .05 seconds,
5 ms key holds, 300-second episode limit, detector off and existing safety checks.
Atomic latest checkpoints are saved at optimized boundaries approximately every
five minutes and at finalization. `initial.zip` preserves the fresh initialization.
Interrupted optimization retains the preceding complete checkpoint.

## Results

Each arm has separate train/eval folders and logs. `comparison.json` aggregates
completion, optimized transitions/second, mean/median/max survival, actions,
death/time-limit counts, recent reaction latency, capture statistics and an
episode-bootstrap mean interval. An arm is eligible
only after a complete training budget, settled-learning audit and full evaluation.
All stages fail closed on errors; no automatic restart or skipped failed arm.
`temporal_capture.json` reports actual captures per wall second (including pauses)
and historical-slot availability. `learning_transitions.jsonl` retains the existing
reward/terminal audit and reaction-latency records; reported capture targets are
not a claim of achieved frame rate or Unity simulation ticks.

## Try it now without a game

From the repository root in PowerShell:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s development/experiments/temporal_history/try_rates.py `
    --frames 8 --span-ms 600 --capture-hz 60 --decision-hz 30 --samples 100

& $py -B -s -m unittest discover `
    -s development/experiments/temporal_history -p 'test_*.py' -v
```

Options: `--frames` accepts 6 through 12 (or 4 for a baseline); `--span-ms`
sets the oldest target relative to the newest captured frame; `--capture-hz`
and `--decision-hz` independently set the simulated clocks; `--samples` sets
the number of simulated decision attempts. Defaults are 8, 600, 60, 30 and 100.
Try `--frames 12 --span-ms 500 --capture-hz 60 --decision-hz 60` too.
These rates are requests for the experiment, not achieved rendering/inference rates.
The CLI exits 2 if any requested sample is rejected and prints rejection reasons.
`try_history.py` is the simpler original one-rate allocation harness.

## Buffer semantics: strict offline harness

- Input: owned copies of uint8 CHW RGB frames, after the existing crop/resize.
- One monotonic timestamp domain; use a conservative source timestamp, never
  relabel an old frame with the time preprocessing finishes.
- Uniform target ages across the span. Choose the most recent frame at or before
  each target, with distinct timestamps. Latest frame is always included.
- Output: stacked oldest-to-newest channels plus actual per-frame ages in seconds.
  Ages are metadata here; they are not yet wired into the policy's auxiliary inputs.
- Default newest-age limit 20 ms, historical selection tolerance 60 ms; these
  are prototype limits, not changes to existing gameplay guards.
- Explicit reset/segment invalidation at episode changes and pause/resume gaps.
  Reject insufficient history rather than silently repeating frames. No sleep or
  extra capture occurs inside sampling. A future live adapter must account for
  startup/passive waiting in learning transitions; it must not silently omit time.
- Retention is bounded by span plus tolerance and a 256-frame hard cap. A cap
  too small for the requested span causes rejection rather than fabricated history.
- Single-owner/event-loop buffer, not thread-safe. RateGate can independently
  limit capture and policy requests without a catch-up burst after delays.

## Live adapter semantics

Six/eight/twelve frames produce 18/24/36 image channels. The existing extractor
already derives its input dimensions from the observation space. Scratch models
also receive N normalized historical ages and N validity bits appended to the
existing 557 auxiliary fields. Earlier action-history/timing fields stay intact.
Frame ages use conservative request timestamps, not inferred Unity ticks.

Unlike the strict offline harness, the live adapter emits zeros with validity=0
for missing/duplicate/out-of-tolerance history. It never duplicates a new image to
pretend it has old history. Latest pixels always come from the current guarded
capture; older frames cannot substitute for current safety evidence. This avoids
adding an unrecorded startup delay. Reset and pause/resume invalidate context;
the four existing resync gameplay checks still run before any policy action.

The producer runs in the browser event loop, with one capture in flight and one
latest full-size frame, plus bounded resized context. Inference remains outside
the callback. Guard requests still demand a capture started after the request;
they do not reuse cached context. Metadata belongs to the delivered frame, not the
producer's subsequent capture. Context generation tokens exclude in-flight frames
crossing resets. During pauses it captures on guard demand only. Existing Electron
serial/geometry/freshness checks remain active, with a 50 ms delivery recheck.
If delivery exceeds 50 ms, the adapter discards that frame and requests a new
capture, up to three attempts. It never relabels or delivers the stale frame.
Persistent staleness still stops safely. `temporal_capture.json` records discarded
deliveries and their maximum age; delivered-frame timings record retry counts.
The source can also reject frames before delivery. A `StaleElectronFrame` after
Electron's three internal attempts is now retried by the pump at most twice,
yielding 10 ms between calls (at most nine future-paint requests total). Only
this fully consumed stale-reply error is recoverable. Invalid geometry, serials,
clocks, connection failures and cancellation still propagate immediately.
Rejected frames never enter history or reach policy/guard code. Source recovery
counts and the last rejection are saved separately from delivery-age discards.
The existing Electron renderer is capped at 60 FPS; this experiment does not
change its renderer rate. The capture option controls requests to that renderer.
This can add request wait time and CPU contention: 60 Hz is a target, not a promised
reaction-rate improvement. Historical frames are visual context, not separately
authorized gameplay/terminal decisions. No blending, contact sheet, recurrent
network, detector or BC changes are introduced.

At 512x288 RGB, dense 128-step image rollout storage is 216 MiB for four frames,
324 MiB for six, 432 MiB for eight and 648 MiB for twelve, per worker before
additional copies. This prototype does not claim memory or learning improvements.

## Validation

43 temporal offline tests pass, including 42 frame-count/span/decision-rate combinations,
buffer ownership and chronology, pump timing/serial ownership, pause idling,
reset isolation, model gradients and deterministic save/load for 4/6/8/12 frames,
batch matching/failure eligibility, and a synthetic settled trial exercising
checkpoint finalization and the learning audit. Model tests used one CPU thread
and small synthetic images; no GPU workload or live browser was launched.
Short live smoke trials have exercised capture, optimization and saving; learning
benefit and long-duration reliability remain unverified.

## Continue a partially completed smoke batch

Run after other training stops, with the localhost:8000 game server available:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\temporal_history\run_batch.ps1 `
    -SkipCapture `
    -ContinueSmoke "reports\temporal_batch_20261101_165534_645446"
```

`-SkipCapture` retains the already completed capture matrices without repeating
them. `-ContinueSmoke` creates a new plan referring to audited completed smoke
stages in the given folder. It validates checkpoint ZIP integrity and hashes
retained artifacts. Failed/missing stages get new output paths; old files are
never overwritten. Evaluations use the matching retained or newly trained model.
All unfinished smoke training runs before smoke evaluation. If the smoke gate
passes, six fresh 30-minute training trials run, followed by their 100-episode
evaluations. `-TrainMinutes` and `-Seeds` configure that full learning batch.

The new `comparison.json` combines retained and new smoke results. If another
smoke stage fails, pass the newly printed batch folder to `-ContinueSmoke` next
time; it retains the earlier references too. Cross-version reuse is deliberately
limited to smoke validation (at most one minute training and three evaluation
episodes). Full learning comparisons must use one unchanged implementation;
old results remain preserved evidence, not silently pooled learning samples.

The referenced smoke batch has now completed all six trainings and six
evaluations. All twelve stages are retained; no live smoke repetitions are
needed by this command. The subsequent full batch uses `--keep-going`: after a
failed child exits with confirmed clean finalization, it records the failure,
continues independent configurations and skips evaluation of that failed model.
Missing reports, cleanup errors, changed sources and changed retained artifacts
still stop the batch. A batch with any failure exits nonzero after the remaining
eligible stages finish; failed arms never count as learning-comparison winners.

To retry unfinished full stages later **with unchanged sources**, also pass
`-ContinueBatch "reports\<full_batch_folder>"`. This wraps
`batch.py --continue-batch <folder>/plan.json --keep-going --execute`, retains
completed audited stages, and assigns fresh paths to unfinished stages. Failed
training restarts from scratch with the full budget; interrupted optimizer/RNG/
environment state is not silently treated as continuous training. If sources
changed, full-batch reuse is refused to avoid mixing implementations. Reports
and checkpoints remain preserved regardless. New trial reports distinguish
unoptimized collected transitions from optimized transitions after an abort.

The cumulative review of the two completed capture matrices and partial temporal
batch is in `reports/EXPERIMENT_PROGRESS_20261101.md` and its JSON companion.
