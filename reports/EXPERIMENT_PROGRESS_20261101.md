# Cumulative experiment review — November 1, 2026

Later update: all six temporal smoke trainings/evaluations subsequently passed;
the first full trial stopped after about 12.5 minutes. See
[the source-staleness recovery review](TEMPORAL_RECOVERY_REVIEW_20261101.md).
The JSON companion retains these subsequent batches alongside the earlier data.
The sections below document the earlier checkpoint in the investigation.

Both capture matrices completed. The temporal batch completed three one-minute
scratch training stages before the twelve-frame startup failure. No 30-minute
learning trials or matched batch evaluations have finished. Capture performance
is promising; learning benefit remains unknown.

## Preserved evidence

- Original matrix: `capture_matrix_20261101_154254_066769/comparison.json`.
  Five calibrations and twelve frozen-policy stages, source checkpoint unchanged.
- Targeted repeat: `capture_matrix_20261101_161307_636905/comparison.json`.
  Two calibrations and twelve frozen-policy stages, source checkpoint unchanged.
- Partial smoke: `temporal_batch_20261101_163025_951906/comparison.json`.
  Four/six/eight frames completed 640 optimized transitions each; each passed the
  settled-learning audit. Twelve frames stopped before any transitions.
- Earlier capture analysis: `CAPTURE_MATRIX_REVIEW_20261101.md`.
- `EXPERIMENT_PROGRESS_20261101.json` retains stage-level results, paths and report
  hashes for both matrices, partial temporal results and the repair smoke test.
  These are separate measurements, not averaged evidence for a learning winner.

## Capture and concurrency

Targeted repeat ranges across stages and individual workers:

| Capture path | Workers | Frozen transitions/s | Policy request-to-command p95 |
|---|---:|---:|---:|
| Bitmap, full resolution | 3 | 43.76–48.37 | 38.6–54.1 ms |
| GPU texture, full resolution | 3 | 46.83–48.09 | 37.7–40.5 ms |
| Bitmap, full resolution | 4 | 45.59–56.20 | 63.6–80.8 ms |
| GPU texture, full resolution | 4 | 57.26–60.15 | 39.7–41.0 ms |

Four GPU workers are a worthwhile next actual-training candidate: the repeat
shows higher collection throughput with much better tails than four bitmap
workers. This does not establish optimized transitions/s, learning speed, Unity
response latency or long-run stability. Track differences, short stages and
system drift remain confounders. Three/four workers also change samples/update.

The original matrix did not show a convincing benefit from lowering Unity render
scale: three-worker bitmap full/75%/50% collected 47.77/46.69/42.59 transitions/s
in those stages. Full GPU and half-size GPU readback collected 48.92/49.39/s.
The small half-readback gain does not yet justify losing coin detail. Prefer
full-resolution GPU for a future controlled training validation.

## Temporal history

| Completed smoke arm | Optimized/s | Capture Hz, including pauses | Policy latency median / p95 |
|---|---:|---:|---:|
| Four frames, 500 ms, target 60 Hz | 10.47 | 42.43 | 32.43 / 71.35 ms |
| Six frames, 500 ms, target 60 Hz | 10.47 | 41.26 | 32.33 / 51.56 ms |
| Eight frames, 500 ms, target 60 Hz | 9.98 | 39.82 | 33.96 / 53.51 ms |

These fresh random policies die frequently; comparison with mature production
models is not controlled. Target capture cadence is not achieved decision rate.
About 76–78% of context slots were valid across all observation packs, including
startup/reset/resync. Missing slots are zero-filled with explicit validity flags;
this aggregate does not isolate steady gameplay history coverage. Do not assume
the model constantly receives a complete half-second history.

## Failure and repair

The twelve-frame trial failed in initial menu capture with zero collected or
optimized transitions. One pump delivery exceeded its 50 ms conservative
request-age limit. The old report lacks the exact age/cause; this does not prove
twelve channels intrinsically overload the machine.

The isolated pump now discards stale candidates and demands a newly started
capture, at most three attempts. It never delivers the rejected frame or resets
its timestamp. Persistent staleness still raises a safety stop. Source errors,
serial checks and all gameplay/terminal guards remain enforced. New counters
record discarded delivery count and maximum age.

36 temporal plus six capture offline tests passed, including stale-then-fresh,
persistent-stale, checkpoint-corruption and continuation-routing regressions.
The repaired twelve-frame live smoke at
`temporal_deliveryfix_12f_smoke_20261101_01` completed two updates / 256 optimized
transitions, passed the learning audit, exercised optimizer pause/resume and
resets, saved a checkpoint and exited cleanly. It required no stale retries;
the retry path itself is covered by injected-delay offline tests. This is a
functional check, not long-run reliability or a learning comparison.
The matching two-episode twelve-frame evaluation in
`temporal_deliveryfix_12f_eval_20261101_01` also completed and exited cleanly,
verifying checkpoint reload and evaluation input compatibility.

## Continuation and next work

`-ContinueSmoke` retains completed audited smoke checkpoints and their original
provenance, while failed/missing stages get new directories. The new summary
includes both retained and newly finished results. Retained artifacts are hashed
and checked before stages execute. The generated plan was verified to retain
three training stages and schedule three unfinished trainings plus six evals.
Cross-version reuse is labeled smoke validation, never pooled full learning data.

Next batch skips both completed capture matrices, finishes the smoke gate, then
runs six fresh 30-minute trials followed by six 100-episode evaluations. Budget
roughly 4–6 hours, strongly dependent on survival and resets; individual evals
have a 60-minute ceiling and incomplete evaluation is a failed stage. All temporal
arms retain the same bitmap backend to avoid mixing a capture change into the
history comparison. They do not load or adapt the production checkpoint.

Do not add another broad capture sweep now. After the temporal gate/comparison,
validate four full-resolution GPU workers with real shared PPO updates, saving
and learning audits before adopting them. The experimental capture adapter is
not a new production default.
