# Road and spacing screening batch

Run from repository root with the local Practice server available on port 8000:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u development/experiments/road_vision/overnight_batch.py --execute
```

Plan only: omit `--execute`. Short checks only: `--execute --smoke-only`.
Continue an interrupted batch, preserving verified completed stages:
`--resume reports/road_overnight_REPLACE --execute`. Sources must be unchanged;
new results go into a new folder and old results are never overwritten. Failed
training trials restart fresh rather than receiving a different effective budget.

All offline road, temporal and capture tests run before browsers. All ten
one-minute training smokes and their three-episode evaluations finish before any
longer training. Any failed smoke gate stops the longer phase. Then all 20 full
training runs precede their evaluations. Independent full-trial failures are
reported and valid completed results retained; failed training is not evaluated.

## Comparison

Seven crop/encoder configurations use four uniform frames over 500 ms:

- full128_tiny
- road128_tiny
- road192_tiny
- road128_spatial
- road192_spatial
- road256_spatial
- road384_spatial

The road192_spatial configuration additionally tests consecutive, golden and
logarithmic histories. This is ten configurations total, not a full factorial.
It isolates encoder at 128 and 192, resolution within the spatial encoder, and
spacing at road192. It cannot establish interactions between every spacing and
every resolution. Confirm the promising combination later if needed.

Each configuration uses seeds 123 and 456, a fresh model, 30 minutes of PPO and
300 deterministic evaluation episodes (90-minute cap per evaluation). Timed-out
evaluations remain incomplete, not passing results. Training game-track seeds
are not controlled. Two seeds improve screening but are not a definitive ranking.

All use one worker, full-resolution gpu_100 capture, detector off, requested 60-Hz
capture and maximum 30-Hz decisions, unchanged action timing/rewards/guards. No BC.
Model weights and optimizer start fresh for every full trial, not from smoke or
the latest production model. Defaults total ten hours of training plus 6,000
evaluation episodes, approximately 20-26 hours overall at recent episode/reset
speeds; stronger policies may take longer. The default `--max-hours 0` disables
the whole-batch time limit. An optional positive `--max-hours` checks the budget
before each new stage, reserving its configured duration plus three minutes for
startup/finalization. It stops with `BUDGET_LIMIT` and preserves completed work if
there is insufficient time. It never kills a running process; pathological stalls
can still exceed the estimate. Resume starts a new session budget with the saved
plan. `batch_status.json` distinguishes complete, stopped and budget-limited runs.
`COMPLETED_WITH_FAILURES` means all runnable stages were attempted, with failures
preserved separately. Smoke failure stops immediately. Full-trial failures may
continue only after recorded successful cleanup; missing cleanup evidence or user
interruption stops the batch. Failed training is never evaluated. Even a zero
process exit code does not override a failed report/audit. New budgets apply to
new plans; resume deliberately preserves the original plan's settings.

## Exact history meaning

Ages below are relative to the newest frame; tensors retain oldest-to-newest order.

| Pattern | Nominal ages in milliseconds, newest first |
|---|---|
| consecutive | Actual current and previous three observation-pack timestamps |
| uniform | 0, 166.7, 333.3, 500 |
| golden | 0, 95.5, 250, 500 |
| logarithmic | 0, 64.1, 202.3, 500 |

Golden uses cumulative gaps in ratio 1:phi:phi squared, normalized to 500 ms.
Logarithmic uses `(10**(i/3)-1)/9 * 500`, i=0..3. Both concentrate more detail near
the present; neither has a special theoretical advantage established here.

The original production stack appended consecutive observation frames with a
variable span. It cannot simultaneously be consecutive and have a fixed 500-ms
span. The consecutive control recreates that selection style within the new
pipeline, keeping age/validity features and explicit zero-padding. It is not an
exact replay of original production timing or its repeated startup padding.

Oldest-frame span for consecutive depends on measured execution; `span_ms=500`
only supplies the common age-normalization scale for that arm. The other three
share exactly the same requested span. Actual selected ages and validity appear
in policy auxiliary fields; unavailable history is zero-filled and flagged.
Temporal state clears at reset/pause boundaries. All patterns retain the newest
guarded frame and the existing freshness/selection constraints. No waiting to
accumulate half a second before acting.

Spacing is in the checkpoint observation contract. Nonuniform/consecutive BC is
explicitly refused until a matching replay selector is implemented; BC is deferred.
Production defaults are unchanged. Temporal trial finalization now also handles
failure before settled-model initialization, and its batch runner accepts the
optional before-stage budget hook.

## What the policy receives

The four selected RGB frames are accompanied by four normalized frame ages and
four validity flags. Ages use the conservative monotonic capture-request timestamp,
not a verified Unity tick, and are divided by span+60 ms and clamped to [0,1].
Raw wall-clock/absolute timestamps are not neural inputs. Existing auxiliary
features retain four recent action choices, time since each of the five action
types (clamped at one second), recent observation intervals and episode age.
Action history and timing are separate from selected images: they are not a
complete timestamped command sequence paired to each older frame, and do not
prove a command caused the intended Unity movement. Detector geometry stays
unavailable; unverified lane odometry remains disabled. No new reward shaping or
recurrent model is added in this comparison.
