# Accumulated temporal-history and capture review — 1 November 2026

## Decision

Defer new BC recordings. Use the next overnight budget to test repeatability and
image representation, not another large frame-count sweep. No globally optimal
configuration, visually adaptive policy, or benefit from longer history has been
established. The completed learning trials mostly converged to one deterministic
action. This is the most important limitation of their rankings.

This review changes no training, guard, model, or experiment source. No live tests
were launched. The next schedule below is a proposed experiment plan, not an
implemented 12-hour watchdog or a claim that the existing batch runner enforces
that deadline.

## Evidence retained

Raw final batch: `temporal_batch_20261101_180312_752232`.
Machine-readable aggregation: `TEMPORAL_FINAL_ACCUMULATED_20261101.json`.
Reproducible read-only analysis: `analyze_temporal_final_20261101.py`.

Five 30-minute scratch trials completed and passed PASS_SETTLED_LEARNING; their
five 100-episode deterministic evaluations completed. A sixth training trial
stopped, and its evaluation was correctly skipped. Successful stages are valid
artifacts despite the batch's nonzero final exit status.

Earlier smoke, interrupted training and different source versions are retained
as separate evidence in the JSON inventory. Reused smoke stages are not additional
independent trials. In particular, one-minute smoke scores and partial training
are not pooled into the 30-minute learning comparison. All original outputs remain.

## Completed temporal comparison

All models used seed 123, one worker, 512x288 policy images, no BC/detector,
30 Hz decision cap, and requested 60 Hz capture (except the failed 30 Hz arm).

| History | Evaluation mean / median, seconds | Episode-bootstrap 95% interval for mean | Optimized transitions/s | Deterministic decisions |
|---|---:|---:|---:|---|
| 4 frames / 500 ms | 4.35 / 4.75 | 4.03–4.72 | 11.80 | 7,360 JUMP; nothing else |
| 6 frames / 500 ms | 3.68 / 3.75 | 3.42–3.97 | 11.57 | 6,167 JUMP; nothing else |
| 8 frames / 500 ms | 2.95 / 2.95 | 2.76–3.17 | 10.63 | 4,804 ROLL; nothing else |
| 12 frames / 500 ms | 3.00 / 2.94 | 2.76–3.28 | 11.88 | 5,953 NONE; nothing else |
| 4 frames / 100 ms | 3.00 / 2.93 | See raw JSON | 11.97 | 6,399 NONE + 22 JUMP (99.66% NONE) |
| 8 frames / 500 ms, capture 30 Hz | Incomplete | Not applicable | Not comparable | Stopped after 1,408 optimized transitions |

Action counts are policy choices reported by evaluation, not proof of successful
Unity movements on every step. Probabilities were not literally one-hot: the
four-frame wide policy's mean maximum probability was 0.8651, for example.
Stochastic training therefore continued to sample other actions even though
deterministic argmax evaluation chose only JUMP.

One seed cannot separate the effect of history from initialization/trajectory
variation. Even the two always-JUMP evaluations have different survival means;
track sampling and timing can affect scores without different action choices.
The bootstrap intervals describe episodes of those trained models, not variation
between independently trained models. Nonoverlap alone does not establish that
one architecture learns better.

Training did optimize behavior: sums of completed-episode active time in the
learning ledger increased across episode-index quartiles. Four wide frames went
1.66 -> 2.39 -> 3.06 -> 3.28 seconds; six went 1.77 -> 3.40 -> 3.46 -> 3.61.
These are stochastic training trajectories under changing policies, not matched
evaluation results or independent simulation-clock verification. They are
consistent with learning useful simple strategies, not proof of coin-dependent
jump timing or multi-action sequences.

## Timing

Requested 60 Hz capture achieved about 41.5–45.1 captures/s including pauses.
Wide-history valid-slot fractions were 81.0–83.8%; short history was 91.3%.
Padding is represented explicitly; a request for 12 frames does not mean all
12 useful distinct frames were available on every observation.

| History | Last 1,000 recorded samples p95, ms | All recorded ledger command samples p95, ms |
|---|---:|---:|
| 4 / 500 | 34.8 | 60.6 |
| 6 / 500 | 37.0 | 35.4 |
| 8 / 500 | 37.0 | 36.2 |
| 12 / 500 | 71.6 | 71.5 |
| 4 / 100 | 73.9 | 73.2 |

These are policy screenshot-request-to-command measurements, not Unity response
time. Rows have different counts; transitions without recorded command latency
are excluded. The whole-run analysis is more informative than calling the last
1,000 samples representative of the entire run. An input's age can also increase
while a fresh guard recapture is performed. Twelve frames has no demonstrated
learning advantage here and has a worse tail; defer it for the next screening.

## Failed stage

`wide8_30_seed123` stopped about 192 wall seconds into startup/training, with 1,408
optimized transitions and 49 additional collected but unoptimized transitions.
Finalization reported no errors. Its 7.34 optimized/s is an interrupted short-run
measurement, not a valid matched 30-minute ranking.

This failure is **menu confirmation**, not the previous stale-frame exception.
The saved episode-61 image visibly shows FIND RANKED MATCH. All six ML menu
probabilities were 0.992–0.999, but template scores dipped:
0.985, 0.541, 0.489, 0.667, 0.667, 0.600. Replaying the saved readings through the
current logic returns ABORT and `pending_menu_tail=False`.

The initial strong match falls outside the tail predicate's last-five window;
only its last three readings have moderate corroboration. The evidence supports
an overly conservative stop in this instance, not permission to accept arbitrary
ML-only terminals. Before an overnight run, a focused movement-free confirmation
regression/probe should verify any proposed recovery using fresh captures and
unchanged terminal thresholds. A static saved frame cannot supply those future
captures. Do not blindly disable the guard or auto-restart indefinitely.

Include one bounded retry of this missing 30-minute arm, followed by its own
100-episode evaluation, in the next batch's diagnostic phase. No separate manual
full batch is needed. Keep all five completed arms. A retry after code changes
must carry separate source provenance and cannot be labeled the original result.

## Accumulated capture results

Both completed capture matrices are retained. They used the same frozen mature
checkpoint; optimization was off. These are collection rates, not training rates.

The initial three-worker sweep found bitmap full 46.0–47.8 collected/s, 75% Unity
resolution 46.7/s, and 50% resolution 42.6/s. Lower render resolution did not show
a useful three-worker advantage and sacrifices small-object detail. Do not spend
the next night repeating that entire sweep.

The repeated full-resolution follow-up is more useful:

| Capture / workers | Collected transitions/s across runs | Per-worker p95 command latency range |
|---|---:|---:|
| Bitmap / 3 | 43.8–48.4 | 38.6–54.1 ms |
| GPU shared texture / 3 | 46.8–48.1 | 37.7–40.5 ms |
| Bitmap / 4 | 45.6–56.2 | 63.6–80.8 ms |
| GPU shared texture / 4 | 57.3–60.2 | 39.7–41.0 ms |

The four-worker GPU result is promising evidence of better scaling. Stage order,
different game trajectories and short runtimes still matter. It has not established
four-worker PPO update throughput or six-worker reliability. Preserve full pixels;
half readback gives only a small additional three-worker gain and changes guard
image detail. Keep capture backend fixed across learning arms; test concurrency
separately before combining changes.

## Proposed overnight schedule (maximum budget 12 hours)

BC is excluded. Four frames / 500 ms at requested 60 Hz capture and 30 Hz decision
cap is a provisional inexpensive setting, not a proven temporal winner.

1. Offline implementation gates, road crop smoke/visibility checks, and the
   focused menu-confirmation check. Preserve results and stop unsafe stages.
2. Complete the missing 8-frame/30-Hz arm once (30 minutes plus evaluation).
3. Short full-resolution GPU/bitmap concurrency follow-up at four and six workers,
   with repeated control/order balancing, on the same frozen checkpoint. Report
   collection and reaction latency separately. Do not interpret it as PPO quality.
4. Train the following four configurations, each for 60 minutes with seeds 123
   and 456: `full128_tiny`, `road128_tiny`, `road128_spatial`, `road192_spatial`.
   That is eight independent fresh models and eight training hours. All share
   temporal configuration, capture backend, rewards, input rules and PPO settings.
5. Evaluate every completed model for 100 deterministic episodes; allow at most
   12 minutes each, report partial evaluations as incomplete. Add constant NONE,
   JUMP and ROLL controls with matching input/cadence/guards, retaining comparable
   inference workload so faster scripted decisions do not confound the control.

The comparisons isolate crop (`full128_tiny` vs `road128_tiny`), spatial encoder
(`road128_tiny` vs `road128_spatial`) and resolution (`road128_spatial` vs
`road192_spatial`). Existing 512x288 results remain a historical reference; they
are not a same-source/same-duration control for new 128-pixel runs. The existing
six-arm implementation smoke checks can still include full512 and road192_tiny,
without spending an hour on every combination.

Nominal budget: 480 training minutes + about 60–96 evaluation minutes + about
60–85 minutes for smoke/retry/capture/constant controls = about 10–11 hours before
some startup/finalization overhead. Reserve the balance for overhead and cleanup.
Use a batch-level deadline with graceful checkpoint finalization before promising
a strict 12-hour maximum; individual trial limits alone are insufficient. Never
call a time-truncated evaluation COMPLETE or silently shorten only some training
arms and rank them as equal-budget trials. Independent finished stages must remain
resumable after failure/deadline; no automatic promotion of the best observed mean.

The priority is fewer better-supported comparisons. Keep 100 evaluation episodes:
increasing to 500 cannot compensate for training only one model per configuration.
Two seeds is still a screening compromise; close results need more runs later.
If all eight models still reduce to constant actions, investigate the training
objective/exploration/observation conditioning before another architecture sweep.
Keep rewards and entropy fixed now so they do not become confounding changes.

Select a promising representation on repeatability, survival relative to simple
controls, action behavior, latency distribution and resource cost. Then perform a
longer confirmation and record BC in that chosen input contract. This identifies
a defensible candidate within the budget, not the globally optimal configuration.
