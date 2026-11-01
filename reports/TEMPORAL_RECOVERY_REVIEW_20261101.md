# Temporal batch recovery — source-level stale frame

## Newly completed evidence

`temporal_batch_20261101_165534_645446/comparison.json` contains all six completed
smoke trainings and six three-episode deterministic evaluations, including the
three training artifacts retained from the previous batch. They remain usable
as functional checks, not adequate learning comparisons.

| Arm | Optimized transitions | Optimized/s | Three-episode eval mean |
|---|---:|---:|---:|
| wide4_60 | 640 | 10.47 | 4.75 s |
| wide6_60 | 640 | 10.47 | 2.94 s |
| wide8_60 | 640 | 9.98 | 1.41 s |
| wide12_60 | 640 | 9.66 | 3.41 s |
| short4_60 | 640 | 10.34 | 4.86 s |
| wide8_30 | 512 | 8.08 | 5.02 s |

Three evaluation episodes and about one minute of scratch training cannot rank
these configurations reliably. The 30 Hz capture arm also collected fewer
transitions per second. The original two completed capture matrices remain
valid historical evidence; no new capture sweep is justified by this failure.

## Interrupted full trial

`temporal_batch_20261101_170206_256137/wide8_60_seed123/train` was the first
30-minute arm. It stopped after 756.47 seconds including setup, with 61 complete
updates, 7,808 optimized transitions and 7,884 collected transitions. The 76
remaining collected transitions were not optimized. The old raw stats' zero
`discarded_partial_transitions` does not mean all interactions were trained.
New reports explicitly expose `unoptimized_collected_transitions` on abort.

The checkpoint ZIP passed its integrity check, with 305 optimization epochs
(61 updates x 5). Its saved timestep counter is 7,884, which counts interactions;
the learner ledger is the authority for 7,808 optimized transitions. There were
no recorded finalization errors. Preserve this model, but do not use it as an
equal-budget completed arm or as the initialization of other scratch arms.

234 completed episodes: mean survival 1.84 s, best 6.55 s, first 25 mean 1.63 s,
last 25 mean 2.173 s. This is an encouraging within-training change, not proof of
temporal-history benefit. Sampled LEFT/RIGHT actions remained present. The last
single-observation policy snapshot favored NONE (0.749 probability), which is
not a global policy distribution or evidence that it learned complex combos.

Last 1,000 reaction samples: policy request-to-command median 32.65 ms, p95
71.47 ms; guard request-to-command median 32.38 ms, p95 36.27 ms. A fresh guard
image does not make the older policy decision equally fresh. Across gameplay
intervals the median was 51.12 ms and p95 81.88 ms. Aggregate capture rate was
41.22 Hz including pauses, with 78.2% valid history slots across observation packs.
None of these timings measure Unity's action-response latency.

## Cause and targeted change

The previous delivery-level fix worked: it discarded eight late deliveries,
including a 127 ms candidate. The new exception came earlier, inside Electron's
decoder after its three capture attempts. The last rejected reply was 74.367 ms
old by the conservative request-roundtrip bound. The pump treated that transient
source error as permanent and stopped producing frames.

The isolated pump now permits at most three source calls (nine underlying paint
requests) for this specific consumed-reply staleness error, yielding 10 ms between
calls. No rejected image is published and all 50 ms limits remain. Persistent
staleness, invalid geometry/serial/clock, connection failures and cancellation
still terminate safely. Counters record source recovery separately from delivery
discards. This reduces one failure mode; it cannot guarantee an arbitrarily long
run against sustained contention or other faults.

43 temporal and six capture offline tests passed. New tests cover actual
Electron retry exhaustion followed by recovery, nine-paint exhaustion without
publishing any pixels, immediate propagation of protocol errors, full-batch
continuation source matching, and continuing independent arms while skipping
evaluation of a failed model. Cleanup failure still aborts the batch.

The repaired live eight-frame check in
`temporal_source_recovery_smoke_20261101_01` completed 384 optimized transitions
across three updates, passed the learning audit, exercised pause/resume and
resets, and saved/exited cleanly. Its policy latency median/p95 was 33.49/37.94 ms
over 307 samples. No source retry was needed in this short run; injected-error
offline tests validate that branch. This does not establish 30-minute reliability.

## Next batch and preservation

The next command retains all twelve completed smoke stages and skips both
capture matrices. Six fresh 30-minute trials then run before their matched
100-episode evaluations. Restart the incomplete eight-frame arm from scratch
to preserve comparable budgets; its partial checkpoint and reports stay intact.

The full batch now continues independent stages after a failed child with
confirmed clean finalization. It records failure and excludes that model's
evaluation. Missing reports, cleanup failures and integrity violations still
abort. Unchanged-source full batches can be continued using `-ContinueBatch`;
completed stages remain referenced at their original paths, failed stages get
new paths. Cross-version learning results are retained separately, not pooled.

No additional broad experimental arms were added: six temporal conditions and
the already measured capture/concurrency conditions are enough for the next
screening step. All changes in this repair are isolated to the temporal experiment.
