# Decision latency investigation, 7 November 2026

The command-history configuration remains RGB 384×384, four frame ages
`[0, .05, .25, .75]`, 789 auxiliary inputs, four workers, Halloween + Tokyo.
No checkpoint conversion, classifier threshold change, or altered PPO loss is
needed for these runtime optimizations.

## Enabled defaults

Normal command-history training and evaluation now install precise pacing and
the redundant-stack removal. Training additionally uses bounded pinned transfers,
with ordinary transfers as fallback if allocation is unavailable. The original
optimizer/loss/statistics implementation remains in use. GPU caching, deferred
metrics and native capture modifications are not enabled.

Existing `multimap_classifier/resume.ps1` and `evaluate_both.py` commands pick up
these defaults. `[PIPELINE RUNTIME]` prints the selected configuration;
`pipeline_runtime.jsonl` records actual transfer activation per update. Set
`RS_PIPELINE_FAST_PATH=0` for an explicit rollback; staged experiments independently
select their settings. `RS_RATE_WAIT=sleep` can isolate the old pacing for diagnosis.

## Finding

The pre-dispatch decision limiter called Python 3.10 `time.sleep(remaining)`.
On this machine, requests of 1–10 ms took approximately 16 ms. A policy decision
could therefore wait long enough to expire its 20 ms prepared guard reading,
requiring a fresh guard capture/classification before its movement command.
This is especially visible with a policy that frequently selects NONE: its fast
steps hit the cap more often than steps that send a key.

`pacing.py` substitutes only that wait with a process-local Windows high-resolution
waitable timer. The deadline is rechecked so it never intentionally undershoots
the requested delay. It does not change the 30/s limit, guard retries, freshness
thresholds, image history, optimizer pauses, or the game's clock. Unsupported
platforms fall back to ordinary sleep. Native handles close with the environment.
It does not busy-spin or change the system-wide timer resolution.

API reference: [CreateWaitableTimerExW](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw),
[SetWaitableTimer](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-setwaitabletimer).

## Diagnostics

`analyze.py` summarizes probe ledgers, separating initial, reset, resume, and
ordinary commands. Warm optimizer times exclude the first update. Component
timings overlap and must not be added together. Command metrics omit NONE steps;
they measure dispatch, not when Unity consumes the input.

`eval_probe.py` bounds a single-map evaluation to at most two minutes including
owned-process cleanup. `--rate-wait sleep` measures the old pacing;
`--rate-wait precise` tests the replacement. `--pack` selects the staged evaluator
that also omits the redundant legacy stack. Every arm uses an unchanged source
checkpoint. This is a timing check, not a reliable policy-quality comparison.

The normal evaluator now writes `evaluation_latency_trace.jsonl` and phase
summaries alongside its existing report. Traces include first movement, preceding
action, prediction time, guard fallback counters and pacing waits. They are
buffered (last 20,000 decisions) and saved on close, without extra captures or
model forward passes. `decision_pacing.json` retains bounded wait measurements.

Never edit runtime source while a probe is running: the final source fingerprint
check deliberately rejects such a run, even if its learning audits pass.

## Initial comparisons

All arms started from `reports/bc_actor_repair_20261107_014610_719776/bc_best.zip`.
Individual probes use 0.6 active minutes, complete whole PPO updates, and have a
hard 120-second wall limit. They are diagnostic outputs, not continuation models.

| Arm | Ordinary median / p95 ms | Warm update seconds | Optimized transitions/s |
|---|---:|---:|---:|
| Baseline | 34.2 / 82.2 | 3.61 | 30.4 |
| Pack only | 34.5 / 76.8 | 3.93 | 31.8 |
| Deferred metrics only | 33.8 / 75.2 | 3.62 | 31.7 |
| Pinned transfers only | 34.8 / 76.2 | 2.81 | 35.5 |
| GPU cache only | 34.2 / 77.6 | 2.79 | 35.2 |
| Precise pacing + pack + pinned | 33.5 / 36.2 | 2.84 | 38.5 |
| Reverse baseline repeat | 33.8 / 75.9 | 3.59 | 32.1 |
| Normal optimized runtime | 33.7 / 37.2 | 2.83 | 38.9 |

The combined passing run is `reports/pipeline_probe_20261107_021858_573397`.
It completed three updates, all four settled-learning audits, and cleanup.
The reverse baseline is `022058_501726`; normal-runtime verification is
`022419_955407`, also three updates, four passing audits and clean shutdown.
Ordinary-command counts are 115 combined, 164 reverse baseline, 125 normal runtime.
The normal runtime still had an isolated 74 ms ordinary command and roughly
54–56 ms startup/resume commands; this reduces tails, not a worst-case guarantee.
The timer-only preliminary run `021455_226737` is excluded from passing evidence:
an evaluator source edit during the run failed the final source check.

Pinned transfers use approximately 54 MiB of host staging for batch size 32,
bounded by 96 MiB. GPU caching adds approximately 864 MiB of image storage and
did not improve update time appreciably over pinned staging in these checks.
Deferred metrics alone showed no clear benefit. Keep those alternatives
experimental, and defer native GPU capture changes.

Short-run throughput depends on deaths/resets; it is not evidence of improved
learning or an exact sustained speedup. Startup/resume tails remain distinct.

## Evaluation verification

Normal optimized headless evaluation completed eight episodes per map:

| Map | Movement commands | Request → command median / p95 ms | Maximum ms |
|---|---:|---:|---:|
| Halloween | 26 | 31.5 / 33.7 | 33.8 |
| Tokyo | 34 | 31.2 / 33.2 | 33.6 |

Reports: `reports/multimap_headless_20261107_022620_450840/halloween` and
`reports/multimap_headless_20261107_022739_818502/tokyo`. Both had unchanged source
checkpoint hashes, clean shutdown and no forced process cleanup. These are small
timing samples, not policy-quality evaluations. The visible attempt
`reports/eval_tail_probe_20261107_022551_939810` stopped before gameplay because
the window was not verified foreground/unobscured. Desktop timing therefore still
needs a user-visible run; that protection was not bypassed.

Full comparison and evaluation summaries are saved under
`reports/latency_review_20261107/`. CPU/CUDA tests cover numerical optimizer parity,
image/aux parity, retained minibatch storage, timer deadlines/cleanup, eval tracing,
and restoring the original rollout buffer after optimizer exceptions.

## Resumed Adam counters (7 November)

The normal runtime now restores ordinary Adam's scalar `step` tensors to CPU
before optimizing. SB3's CUDA checkpoint load moved them onto CUDA, causing
repeated scalar-read synchronization during minibatches. This preserves their
values, the learned momentum tensors and all PPO settings. It is idempotent and
skips fused, capturable, differentiable and non-Adam optimizers. Each update logs
the number moved in `pipeline_runtime.jsonl`. It has no evaluation/inference cost.

Three-active-minute four-worker comparisons on the completed 30-minute checkpoint
measured warm update medians of 4.11 s before versus 2.87 s after, with ordinary
command median/p95 approximately unchanged (33.3/37.1 versus 33.1/36.8 ms).
Optimized throughput was 36.4 versus 39.4/s; short-run reset variation still
affects this metric. CUDA tests matched parameter updates, optimizer moments,
step values and learning accounting exactly for active/mixed/passive batches.

The four other candidates (capture-pump pacing, classifier channels-last, policy
channels-last, fused Adam) remain experimental. In the longer pump comparison,
median latency improved to 29.9 ms but p95 worsened to 42.0 ms. Layout changes and
fused Adam did not show a useful benefit. See
`development/staged/latency_candidates_20261107/README.md` for reports and scope.

The completed 30-minute run remains the continuation checkpoint:
`reports/multimap_stability_20261107_023453_536275/stability/train/latest.zip`.
Do not select the short latency probe checkpoints as continuation models.

Normal-runtime verification after promotion:
`reports/latency_candidate_20261107_034335_687507` completed five updates,
all four learning audits, final checkpoint saving and clean process shutdown.
Warm update median was 2.94 s; ordinary command median/p95 was 33.4/37.2 ms
over 1,618 movement commands. The first update moved 20 counters; subsequent
updates moved zero. Source/checkpoint hashes remained unchanged during the probe.
Twelve production CPU/CUDA runtime checks, staged pump tests and the actual
checkpoint's CUDA policy/optimizer parity checks passed after integration.

## Live delay after optimizer resume (7 November)

The ordinary request-to-command metric does NOT include the earlier four-frame
resume verifier. The game is running during that verifier. The completed
15-minute diagnostic showed roughly 294 ms median from confirmed Resume overlay
disappearance to the fourth accepted frame, before subsequent packing/inference
and action dispatch. Do not describe the 60-80 ms first-command latency as the
entire post-resume control gap.

`resync_spacing.py` changes only the verifier's inter-capture pacing. Previously
it slept another 33 ms after the previous capture, classification and packing.
Now it waits until at least 33 ms after that capture's actual request start.
All four accepted frames, pause-DOM and rendered-pause checks, focus, fresh-frame
delivery, confidence thresholds, terminal confirmation and passive-transition
accounting remain intact. No cached pre-pause frame is used as current imagery.
Missing, stale or invalid capture timestamps retain a conservative full gap from
completion. Clock shifts still apply only to independently verified frozen time.

Matched four-worker, three-active-minute probes on
`reports/multimap_stability_20261107_035054_502059/stability/train/latest.zip`:

- Baseline: `reports/latency_candidate_20261107_041534_159287`.
- Candidate: `reports/latency_candidate_20261107_042110_236519`.
- Resume-to-fourth-frame median: 293 ms -> 195 ms (56 completed windows per arm).
- Candidate p95: 211 ms; maximum: 360 ms. This does not eliminate all tails.
- Ordinary request-to-command median/p95: 33.3/37.1 -> 33.3/36.9 ms.
- Throughput: 40.0 -> 39.8 optimized/s; no throughput gain established.
- All four audits passed, no forced cleanup, checkpoints/source hashes preserved.
- Every measured candidate capture interval exceeded 33 ms (minimum 33.136 ms);
  no invalid-timestamp fallback. All 56 completed same-episode windows restored
  history. Seven focused tests cover actual verifier success, menu, pause, focus,
  source drift, timestamp fallback and early timer wakeups.

The cadence correction is enabled by the measured runtime. Per-worker
`resync_spacing.json` stores up to the last 512 verification windows with measured
request intervals, waiting time, fallbacks and verification duration. Full worker
logs retain existing per-frame resume trace lines. The first-command interval is
still separate; this is a measured reduction in live verification delay, not a
claim that learning or long-run survival has improved.

Normal-runtime confirmation: `reports/latency_candidate_20261107_042559_757479`
completed all worker learning audits, source/checkpoint checks, saving and cleanup
after promotion. Detailed baseline/candidate/confirmation results are
`reports/latency_review_20261107/resync_{baseline,candidate,promoted}.json`.

## Five-worker retest

`reports/latency_candidate_20261107_042846_894128` completed three active minutes
with five workers from the same 15-minute checkpoint as the four-worker resume
candidate above. All five learning audits, saving, source checks and cleanup
passed; no hard guard stop. Defaults remain four workers pending longer evidence.

| Measurement | Four workers | Five workers |
|---|---:|---:|
| Optimized transitions/s | 39.82 | 43.54 |
| Ordinary request-to-command median/p95, ms | 33.3 / 36.9 | 34.1 / 40.5 |
| Ordinary maximum in probe, ms | 91.4 | 255.9 |
| Resume verification median/p95, ms | 194.9 / 210.9 | 194.0 / 211.5 |
| Peak system RAM percent | 64.0 | 72.6 |
| Peak sampled GPU allocation, MiB | 4533 | 5293 |

Five workers used three Halloween/two Tokyo environments, versus two each with
four workers, and 640 versus 512 transitions per optimizer update (128/worker).
Short-run throughput depends on deaths, policy actions and reset frequency.
The roughly 9% throughput gain is promising, not proof of faster learning.
Two of 5,447 ordinary movement commands exceeded 100 ms (107 and 256 ms).
The 256 ms outlier included about 209 ms waiting in a fresh guard-image request
after policy inference; inference itself was 4.75 ms. A fresh guard did not make
the preceding policy decision fresh again. Similar isolated stalls have occurred
with four workers, so this single event cannot establish a worker-count cause.

Full phase/per-worker/resource comparisons and spike context are saved in
`reports/latency_review_20261107/five_worker_comparison.json`. This probe remains
diagnostic; continue from the real 15-minute checkpoint, not this test model.
