# Readback latency optimization, 6 November 2026

Implemented without changing image pixels, model shape, weights, rewards, guard
thresholds, capture freshness requirements or frame spacing. BC remains separate.

## Scope and results

1. **Guard preprocessing:** tested moving BGRA conversion after resizing, cropping
   before conversion, and OpenCV thread counts 1/2/4. The tested pixels matched,
   but savings were negligible or negative. No classifier change was installed.
2. **Readback copies:** the GPU bridge now copies tightly packed rows in one call,
   and writes directly into its existing shared-memory slot. Padded rows retain
   row-by-row copying. This removes a 7,257,600-byte temporary per 1920×945 frame
   plus the subsequent full-frame copy. GPU staging/Map still blocks; this is NOT
   a zero-copy GPU policy pipeline, GPU-side guard preprocessing, or asynchronous
   staging-ring implementation.
3. **Pinned PyTorch buffers:** a real-policy offline benchmark with alternating
   standard/reused-buffer blocks saved about 0.15–0.23 ms per complete forward and
   action retrieval. All outputs matched. Not installed in the runtime: this small
   isolated benefit does not establish a four-worker win or justify shared-buffer
   lifetime complexity yet.

Source checkpoint for every live test:
`reports/multimap_stability_20261106_171018_244012/stability/train/latest.zip`.
SHA-256 `b32149ae9d436a4944cc25c72fee904e2df03ba9185ee9923d3519733e9e5f09`.
Every probe used a separate diagnostic output; the source remains unchanged.

| Case | Ordinary median | Ordinary p95 | Median mapped + shared-copy cost |
|---|---:|---:|---:|
| Baseline A | 33.65 ms | 39.08 ms | 1.87 ms |
| Baseline B | 33.61 ms | 40.79 ms | 1.92 ms |
| Bulk copy only | 33.43 ms | 39.40 ms | 1.44 ms |
| Direct shared A | 33.39 ms | 38.14 ms | 0.42 ms |
| Direct shared B | 33.14 ms | 37.59 ms | 0.39 ms |

These are short four-worker training probes with real optimizer updates and
pause/resume, not frozen throughput tests. Each finished within 120 seconds and
left no owned processes. Latency excludes no-command/NONE steps; reset/resume
phases are separately retained in the JSON reports. No reliable throughput or
learning gain follows from these small samples. Direct B also recorded two rare
473/508 ms command latencies; those were not discarded. Profiling places the long
gap after observation packing and before policy inference. The precise cause is
unresolved, so overall responsiveness is not declared fully optimized.

New profiler events `env_step_total`, `step_callback`, `rollout_buffer_add`, and
`transition_ledger` locate future gaps without extra captures or GPU synchronization.

Evidence:

- `reports/readback_comparison_20261106_180910_820599.json`
- `reports/transfer_buffers_20261106_181124_174902.json`
- Baselines `latency_probe_20261106_175557_669394` and `latency_probe_20261106_180407_618753`
- Bulk-only `latency_probe_20261106_175753_128289`
- Direct `latency_probe_20261106_180142_559289` and `latency_probe_20261106_180626_196633`

## Evaluation

`evaluate_both.py` now accepts `--headless` to use the same GPU/shared-memory capture
path as training. Default remains the actual visible game with desktop capture.
Both modes write `evaluation_latency.json` with policy/guard request-to-command
statistics and decision cadence. It is a bounded 20,000-sample window, includes
startup/reset command timings, and excludes across-episode gaps from cadence.
Logging adds no captures and no keys; it does not claim to measure Unity's response.

Two-episode-per-map headless smoke check completed in 54.8 seconds:

| Map | Request→command median / p95 | Median decision interval |
|---|---:|---:|
| Halloween | 29.92 / 33.21 ms | 50.56 ms |
| Tokyo | 26.43 / 31.35 ms | 50.39 ms |

Report: `reports/multimap_headless_20261106_181537_282038`.
This establishes headless integration, not long-run tails or visible-mode latency.
Single-worker evaluation has less contention than four-worker training, so identical
capture settings do not imply identical achieved decision cadence. Real-time frame
ages and action timestamps remain in the policy input.

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$checkpoint = "reports\multimap_stability_20261106_171018_244012\stability\train\latest.zip"
# Same capture backend as training, both maps:
& $py -B -s -u .\development\experiments\multimap_classifier\evaluate_both.py --checkpoint $checkpoint --headless --episodes 15
# Actual visible game, both maps; includes its own latency report:
& $py -B -s -u .\development\experiments\multimap_classifier\evaluate_both.py --checkpoint $checkpoint --episodes 15
```

The GPU direct-write path is enabled by default for subsequent runs. To reproduce
the old path, set both `RS_GPU_DIRECT_SHARED=0` and `RS_GPU_BULK_COPY=0` in the child
environment. The latency probe explicitly sets these switches from its flags, so
omitting `--bulk-readback --direct-readback` remains a baseline measurement.

## Validation

- Native tightly packed and padded-row byte parity, including full-size frames.
- Native shared mapping checks: publication invalid throughout writing, header
  committed only on success, failures stay unpublished, close/error behavior.
- Existing Electron decoder/shared-frame tests: 15 passed.
- Collector/optimizer regression tests: 14 passed.
- Evaluation timing test: real command only, source timing preserved, reset gaps excluded.
- Live probes verify optimizer audits, clean shutdown and unchanged source checkpoint.

Run native tests with `node development/experiments/capture_pipeline/test_mapped_copy.cjs`
and `node development/experiments/capture_pipeline/test_direct_shared.cjs`.
The pinned-buffer benchmark is `transfer_bench.py --checkpoint ...`; it does not
install any runtime patch. `analyze.py REPORT_FOLDER ...` creates a new comparison
JSON from completed probes without rerunning games.

Keep RGB for this optimization pass. Grayscale changes the observation architecture
and discards color cues; it needs a separate learning comparison, not just a capture
speed claim. It would not automatically eliminate color-based guard/capture work.

Final instrumented probe `latency_probe_20261106_181753_546608` completed two
optimizer rounds (1,024 transitions) in 80.9 seconds including startup/cleanup,
with no remaining owned processes and the source checkpoint unchanged. Ordinary
request-to-command median/p95/max were 33.98/38.70/97.54 ms. The rare half-second
delay did not recur; it is not established as fixed. Maximum callback, buffer-add
and ledger times were 11.66, 3.82 and 1.30 ms respectively. Environment-step time
includes episode resets and must not be interpreted as command latency. This
shorter probe achieved 24.98 optimized transitions/s, so the copy improvements
do not establish an overall throughput increase.

## Next long run: spike retention

Training now attaches `latency_spike` to its existing worker-local
`learning_transitions.jsonl` for dispatched policy commands above 100 ms.
This continues throughout the run, even after the initial 100,000-event detailed
profile fills. Each spike includes request/reply-to-policy delay, policy host time,
policy-ready-to-command time and an episode-start marker. Up to 256 recent timing
events accompany a spike at most once per ten seconds; compact spike records are
not rate limited. Context includes callbacks, buffer insertion, ledger writes,
capture, guard and inference timings. Passive/no-command steps are not classified
as command spikes. The threshold is diagnostic only and does not change control.

Records use the existing per-transition flushed ledger write, with no additional
capture, file write or GPU fence. An ordinary exception preserves prior records;
power loss can still lose OS-cached writes, and a crash during a transition can
lose that unfinished transition. Final profiler metadata records spike/context
counts. Current multimap training already enables the profiler, so no new flag is
needed. Three spike tests and fourteen collector regression tests pass. This
instrumentation has not yet been validated in a long live run.
