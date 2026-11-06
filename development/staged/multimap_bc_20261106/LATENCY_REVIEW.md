# Latency review, 6 November 2026

No live benchmark or training code was changed for this review. These are saved
measurements and proposed experiments, not claimed new speedups.

## What is measured

`reports/latency_phases_20261106_165418_723679.json`, four-worker probe
`reports/latency_probe_20261106_164755_906174`:

| Phase | Samples | Median request → command | p95 |
|---|---:|---:|---:|
| Ordinary decisions | 1,402 | 33.85 ms | 39.94 ms |
| First decision | 4 | 100.64 ms | 102.03 ms |
| After reset | 19 | 41.33 ms | 79.52 ms |
| After optimizer resume | 7 | 72.83 ms | 74.97 ms |

The first episode is now excluded from PPO. Other reset/resume phases still matter.
Few boundary samples cannot characterize a stable p95. Other short repetitions had
worse tails, so do not treat this table as a sustained guarantee. The ongoing
30-minute run is the appropriate next source of evidence, separated by map/worker
and phase rather than only pooled throughput.

Request-to-command excludes the pixels' age before the request, browser input
processing, Unity simulation, and visible response. Approximately 15 decisions/s
also means roughly 67 ms between decisions; a hazard appearing just after one
observation waits for the next. Neither a 60 FPS capture target nor a 34 ms median
proves high-speed coin sequences are controllable. The relevant measurement is
freshness plus decision cadence plus command-to-effect delay, particularly the tails.

## Priorities after the current run

1. **Remove redundant guard preprocessing/copies.** Saved isolated guard timings
   show preprocessing costs more than its small classifier forward pass. Inspect
   BGRA conversion, guard composition, and resize reuse on the exact delivered
   frame. Require identical guard pixels/decisions on both maps before a live A/B.
   The threaded policy crop reuse is already implemented; do not count it twice.

2. **Reduce GPU-to-CPU readback and blocking waits.** Our D3D bridge still copies
   into a staging texture, performs blocking `Map`, and copies rows into a CPU
   buffer. Shared-texture capture therefore does not make our complete pipeline
   zero-copy. Candidate changes: GPU crop/resize before smaller transfers, or a
   bounded staging-buffer ring with completion fences. Preserve the full guard
   image or validate a separate guard path. Never consume an older completed frame
   merely to improve a timing number. Electron documents GPU texture import as the
   route avoiding CPU transfer, requiring native integration:
   [Electron offscreen rendering](https://www.electronjs.org/docs/latest/tutorial/offscreen-rendering).

3. **Reuse transfer buffers and benchmark asynchronous copies.** Preallocated pinned
   CPU buffers may help where CPU→CUDA transfer is measurable. Protect buffer
   lifetimes and stream ordering; `non_blocking=True` does not remove dependencies.
   Pinning fresh memory on each decision can be slower, as the official tutorial
   demonstrates: [PyTorch transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

4. **Optimize inference only after capture costs.** Fixed shapes make compilation,
   CUDA graphs, or channels-last worth isolated tests, but policy inference is
   already only a few milliseconds. Warmup must finish before gameplay. Check
   logits/action parity and GPU memory under all four workers, not only batch-one
   microbenchmarks. `inference_mode` alone previously saved only about 0.06 ms in
   `reports/inference_context_20261106_163051.json`.
   [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

5. **Consider centralized inference only with a bounded wait.** A larger inference
   batch can improve total throughput while making the first queued worker wait.
   It must win per-worker end-to-end p95/freshness, not merely GPU examples/s.
   Additional keyboard threads are unlikely to help the already short dispatch
   handoff. Keep four workers until five-worker tails improve.

Start with saved telemetry and CPU pixel-parity checks. Then use short alternating
A/B runs with the same checkpoint/maps/worker count, including at least an optimizer
pause/resume and full cleanup. Compare ordinary, reset and resume latency separately,
decision cadence, frame age, dropped/rejected frames, memory and unique optimized
transitions. Do not change guard thresholds, rewards or model architecture in the
same latency experiment. A sustained check is needed only after a short test wins.

This is a plausible optimization order, not evidence that latency alone explains
the previous policy regression or that any proposed change will improve learning.
