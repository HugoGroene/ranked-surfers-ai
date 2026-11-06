# 384x384 five-frame shared-PPO worker benchmark

Opt-in wrapper around the existing shared PPO coordinator. No production or
earlier experiment source is modified. Each Windows child installs the same
road crop, spatial encoder and hybrid history before constructing its environment.

Configuration: 384x384 RGB, three recent observations plus 0.5s/1s context; actual
age and validity features; detector off; separate Electron GPU shared-texture hosts;
60 Hz requested capture; 30 Hz decision cap; 5 ms keypress; shared-memory dense
rollout transfer; one shared optimizer; per-worker inference; one Torch/OpenCV thread.
No dashboard, preview encoding or end-of-run plot. Existing freshness limits,
rewards, guarded pause/resume and settled-learning audits remain enabled.

## Start the local server (terminal 1)

If the server from the previous tests is still running, keep using it.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\server.py
```

Leave it running at `http://localhost:8000`. No public-site automation is used.

## Run the benchmark (terminal 2)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\development\staged\worker_scaling_20261102\benchmark_workers.py --execute
if ($LASTEXITCODE -ne 0) { throw "Worker benchmark stopped or contains failed cases. Preserve the printed report folder." }
```

Default source checkpoint (unchanged by this suite):
`reports/hybrid_capacity_20261102_193304_470308/performance/road384_recent3_context2/train/latest.zip`.
This is the three-minute capacity checkpoint with the matching five-frame schema,
not a migration of the old four-frame long-trained model. Every independent case
starts from the same checkpoint and uses the same base seed (distinct per worker).

Allow **about 40–50 minutes**, including startup and finalization. Scheduled active
budgets total **31 minutes**:

1. Offline regression checks, actual CPU shared-buffer/optimizer/save/reload check,
   checkpoint contract validation and Windows spawn validation; no game here.
2. One-worker, one-minute live smoke test. Any failure stops the entire suite.
3. **1, 2, 3, 4, 5 workers**, five active minutes each, with real PPO updates.
4. A five-minute one-worker repeat from the same original checkpoint to help spot
   order/load drift. Different tracks/policies mean this is only diagnostic.

Each stage writes separate logs, reports and atomic checkpoints. It finishes at a
complete shared update, so budgets can overrun. Worker startup is outside the
active training budget; total stage time is also recorded. The script checks for
competing trainers, including other copies of this benchmark, while excluding its
own Windows venv launcher. Avoid other training or heavy applications during it.

## What is measured

- Optimized unique transitions/s, processed transitions, audit results and saves.
- Per-worker policy request-to-command median/p95 and the **worst worker p95**.
  This is not a pooled percentile or Unity's visible response time.
- Buffered capture/preprocessing/guard/policy host timings and CUDA policy stream
  timing, with no added per-decision GPU synchronization. All cases use the same
  profiler; compare against the profiled one-worker baseline, not tiny differences
  from the earlier unprofiled capacity measurements.
- Requested versus achieved capture rates, stale-frame retries, history ages and
  mature-slot availability, guard failures, frame-state cache reuse.
- Per-worker rollout duration, cohort collection time and central optimizer time.
  Cohort collection includes waiting for workers and transport; it does not isolate
  pure socket/shared-memory transfer latency. Pure GPU readback is not isolated.
- System CPU load, busiest core, available RAM, GPU use/memory and worker/browser
  process CPU telemetry. These include other system activity; RAM/VRAM values are
  not per-worker private allocations.
- Explicit per-worker browser/pump cleanup status before another case can run.

The root `scaling_summary.md` is the readable table; `scaling_summary.json` includes
worker detail. Raw per-stage files include `parallel_training.json`, `config.json`,
`round_timings.jsonl`, `system_usage.jsonl`, `latest.zip`, and each worker's profile,
history, capture and learning-ledger files.

One worker collects 128 transitions per cohort; five collect 640. Larger cohorts
change PPO's update schedule. This measures the actual training system, not a pure
hardware benchmark or which worker count learns best. Each test trains independently
and can develop different behavior. Short passes do not prove overnight reliability.

## Failure and resume

A smoke failure stops immediately. Later independent cases can continue only if
all workers explicitly completed cleanup and no interrupt occurred. Missing cleanup
evidence stops the suite to avoid contaminating subsequent measurements. The script
never disables guards to make a worker count pass.

To stop manually, press Ctrl+C once and wait for finalization and the terminal prompt.
To resume, reuse the printed folder with identical code/checkpoint/budget:

```powershell
& $py -B -s -u .\development\staged\worker_scaling_20261102\benchmark_workers.py `
    --execute --resume "reports\hybrid_workers_REPLACE_WITH_PRINTED_FOLDER"
```

Completed, audited cases are retained; incomplete cases restart into a new folder.
No previous report or source checkpoint is overwritten. Omit `--execute` to inspect
a plan without launching games. `--check-processes` is a read-only diagnostic.

Choose the worker count from throughput **and** tail latency/history availability,
not throughput alone. No result automatically changes the production defaults.
