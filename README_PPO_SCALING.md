# Opt-in scaling paths and detector ablation

These changes keep the existing rewards, five actions, four-frame image and
557-value auxiliary schema. Each optimization is independently selected; the
previous per-worker model/renderer and pipe transport remain defaults. Source
checkpoints and prior results are never overwritten. Use LOCAL Practice only.

**Speed-first update:** three workers with separate hosts/inference, dense
observations and shared-memory transfers passed actual updates at 29.9/s,
32/38ms steady median/p95 with detector off; the matched detector-on run was
27.5/s and 33/61ms. Do not choose shared inference/hosts merely to save memory.
See [the speed matrix and exact configuration](reports/SPEED_SELECTION_20261031.md).
`tools.analyze_ppo_speed RUN... --warmup-commands 20 --report NEW_DIRECTORY`
compares saved reports, retaining startup samples and separating latency from
throughput. It refuses to mix frozen collection rates with optimized rates.

The subsequent **15-minute three-worker detector-off run completed 30,336
optimized transitions at 33.57/s**, passed all three learning audits and held
steady request-to-command median/p95 at 31.6/38.3ms. Overall survival was 3.76s;
the latest 100 averaged 3.93s, which does not yet establish better policy quality.
See [the complete diagnostic](reports/SPEED3_DIAGNOSTIC_15M_20261031.md).
The subsequent one-worker five-minute diagnostic passed at 17.50 optimized/s
with 25.6/33.6ms steady median/p95. It is the lower-load background candidate;
three workers retain higher aggregate throughput. The user's three-worker
30-FPS-cap viewer check was stopped early, with no final audit/completion report.
See [the follow-up details](reports/SINGLE_WORKER_PREVIEW_20261031.md).

The terminal now prints `[PPO STATS]` after every shared update and at shutdown,
even without a preview server: completed episodes, `overall_mean`,
`recent100_mean`, `recent_n`, maximum survival, time-limit count, optimized/s and
elapsed time. Means pool completed episodes across all workers in the current
run, excluding previous runs and unfinished episodes. Time-limit episodes remain
included but are counted separately. Empty means show `n/a`. Statistics reuse the
existing background aggregator; `stats_age` shows cache age (normally a few
seconds). There are no extra captures or inference. Frozen diagnostics instead
print `[FROZEN STATS]` and `processed/s`.

## Parallel CLI options

| Option | Default | Behavior |
| --- | --- | --- |
| `--rollout-transport shared_memory` | `pipe` | One coordinator-owned slot per worker; pipes carry metadata. The worker cannot overwrite its publication until the matching optimizer commit. |
| `--compact-observations` | Off | Requires shared rollout transport. Store unique RGB frames with exact stack references; reconstruct only requested minibatches. Handles reset duplicates and entirely new resume stacks. |
| `--cache-detector-features` | Off | Reuse object geometry only for the same immutable detector snapshot. Recompute episode/age validity and action/timing inputs every observation. |
| `--opencv-threads 1` | Existing OpenCV setting | Limit nested CPU threads per worker. Does not change pixel geometry, model settings or guard deadlines. |
| `--shared-inference` | Off | Workers use a versioned policy/YOLO service and CPU observation slots. There is one service copy of each model, plus the coordinator's optimizer policy. |
| `--inference-timeout-ms 50` | 50 | With shared inference, expire a policy request if its round trip exceeds this limit. No action is dispatched from a late reply. Allowed range 1..100. |
| `--inference-batch-wait-ms 0` | 0 | Batch already-ready policy requests without intentional waiting. An optional wait is bounded to 2ms. Policy work is prioritized before another detector job. |
| `--electron-host shared` | `per_worker` | Multiple offscreen windows in one host, with separate storage partitions, frame slots, serials and client connections. Closing one worker must not close sibling windows. Requires headless Electron; detailed Node profiling is currently per-worker-host only. |
| `--detector-mode off` | `on` | Do not load, warm or schedule YOLO. Detector fields use the existing unavailable representation; RGB, action history/timing and the separate screen-state safety classifier remain enabled. |
| `--preview-fps 1` | 1 | Set the initial preview cap, 1..30. The dashboard provides 1/5/10/15/30 controls and metrics-only mode. |
| `--end-plot auto\|show\|save\|off` | `auto` | After successful training and worker cleanup, auto opens a Matplotlib survival curve if no dashboard was enabled. Show forces a window, save writes a PNG only, off disables plotting. Auto skips frozen benchmarks. |
| `--benchmark-only` | Off | Freeze policy weights and skip optimization/checkpoint writes. Count processed transitions separately; do not call this training throughput or a deterministic policy evaluation. |

`--detector-mode` and `--cache-detector-features` are also available in the
standalone trainer/evaluator. Explicit configs record the selected mode; new
trained checkpoints record detector mode. Disabling the detector changes model
inputs even though dimensions stay compatible. It can worsen an existing policy.
Do not silently promote a detector-off model or compare it with unmatched inputs.

## Plot the mean curve after training

Without `--preview-port`, successful training now opens a Matplotlib window by
default, **after** worker cleanup, final checkpoint/report saving and learning
audits. It plots the web dashboard's exact saved `survival.json` points: a line
with red dots for the latest up to 100 completed episodes across workers, sampled
every 10 seconds. The x axis spans the full requested runtime in minutes; the
y axis is pause-aware active survival in seconds. Earlier dots remain fixed.
Time-limit episodes are included, with their count displayed separately.

The figure is also saved as `survival_curve.png` in the run folder. Replotting
preserves prior images by using numbered filenames. Close the plot window to
return to PowerShell; training is already finished while the window is open.
Matplotlib is imported only after training, not by the worker/action loop.
Plot failures do not change successful training results or remove checkpoints.

Use `--end-plot show` to open it even with a dashboard, `--end-plot save` for PNG
only, or `--end-plot off` for unattended batches that should return immediately.
Frozen `--benchmark-only` runs skip automatic plots; explicit show/save still
works. Stopped/error runs do not automatically open a plot.

To plot a previously completed run without starting any training:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -m tools.plot_ppo_survival --report "reports\YOUR_RUN"
# Add --save-only to create a PNG without opening a window.
```

Post-run plotting, cached terminal statistics and parallel-learning regression
checks passed 17 tests using a noninteractive plotting backend; no live game
was launched for this display change.

## Ownership and learning invariants

Shared rollout messages carry the worker ID, policy version/hash, mapping identity
and published frame count. Consumer checks reject stale publications, wrong
versions, invalid frame references and mismatched array shapes/dtypes. A worker
waits for the coordinator's commit before publishing again. The coordinator
retains mappings until workers stop. Returns/advantages are computed per game;
no trajectory crosses worker IDs. Compact minibatches reproduce the exact pixels
and sample order; no image compression, downsampling or feature deletion occurs.

Shared policy replies carry the exact request serial, version/hash, value, sampled
action and its log probability. Passive and terminal bootstrap requests use the
critic without sampling an actor action. Late/mismatched responses stop safely.
Batching can change random-number consumption/numerical rounding relative to
independent replicas; log probabilities must still belong to the returned action
under that exact policy version. This is not an asynchronous/off-policy learner.
Shared-policy diagnostic probabilities are read from the completed policy forward
without another network pass or random draw. Their source/version/age is recorded
explicitly as the previous decision, not a prediction on the following observation.
Detailed worker profiling measures shared-inference client wall time; it does
not provide the remote GPU's kernel execution time.

Shared YOLO remains asynchronous to each game. Its replies retain the original
game/episode/request/source timestamp. The existing stale detector limit still
applies. A busy shared service may reduce detection freshness or add policy wait;
sharing is not assumed to improve latency without measurement.

Preview encoding runs in a background thread with one replaceable pending RGB
thumbnail. Slow encoders drop superseded previews. It never requests game frames.
A 30 FPS setting cannot produce 30 unique observations if the game collector
delivers fewer. Dashboard image requests have a separate cadence from metrics;
hidden previews stop encoding/fetching. Higher preview caps may still consume CPU
and I/O, so keep 1 FPS or metrics-only mode for throughput measurements.

## Detector on/off benchmark after training

Start the normal local server in a separate terminal if needed:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s server.py
```

In another terminal, choose a new report name and run the frozen-policy benchmark.
Default order is ON/OFF/OFF/ON to expose order variability. Every stage starts
from the SAME checkpoint; none trains or writes a new model:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$checkpoint = "reports\v612_sharedmem_4workers_2h_20261031_112933\latest.zip"
$report = "reports\detector_ab_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& $py -B -s -u -m tools.benchmark_ppo_detector `
  --checkpoint $checkpoint --workers 4 --minutes 1 `
  --rollout-transport shared_memory --compact-observations `
  --cache-detector-features --opencv-threads 1 `
  --report $report
```

Leave shared inference and a shared Electron host out of the first ablation;
measure each independently afterward by adding its flag. Source LR .0003,
128 steps per worker, five epochs and safety bounds remain unchanged. Four
workers still form 512 transitions per cohort. Wall limits finish a cohort and
exclude startup, so actual elapsed time is longer than four minutes.

Inspect comparison.json and each stage's worker result.json, reaction metrics,
survival.json and system_usage.jsonl. Different detector inputs can change actions,
episode lengths and reset rates, so this is an end-to-end ablation rather than an
isolated measurement of detector compute. Frozen-policy throughput excludes PPO
optimization. A short benchmark cannot establish improved learning. Follow with
matched evaluations before adopting a detector-off continuation.

## Validation status

Offline tests exercise exact dense/compact minibatch and optimizer-weight parity,
shared slot ownership, detector feature age/reset parity, detector-off construction,
latest-only previews and real spawned CPU policy-service values/log probabilities
and version updates. Live scaling measurements and failed attempts are recorded separately;
no 4-to-6-worker speedup or long-run stability is assumed from implementation.

Actual checks: the lighter four-worker storage/preparation bundle completed two
updates at 27.3 optimized transitions/s with 39/81ms pooled median/p95 command
latency. The shared-service/shared-host four-worker bundle passed the same budget
at 19.9/s with 59/118ms latency, using less sampled RAM/VRAM. Six shared-host workers
remain unreliable: startup capture timeouts persist, and the brief gameplay
attempt had poor reaction timing. Shared-host Unity startup is serialized, but
that is not a complete solution. Keep those paths experimental. Full evidence:
[dated report](reports/SCALING_PIPELINE_20261031.md).

Final offline validation: 211 focused pipeline tests and 100 existing project
tests passed. JavaScript syntax and Git whitespace checks passed. Full ON/OFF/OFF/ON
ablation and long-run stability of these new options remain untested.
