# Four fixed-spaced frames: worker scaling

Short performance/stability benchmark, **including actual PPO updates** so optimized
transitions/s has the same definition as the previous five-frame worker test.
This is not a long learning comparison. No separate five-frame sweep is included.

- Road crop, 384x384 RGB, existing spatial encoder and late-fusion auxiliary MLP.
- Four timestamp-selected frames, requested ages newest-first **[0, 0.05, 0.25, 0.75] seconds**.
- Stored oldest-first: approximately [0.75, 0.25, 0.05, 0].
- Same 60 Hz capture target, 30 Hz decision cap, GPU shared-texture capture, shared-memory
  rollouts, separate Electron hosts and per-worker inference as the prior benchmark.
- Detector off, 5 ms keypress, dashboard off, no final plot.
- Existing guard limits, rewards, pause/resume, timed discounting and learning audits.
- One fresh compatible checkpoint (seed 123), reused unchanged as the source of every case.
  No migration of the previous five-frame checkpoint, even where image dimensions could match.

The existing timestamped buffer selects the latest available capture **at or before** each
requested age. Its 60 ms selection tolerance is unchanged. For example, a requested
50 ms slot can be 60-80 ms old when captures are irregular; under worse conditions it
can be up to 110 ms old or marked unavailable. Actual ages and validity accompany every
frame. The report also includes target-versus-actual age deviations. This is request-time
history, not independently verified Unity simulation time.

Missing startup/reset/pause history is zero-padded and marked invalid; the policy does
not wait for 0.75 seconds. No older episode or pre-pause history crosses a reset boundary.

## Run

If needed, start the local server in a separate terminal and leave it running:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\server.py
```

Benchmark terminal:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\development\staged\fixed_spacing_workers_20261103\benchmark_workers.py --execute
if ($LASTEXITCODE -ne 0) {
    throw "Worker benchmark contains a failure. Preserve the printed report folder."
}
```

Allow approximately **40-50 minutes**, with **31 scheduled active minutes**:

1. Offline history/flags/process checks, CPU model initialization, exact shared-buffer
   validation, tiny synthetic CPU optimizer/save/reload check, Windows spawn validation.
2. One-worker, one-minute live smoke. Any failure stops the entire suite.
3. 1, 2, 3, 4 and 5 workers, five active minutes each, real PPO updates.
4. Five-minute one-worker repeat, starting from the same initial checkpoint.

The smoke retains up to six actual actor observations in memory and exports them after
the game closes. Performance stages do not copy/export these samples. The root
`policy_input_gallery.html` links the smoke images and exact auxiliary data.

Each stage saves its report/log/checkpoints independently. Interrupted or failed cases
do not overwrite completed ones. A later failure continues only when complete cleanup
is verified; unknown cleanup stops the batch. Ctrl+C once and wait for finalization.
To retain completed cases and rerun incomplete ones with identical code and budget:

```powershell
& $py -B -s -u .\development\staged\fixed_spacing_workers_20261103\benchmark_workers.py `
    --execute --resume "reports\fixed_spacing_workers_REPLACE_WITH_PRINTED_TIMESTAMP"
```

## Interpret results

`scaling_summary.md` and `.json` report optimized transitions/s, each worker's latency,
worst worker p95, collection and optimizer timing, capture rate, history availability,
guard failures, learning audits, source integrity, cleanup and system resource use.
Latency is request-to-command issue, using the final up-to-1000 command samples per
worker. It is not Unity response time. All cases use the same profiler.

The benchmark protocol and throughput definition match the prior five-frame test.
The input schema and initial checkpoint differ: this model is fresh, whereas the prior
suite used a three-minute capacity checkpoint. Gameplay trajectories, reset frequency,
machine load and growing PPO cohort sizes affect results. Treat comparisons as practical
system measurements, not a perfectly isolated causal estimate of frame-count cost or
evidence of better learning. Five minutes cannot establish overnight reliability.

The four-frame stack has 20% fewer image bytes than five frames at the same size.
Only the first convolution's input-channel work reduces directly; later CNN layers,
full-canvas capture and safety classification remain. A 20% overall speedup is not promised.
