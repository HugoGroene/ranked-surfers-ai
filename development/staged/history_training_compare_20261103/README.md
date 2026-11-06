# Fresh four/five-frame comparison: 260 minutes per arm

Both arms use fixed, increasingly spaced capture ages (the user's "log scale"
schedules). Neither uses a variable three-recent-observation stack.

| Arm | Workers | Requested frame ages, newest first | Training budget |
|---|---:|---|---:|
| fixed4 | 4 | 0, 0.05, 0.25, 0.75 seconds | 260 minutes |
| fixed5 | 3 | 0, 0.05, 0.20, 0.45, 0.95 seconds | 260 minutes |

The batch runs sequentially on this machine: both short smokes first, then fixed4,
then fixed5. It creates a separate fresh seed-123 PPO checkpoint for each shape.
The smokes use copies of those initial weights; their trained outputs are **not**
used to initialize the long runs. No old trained checkpoint or BC is applied.
Evaluation is intentionally deferred until both models are available.

## Settings

- 384x384 padded road crop, existing spatial CNN, auxiliary MLP and shared actor/critic
  feature extractor. Four images = 12 channels and 565 auxiliary values; five images
  = 15 channels and 567 auxiliary values.
- Four recent one-hot action entries, existing observation intervals and time-since-action
  features, plus actual selected frame ages and validity flags through the auxiliary MLP.
  These are not a complete timestamped action sequence over the entire image-history span.
- Timestamped capture selection at or before each target, existing 60 ms selection
  tolerance; missing history zero-padded and invalid, without a startup waiting period.
  Episode and optimizer-pause history isolation unchanged.
- Detector off; separate Electron GPU shared-texture hosts, shared-memory capture and
  dense rollout transfer, per-worker policy inference, one central optimizer.
- Capture target 60 Hz, decision cap 30 Hz, keypress 5 ms. Achieved rates are measured.
- 128 transitions per worker; aggregate cohorts 512 and 384 respectively. Batch size
  32, 5 epochs, LR 0.0003, entropy coefficient 0.002, gamma 0.995, timed discounting,
  existing guards/rewards/settled-learning fixes. One Torch/OpenCV thread per process.
- No dashboard or preview encoding. Profile/telemetry settings match the short tests.
  Curves are saved to PNG after completion (`--end-plot save`); no plot window blocks
  the second run. Terminal PPO means and training updates remain enabled.
- Atomic checkpoints about every two active minutes, at completed updates, plus final
  checkpoints when finalization permits. Initial checkpoints and earlier runs remain intact.

## Run

Start the existing local Practice server if it is not already running, in terminal 1:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\server.py
```

Run the comparison in terminal 2:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"

& $py -B -s -u `
    .\development\staged\history_training_compare_20261103\run_comparison.py `
    --execute --minutes 260

if ($LASTEXITCODE -ne 0) {
    throw "Comparison stopped or contains a failed stage. Preserve the printed report folder."
}
```

Expected duration: **about 8 hours 45 minutes to 9 hours**: 8h40m active training,
two one-minute smokes, startup, finalization and complete-update budget overruns.
Both offline preflights and both live smokes run before long training. Any smoke failure
stops everything. A later independent arm can continue after a failure only if worker
cleanup is explicitly confirmed. Uncertain cleanup, source changes or manual interruption
stop the batch. No guard is weakened or bypassed.

## Outputs and resume

The printed root folder contains:

- `plan.json`: exact settings, initial checkpoints, budgets and stage commands.
- `training_comparison.json` / `.md`: per-arm status, recovered-plus-current time,
  optimized transitions and final checkpoint/report paths. Evaluation remains pending.
- `scaling_summary.json` / `.md`: detailed per-stage performance and audits.
- `policy_input_gallery.html`: exact smoke-test actor inputs with ages and validity.
- Per-stage logs, `latest.zip`, `latest.prev.zip`, numbered checkpoint snapshots,
  learning ledgers, resource telemetry, runtime/survival statistics and saved curves.

Ctrl+C once, then wait for cleanup and the terminal prompt. A hard power loss cannot
save unsaved updates; the existing atomic checkpoints preserve earlier work.

Resume with identical source code and the same original budget:

```powershell
& $py -B -s -u `
    .\development\staged\history_training_compare_20261103\run_comparison.py `
    --execute --minutes 260 `
    --resume "reports\history_training_compare_REPLACE_WITH_PRINTED_TIMESTAMP"
```

Resume creates a new output folder and retains validated completed stages. Failed
smokes restart from fresh weights. Incomplete training resumes the newest intact,
compatible checkpoint for which saved time accounting is available. Only wall time
corresponding to recovered model updates is deducted; lost work can therefore be
repeated. If nothing was saved, it retries that arm from its original input checkpoint.
Unverifiable checkpoint/time combinations stop for review instead of silently starting
over. A failure at the budget boundary requires a minimum 15-second continuation to
obtain a clean completion, potentially exceeding the requested budget slightly.
History/environment restarts on resume; this is model/optimizer recovery, not a
bit-for-bit recreation of the interrupted game trajectory. Segments keep separate curves.

## What the comparison can establish

Equal active wall time answers which **complete configuration** learns faster on this
machine. Frame count, spacing, worker count, cohort size and samples collected differ;
this does not isolate only the effect of one extra image. Same seed does not mean
identical initialized weights across different shapes or identical game tracks.
One seed per arm is a useful first comparison, not a definitive winner across seeds.
Compare matched evaluations afterward, with each checkpoint's exact input contract.

Long-run pipeline profiling retains up to 100,000 events per worker, so its detailed
stage timings describe the retained early samples. Metadata records dropped events.
End-of-run reaction summaries use the final up-to-1000 command samples, while resource
telemetry, per-round update/collection timing and training statistics cover the run.
Command issue latency is not independently measured Unity response latency.
