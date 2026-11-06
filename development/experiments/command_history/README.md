# One-second command history for the selected fixed4 model

Opt-in extension only. No existing trainer, reward, guard, capture, image encoder,
or image-history file is edited. The existing four-frame source checkpoint stays
unchanged. Continue through this directory's entry points so all Windows workers
install the matching input contract.

## Inputs and recording

- Keep the existing 565 auxiliary values, including image ages/validity.
- Append 32 chronological entries, each five-way one-hot action + age + validity:
  224 additional values, for **789 auxiliary values** total.
- Ages are relative to the exact policy image's monotonic request-start timestamp.
  Events newer than that observation are excluded; events older than one second
  are excluded. This is not a Unity simulation clock or proof of game movement.
- Record successful keyboard-press dispatch-start timestamps from the existing
  timing record. Record guarded no-op decisions too. Blocked actions, passive
  observations and reset/UI commands are not added. Repeated actions are retained.
- Right-align oldest-to-newest entries; missing entries are zero with validity=0.
  There is no startup wait. Each environment owns its own buffer.
- Clear on episode reset or uncertain continuity. Across a verified same-episode
  optimizer pause, retain image and command history and shift saved timestamps by
  exactly the verified frozen duration. Restore only after successful guarded resync.
  Active pause-entry/resume time still ages history; newest guard/capture timestamps
  are never shifted. In-flight pre-pause captures are rejected by generation IDs.
  `command_history.json` includes restoration/discard counts and frozen seconds.
- Retain at most 32 events. Overflow drops the oldest and increments telemetry;
  inspect `worker_*/command_history.json` after the run rather than assuming no
  overflow merely because the decision cap is 30 Hz.

Images stay 384x384, fixed ages [0, .05, .25, .75] seconds, same spatial CNN.
Headless GPU capture, detector-off mode, 5 ms press duration, guards, rewards,
128 steps per worker, minibatch 32, 5 epochs and learning settings are unchanged.

## Adapted checkpoint

Current continuation checkpoint (after the successful ten-minute diagnostic):
`reports/command_history_pause_retained_20261103/latest.zip`.

It retains all 506,880 trained transitions and 4,950 recorded update epochs.
Weights and optimizer state are bit-for-bit equal to the diagnostic's final model;
only the explicit input contract records the new pause-retention semantics.
Its migration report is saved alongside the checkpoint. Seventeen offline history
and adapter tests passed, including repeated pauses, same-episode validation,
uncertain resync rejection and unchanged latest-capture timestamps. Live validation
of pause retention remains pending. Earlier v1 checkpoints require explicit migration
with `upgrade_pause_checkpoint.py`; they are not silently accepted under v2 semantics.

Original auxiliary-expansion checkpoint (v1, before the diagnostic):
`reports/command_history_adaptation_20261103/latest.zip`.

Source: `reports/history_training_compare_20261102_234842_138252/fixed4_train_w4/latest.zip`.
The migration expands the 565-to-96 input layer to 789-to-96 and initializes only
the new columns to zero. Existing parameters and Adam moments/step counts are
copied exactly; new moment columns start at zero. Added parameters: 21,504.
Training counters remain 488,448 transitions and 4,770 recorded update epochs.
Optimizer history is retained, so new columns share the existing tensor's Adam
step counter; this is explicitly not a freshly initialized optimizer.

CPU parity on 24 real saved actor observations (with arbitrary nonzero appended
fields): identical deterministic actions, maximum probability difference
1.1920928955078125e-7 and value difference 2.384185791015625e-7, before and after
save/reload. These are sampled checks, not an exhaustive claim. See the saved
`latest.migration.json` report. CPU checks do not establish live CUDA latency.

Eight offline history/adapter tests passed. Additional integration checks verified
exact preservation of old weights and optimizer state, rejection of the old input
contract, four-worker shared-memory rollout merge, a synthetic PPO update that
changes the new columns, atomic save/reload and Windows spawned adapter setup.
No gameplay was launched while implementing this extension.

## First live check: 10 minutes, four workers

Keep the local Practice server (`python -B -s -u server.py`) running separately.
Then:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\development\experiments\command_history\run_training.py `
    --checkpoint "reports\command_history_pause_retained_20261103\latest.zip" `
    --workers 4 --minutes 10
if ($LASTEXITCODE -ne 0) { throw 'Command-history training stopped; preserve the printed report folder.' }
```

No dashboard. Atomic checkpoints about every two active minutes and at finalization;
graphs save at the end without blocking. The printed root contains `train/latest.zip`,
`train/parallel_training.json`, per-worker command-history and capture telemetry,
resource usage, reaction profiling and learning audits. A new run never overwrites
the adapted starting checkpoint. Ctrl+C once and wait for cleanup.

Review the short run's learning audits, command overflow, reset counts, capture and
command latency, throughput and final checkpoint before a long continuation.
Use `--minutes 260` for 4h20m, or up to 2880 minutes, and point `--checkpoint` at the
latest compatible saved checkpoint to retain additional learned updates. A fresh
output directory and wall-time budget are used each invocation.

For a fair unchanged-versus-extended continuation comparison, start both arms from
the original source and its zero-column adapted copy with equal training budgets.
Do not compare an additionally trained model against an untrained continuation and
attribute all gains to history. No performance or learning improvement is claimed yet.

## Matching evaluation

Use the extended wrapper, not the old fixed4 evaluator alone:

```powershell
$evalOut = "reports\command_history_eval_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $py -B -s -u .\development\experiments\command_history\evaluate.py `
    --layout fixed4 --checkpoint "PATH_TO_NEW_TRAIN_LATEST.zip" `
    --output $evalOut --episodes 300 --minutes 180
```

Add `--visible-browser` only for qualitative visible evaluation. The default
headless evaluator matches the training capture backend. Checkpoint compatibility
checks reject the original 565-value checkpoint in this extended environment.
