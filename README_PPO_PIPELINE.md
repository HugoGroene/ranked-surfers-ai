# PPO pipeline foundation

## 29 October: corrected learning transitions

Pause-aware training now defaults to the corrected collector. The monitored
manifest also passes `--settled-learning` explicitly. Use `--legacy-learning`
only to reproduce historical training semantics; it retains the audited gaps.
Training without optimizer pause remains the historical path. Evaluation still
loads either checkpoint through ordinary PPO; its reported returns use the
historical evaluation clock, so do not compare those returns with new training
returns. Compare gameplay survival under matched evaluation flags instead.

The correction settles boundary deaths before GAE/optimization and represents
post-resume refresh as an explicit passive transition. Passive and suppressed
actions have zero actor/entropy/KL contribution; value learning remains active.
No global input cooldown or scripted steering was added.

Gamma and lambda are now raised to `active_seconds / 0.05` per transition.
`--discount-reference-seconds` declares that unit; the stored gamma/lambda
coefficients retain their values. On resume the saved time unit is inherited
unless explicitly overridden; a legacy checkpoint starts with 0.05 seconds.
Survival rewards use confidently verified
gameplay time with pause UI transactions excluded. They no longer apply the
0.15-second per-decision cap on that verified clock, and terminal confirmation
does not advance it. Coefficients remain 0.8/second, 0.15 per 0.5-second
milestone and -1 death. This is a deliberate semantic correction, not a promise
of equivalent historical reward totals or an independent Unity timer.

Wall-time limits are soft boundaries: finish and optimize the current rollout,
then stop without resuming. Reports separate raw environment operations,
collected transitions, processed rollouts, unique optimizer-used transitions,
optimizer sample presentations, selected actions and dispatched keys. A guard
stop or interruption reports any discarded partial collection separately.

The authoritative audit is `learning_transitions.jsonl`, with protocol settings
and source hashes in `learning_protocol.json`. The old reward JSONL is a legacy
diagnostic and must not be fed to the old reward replay checker for corrected
runs. `tools.run_ppo_batch` selects the new audit automatically.

```powershell
& $py -B -m unittest discover -s tests -p test_ppo_settled_learning.py -v
& $py -B -c "from pathlib import Path; from tools.ppo_settled_learning import audit_settled_ledger; print(audit_settled_ledger(Path('ai_data/rl_v2_runs/YOUR_RUN/learning_transitions.jsonl')))"
```

See [implementation and live validation](reports/ppo_settled_validation_20261029/README.md).
These are correctness/safety gates, not evidence of better learned survival.
The exact final-boundary MENU case has offline collector coverage; live
validation exercised post-resume death, ordinary deaths, resets and deadline
completion. Keep longer training monitored while collecting more evidence.

The policy still learns autonomous control from survival/death rewards. No new
BC loss, action rewards, movement cooldowns or recurrent architecture were added.
Local Practice only. See the [validation report](reports/ppo_pipeline_foundation_20261028/README.md)
and [scaling assessment](reports/ppo_pipeline_foundation_20261028/SCALING.md).

## Longer episodes and unattended batches

The default external episode limit is **300 seconds**. Use
`--episode-max-seconds 15` explicitly for historical benchmarks. Default training
steps are now 4,096, divisible by the default 128-step rollout. Time limits are
reported separately from deaths; their terminal observation remains available
to SB3 for value bootstrapping. After a timeout the local page reloads without
movement, followed by the ordinary guarded MENU/R/GAMEPLAY startup.

`tools.run_ppo_batch` reuses one owned Chromium window and worker thread across
sequential stages. Each stage reloads the local game and creates fresh environment
and detector state. Failed or incomplete stages stop the batch. This is sequential
automation, not parallel environment collection. One initial foreground click may
still be necessary if Windows refuses activation. MSS requires the window to
remain visible and unobstructed. All browser resources close at the end.

Startup includes a temporary Windows input-queue handoff as another owned-window
activation attempt. It always detaches, sends no synthetic keys, and still
requires both OS foreground and browser focus. It never runs during gameplay.

## Optional faster observation guard

`--reuse-observation-guard` ties a confident gameplay reading to the policy's exact
image identity and capture timestamp. It is single-use and checked against a
20 ms age bound before use and again before dispatch. Changed/old images,
uncertainty, pending suspicion and menu handoffs fall back to the original fresh
capture/recheck path. Normal confidence/template thresholds remain unchanged.
The timestamp is the existing post-processing capture timestamp; actual pixel
grab age is measured separately and can be slightly older. Training with this
option requires `--pause-ppo-optimizer --resync-rollouts`.

This remains **opt-in**. The final short comparison measured median policy-image
age at dispatch of 28.22 -> 13.77 ms and median observation interval of
51.92 -> 40.19 ms. p95 image age improved, but rare fallback decisions still have
a p99 cost. These are sequential ten-episode latency diagnostics, not a learning
benchmark or Unity input-to-visible-response measurement.

Evaluation now gets action probabilities from its existing model forward rather
than running the CNN twice every eighth decision. Training retains only 10,000
recent debug timing dictionaries in RAM; full reward-step audits stream to disk.

## Saved PPO setting overrides

Saved settings otherwise win on resume. Change rollout size explicitly with
`--n-steps 512 --override-ppo-settings n_steps`. Supported names also include
`batch_size`, `n_epochs`, `learning_rate`, `gamma`, `ent_coef`, `target_kl` and
`seed`. No KL/entropy/learning-rate change was adopted by default.
`v61_effective_ppo_settings.json` records saved, requested and effective settings.
The buffer is rebuilt consistently and steps must divide by its effective size.
Image-buffer memory is about 216 MiB at 128 steps and 864 MiB at 512; other memory
is additional. Inference is unchanged, but individual optimizer pauses get longer.

## Monitored run template

The supplied template is ready for review; it was **not launched automatically**.
It uses the preserved pilot, 65,536 PPO steps, 512-step rollouts, the optional
faster guard and 300-second episode limits, followed by 20 monitoring episodes.
Use a separate matched 100-episode evaluation when claiming a new best model.
Run from the project root:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$tag = "v612_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& $py -B -u -m tools.run_ppo_batch `
  --manifest tools/ppo_batch_monitored.json `
  --run-prefix $tag `
  --report "reports\$tag"
```

The prefix gives every stage a unique output name and remaps references to new
checkpoints. Sources are hash-checked and outputs are never overwritten. The
runner checks training completion, pause/reward audits, guard stops and evaluation
completion. For a conservative baseline, copy the manifest and remove
`--reuse-observation-guard` from both stages. Keep learning changes controlled.
