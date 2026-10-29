# Corrected PPO: learning-rate comparison

29 October 2026. Local Practice only. Status: **primary batch COMPLETE; repeats deferred; no setting adopted**.

## Primary results

All five stages completed, including 32,768 transitions per training arm and
100 deterministic episodes per evaluation. Both corrected transition audits
passed. No partial transitions were discarded; original source checkpoints
remained unchanged. Chromium closed cleanly after one launch and four reuses.

| Policy | Episodes | Mean survival | Median |
| --- | ---: | ---: | ---: |
| Fresh pilot baseline | 100 | 4.6586 s | 4.525 s |
| Learning rate 0.0001 | 100 | 4.7756 s | 4.330 s |
| Learning rate 0.0003 | 100 | 4.4793 s | 4.210 s |

Lower minus higher rate: +0.2963 s, episode-bootstrap 95% interval
[-0.1805, +0.7745] s. Neither continuation established an improvement over
baseline. The predefined close-result rule triggered; no setting is adopted.
See [primary_results.json](primary_results.json).

## Deferred repeat commands (retained for reference)

After reviewing the inconclusive primary results, the user questioned the value
of another long comparison. Repeats were deferred in favor of diagnosing existing
artifacts. The commands below are historical preparation, not the current next
action. Neither learning rate has been adopted. Rollout 512 was common to both
arms, rather than a winning setting established by this learning-rate comparison.

For jobs longer than a few minutes, the user runs the commands in separate
terminals; the assistant reviews saved results afterward without live polling.

Terminal 1, unless the Practice server is already running:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -u server.py
```

Terminal 2 (100 episodes each, higher-rate checkpoint first, then lower-rate):

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -u -m tools.run_ppo_batch `
  --manifest reports/ppo_lr_comparison_20261029/repeat_manifest.json `
  --report reports/ppo_lr_comparison_20261029/repeat_batch
if ($LASTEXITCODE -ne 0) { throw 'Repeat batch stopped; inspect saved logs.' }
& $py -B -s reports/ppo_lr_comparison_20261029/analyze_results.py --repeat
```

Click Chromium when it appears; keep it visible and unobstructed. The batch
reuses one window and performs no further training. Outputs refuse overwrite.
If a repeat fails, preserve it and diagnose before changing names or retrying.

The preserved small BC pilot is the common initialization. Each independent
arm receives 32,768 corrected transitions at rollout 512. Learning rate is
0.0001 versus 0.0003; batch 32, five epochs, gamma .995, lambda .95, entropy
.002, no KL stop, .05-second discount reference, verified-time rewards, input
settings, observation schema and detector settings remain fixed. No BC update
or recurrent model is introduced. Exact settings/hash: [plan.json](plan.json).

A new 100-episode deterministic pilot baseline precedes training. Each arm has
a matched 100-episode evaluation. The predefined close-result rule is an
absolute mean difference below .5 seconds OR a bootstrap 95% interval spanning
zero. In that case both checkpoint evaluations repeat for 100 episodes each,
in reverse order, before any setting is selected. Repeats do not retrain models.

The single persistent-browser batch runs baseline, lower-rate training/eval,
then higher-rate training/eval. Source models are hash-checked, no runtime source
changes are permitted during collection, and corrected transition audits gate
subsequent stages. Failed/incomplete stages stop. All outputs have unique paths.

This is one training seed, sequential unpaired procedural maps. Episode-bootstrap
intervals condition on these trained checkpoints and do not capture training-seed
or session variability. Equal corrected transition budgets include passive rows;
report active durations, actor counts, unique optimizer-used samples, runtime,
terminal handling and guard metrics alongside survival. Historical training
returns are not a matched control for corrected rewards/discounting.

Artifacts: [manifest.json](manifest.json), [code hashes](code_sha256.json),
and the batch directory once launched. No working model or raw demonstration
is overwritten. The result must determine whether either continuation actually
improves on the new baseline, not merely rank two regressions.
