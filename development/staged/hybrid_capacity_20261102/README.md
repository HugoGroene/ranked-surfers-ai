# Reduced hybrid capacity suite

This changes scheduling only. The active `hybrid_history_20261102` folder and all
production sources are untouched. It invokes the same probe code, guards, encoder,
rewards and capture backend as the larger suite.

First stop the old batch with **Ctrl+C once**, and wait until its child has finalized
and PowerShell returns to the prompt. Do not start competing benchmark processes.
The new runner refuses execution while a recognized Python trainer/batch is active.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\development\staged\hybrid_capacity_20261102\capacity_suite.py `
    --execute --reuse "reports\hybrid_short_20261102_190906_469806"
if ($LASTEXITCODE -ne 0) { throw "Capacity checks contain a failure. Preserve the printed report folder." }
```

It will discover stages completed since this command was prepared. Reports and
checkpoint integrity are checked before reuse, with identical probe sources and
matching configuration/seed. Only stages with confirmed successful cleanup qualify.
Incomplete stages restart in a new folder, leaving their original outputs intact.
Completed one-minute smoke stages and four-minute performance stages can satisfy
the shorter budgets; their real durations remain in the reports. This is a capacity
screen, not an equal-training-budget learning comparison.

## Work scheduled

1. Offline tests and CPU checkpoint checks first.
2. All nine history layouts at 256 and 384: 30-second smoke training and one-episode
   matching checkpoint evaluation, retaining already completed work. A full rollout
   and optimizer update may extend a short budget. Any smoke failure stops everything.
3. Three-minute performance runs for each of the following at both resolutions:

| Total frames | Representative |
|---:|---|
| 4 | four recent observations |
| 5 | three recent + 0.5s and 1.0s context |
| 6 | three recent + 0.2s, 0.6s, 1.0s context |
| 7 | four recent + 0.2s, 0.6s, 1.0s context |
| 8 | four recent + 0.2s, 0.4s, 0.7s, 1.0s context |

This is **39 scheduled training minutes from scratch**, plus startup, 18 short
evaluation episodes and finalization. Allow roughly **45–65 minutes**, less when
completed stages are retained. The runner prints remaining training budgets; this
is not a hard wall-time limit. No duplicate full performance probes per frame count.

Smoke trials retain the input gallery and timestamp/validity checks for *all*
spacings, including {0, 0.1, 0.5, 1.0}. Dedicated performance probes still exclude
preview collection. Independent performance failures continue only after confirmed
cleanup; interrupts or uncertain cleanup stop the suite.

Outputs in the printed `reports/hybrid_capacity_...` folder:
- `performance_summary.md`: latency median/p95, throughput and capture rate.
- `performance_summary.json`: audits, history availability, capture retries and
  resource sample paths.
- `policy_input_gallery.html`: actual smoke actor inputs, including retained ones.

To resume this reduced suite, use the same command with `--reuse` pointing to the
latest capacity folder. Omit `--execute` for a plan-only check. Do not use the old
full runner's `--resume` option to resume a capacity plan.

Choose frame count/resolution from these results and the desired history content,
then benchmark worker count for that chosen configuration. Neither these short
checks nor additional capacity repetitions establish which spacing learns best.
