# Ranked Surfers PPO: pause/resume transition trace v1.5.3

## Why

The 512-step v1.5.2 run verified two actual frozen rollouts and clock corrections.
The first refreshed at 7.7 ms image age; the second stopped when the
screen-state classifier saw MENU immediately following the optimizer pause.
The existing log cannot tell when MENU first appeared. We must distinguish
an already-dead game at the first post-resume frame from a death that occurs
while four fresh frames are being collected.

**This is diagnostic-only, not a cure for spontaneous death after resume.**
It retains all classifier thresholds, fail-closed stops, resume/pause behavior,
and no synthetic rewards, steps, resets, or policy actions are introduced.
The original project/checkpoint stays untouched.

## Installation (PowerShell from Claude clone)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_resume_trace_v153.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p 'test_ppo_pause_resume_trace_v153.py' -v
& $py -B -m tools.trace_ppo_pause_resume_v153 --check
& $py -B -m tools.trace_ppo_pause_resume_v153 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

Only run `--apply` if the dry run matches expectations. Source backup is
`train_rs_v612_ppo.py.pre_ppo_pause_resume_trace_v153.bak`.

## One short live test (512 decisions)

Use the known-good original `$checkpoint`, `$state`, `$detector` variables:

```powershell
$run = 'v612_pause_trace_v153_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$log = "ai_data\ppo_pause_$run.log"
& $py -B -u train_rs_v612_ppo.py `
  --mode train --pause-ppo-optimizer --resync-rollouts `
  --resume $checkpoint --run-name $run --steps 512 --max-minutes 5 `
  --fps 30 --rate-mode uncapped --press-delay-ms 5 `
  --state-model $state --detector $detector --imgsz 960 `
  --detect-every 3 --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log
Select-String -Path $log -Pattern '\[PPO PAUSE TRACE\]|\[PPO RESYNC TRACE\]|\[PPO PAUSE\]|\[PPO RESYNC\]|Safe stop|Traceback'
```

What to send back: The `[PPO RESYNC TRACE]` lines, particularly which of the
four existing frames first sees MENU. The safe stop will also save the failed
frame to `ai_data\rl_v2_runs\<run>\ppo_resume_failure_epXXXX_frameN.jpg`.

*Note:* A first-frame MENU means death may have happened before the freeze
was established **or** immediately on unpause; the trace does not prove which.
A later-frame MENU points to death during fresh-stack capture. We can then
choose the smallest safe correction. Never continue training on a menu image.
