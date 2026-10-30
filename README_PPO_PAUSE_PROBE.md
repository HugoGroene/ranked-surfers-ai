# Pausing Unity between PPO rollouts — phase 0 proof

The local site's frontend includes a `Pause game` button, a `Resume game` button, and an overlay labelled `Game paused`; its Escape key handler also toggles the pause state. This is promising but **does not yet establish that Unity's internal simulation clock freezes**. Nor does it establish that episode reward duration excludes pauses.

This package deliberately makes **no modification** to the PPO trainer. It starts one local Practice game with the same environment setup as V6.1.2, captures scene motion, clicks the site's own Pause control, waits at least 1.2 seconds, captures again, resumes, and captures scene motion once more. It records screenshots and a JSON report in a new `ai_data/rl_v2_runs/v612_pause_probe_...` folder. It never trains, loads, modifies, or saves PPO checkpoints and never sends LEFT/RIGHT/JUMP/ROLL actions.

## Setup

With the trainer stopped, in the Claude clone:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_probe_v1.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p 'test_ppo_pause_probe.py' -v
```

Start the local Practice server in a second terminal (if not already running):

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
& "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe" -B server.py
```

Run one game pause probe in first terminal:

```powershell
& $py -B -u -m tools.probe_unity_pause `
  --state-model 'checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt' `
  --detector 'ai_data\v5_hazards\runs\segment_round04\weights\best.pt'
```

**Do not press keys while the probe runs.** It uses the browser's actual game-stage `Pause game` and `Resume game` buttons rather than Escape, because Escape may exit the browser's fullscreen mode. The script verifies the pause overlay and resumes it even on error; the browser is closed before exit. Do not run another trainer simultaneously.

Inspect the terminal summary and the JSON report. If `PASS_CANDIDATE` and screenshots show the scene truly frozen while paused, the next step is to instrument a dry-run PPO rollout-boundary integration in the trainer. If the result is inconclusive, inspect the screenshots rather than weakening any guard.

**Do not run an overnight PPO training job yet**: the trainer's episode and auxiliary-feature timestamps use real wall time; we must correct accounting for paused intervals before relying on PPO rewards. Never automatically treat paused UI as proof of paused Unity clock.

Do not stage `ai_data/` or trained checkpoints into Git.
