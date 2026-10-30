# PPO v1.7.1 — Pause-control disappearance at a rollout boundary

## Why this exists

The live v1.7 run completed 896/1024 steps, verified 6 pause/resume cycles with clock correction and fresh 4-frame observations, then stopped:

```
[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.141 ml_gameplay=1.000
[PPO PAUSE] Safe stop at rollout-end: Unique visible site Pause game button missing
```

The *pre-click* image was classified as gameplay. The site can subsequently remove the pause control when changing state. The log alone does not prove whether this was a death/menu transition or temporary UI inconsistency.

The narrowly scoped v1.7.1 patch performs at most 8 additional **no-input** checks ONLY when `Pause game` is missing/hidden. Each check revalidates browser focus, reads the control's count/visibility, and takes a fresh classifier+template capture.

* If the unique button reappears with two latest independent positive `PLAY` frames, it can take the ordinary pause/overlay/freeze-verification path.
* If four latest consecutive frames corroborate `MENU` with the existing ML plus independent template rule, it saves an early menu-duration anchor and raises the **existing v1.7 boundary-MENU handoff** signal; that callback independently reconfirms the menu, waits through the optimizer, and relies on the guarded next `env.step` for the real `-1` terminal penalty (no movement key).
* Otherwise, it fails closed, writes `ppo_pause_control_epNNNN.jpg`, and never clicks the Pause button or sends movement/restart keys.

**No changes:** rewards, guard/classifier thresholds, PPO actions/optimizer, original checkpoint, `--pause-ppo-optimizer` opt-in, or normal pause path. Terminal reward attribution is still on the next rollout step in the handoff case; v1.7 limitation remains.

## Required precondition

Your local `train_rs_v612_ppo.py` must be **exactly the previously installed v1.7 file**. The patcher verifies its SHA-256. Do not restore the tracked Git version or the v1.6 backup. The archive contains a helper and an exact-block patcher, not a replacement trainer.

## Install & offline checks (PowerShell)

Stop training first. In the Claude clone:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_control_v171.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_pause_control_v171.py" -v
& $py -B -m tools.patch_ppo_pause_control_v171 --check
& $py -B -m tools.patch_ppo_pause_control_v171 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

The installer creates `train_rs_v612_ppo.py.pre_ppo_pause_control_v171.bak`. It refuses if hashes differ or the backup already exists. If a check fails, **do not apply**; share the exact error instead.

## Short live test (1,024 steps, fresh run)

Run local Practice `python server.py` separately. Keep Chromium foreground/visible and the capture unobstructed.

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_pause_v171_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = "ai_data\ppo_pause_$run.log"
& $py -B -u train_rs_v612_ppo.py `
  --mode train `
  --pause-ppo-optimizer `
  --resync-rollouts `
  --audit-reward-timing `
  --resume $checkpoint `
  --run-name $run `
  --steps 1024 `
  --max-minutes 6 `
  --fps 30 `
  --rate-mode uncapped `
  --press-delay-ms 5 `
  --state-model $state `
  --detector $detector `
  --imgsz 960 `
  --detect-every 3 `
  --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log
```

Keep `--recover-rollout-menu` **off**. Never resume from the partial 896-step checkpoint for this test: use the original pretrained input, to isolate the change. Each new run saves to its own run directory.

### Inspect / audit

```powershell
Select-String -Path $log -Pattern "\[PPO PAUSE CONTROL\]|\[PPO MENU HANDOFF\]|\[PPO PAUSE\]|\[PPO RESYNC\]|Safe stop|Traceback"
& $py -B -m tools.ppo_reward_timing_audit --log $log --min-resumes 7 --min-steps 1024
$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit --reward-steps $rewardLog --min-steps 1024
```

Expected if UI gap recurs:
- `[PPO PAUSE CONTROL] Pause button reappeared with two positive GAMEPLAY frames ...` followed by ordinary verified freeze, **or**
- `[PPO PAUSE CONTROL] Missing button; MENU corroborated ...` followed by v1.7 `[PPO MENU HANDOFF] ...` and later actual `death_reward=-1.0000` without movement.

If neither can be proved within the bounded window, stopping is correct and the diagnostic JPEG can help troubleshoot. If the run does not encounter a missing button, successful completion demonstrates stability but does **not** prove that specific race path live.

As before, zero reported episode-clock drift is an internal consistency check, not independent evidence of Unity time. Step reward arithmetic replay is not a proof of unbiased PPO discounting or the 0.15-second reward cap; those are separate evaluation topics. No overnight run until stability and reward audit pass.
