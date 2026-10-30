# Ranked Surfers PPO — Rollout-boundary MENU handoff v1.7

**Exact base:** the six-source ZIP supplied by the user, with v1.6 reward-step auditing and v1.5.3 pause tracing installed. The installer **refuses** any other source SHA256. This package does not modify the original pretrained checkpoint, reward coefficients, screen-state classifier, reset key safeguards, Unity site, or action selection policy.

## What the failing log showed

At step 640, `pause_for_ppo_optimizer()` observed `MENU` before clicking the pause control and therefore correctly failed closed. The post-step death occurred just after the 128th action of the rollout, so it was not yet represented as a terminal step in the collected PPO buffer. Previously, immediately stopping (or using the old `--recover-rollout-menu` reset) could result in missing that death reward.

## v1.7 behavior

1. **Narrow trigger:** Only `Not verified GAMEPLAY at rollout end: MENU, menu=...` from the *pre-click* pause gate is eligible. Focus loss, freeze failure, uncertain screen, and overlay problems still stop training.
2. **Independent confirmation:** `confirm_post_optimizer_menu()` must confirm the real MENU using consecutive learned-state and independent template evidence. The original classifier/restart guards are unchanged.
3. **No unsafe pause or reset:** If confirmed, keep the dead game sitting on MENU during PPO optimization (no hazards run on MENU). Do not send `R`, movement, or call `VecEnv.reset()` inside the callback.
4. **Safe handoff:** After PPO optimization, independently reconfirm MENU. Adjust the existing monotonic/perf/wall/Monitor clocks by this menu-idle interval. Skip the gameplay-only four-frame resync **for this terminal MENU boundary only**.
5. **Actual terminal transition:** The next *normal PPO environment step* goes through the existing multi-screenshot input guard, which must return `TERMINAL` **without sending the sampled movement key**. It produces the actual `died=True` and V6.1 death penalty. If the screen changes to gameplay or is ambiguous, stop safely without pressing a key.
6. **Exclude menu idle from rewards:** The terminal `info.duration` is anchored to the FIRST pre-optimizer MENU observation; no optimization/menu/confirmation waiting time is counted as survival or as reward milestones. The logged reward audit includes `ppo_menu_handoff_terminal=true`.

**Known limitation:** The menu handoff accounts for the death on the first step of the **next** rollout. It does not retroactively reattribute the death to the prior rollout's last action. That one-step credit-assignment approximation remains; no claim is made that it is perfectly unbiased. A final unconsumed death at training shutdown is reported as a warning. The separate 0.15-second per-step survival cap and PPO discount semantics are unchanged.

## Install (PowerShell, stop training first)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"

Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_menu_handoff_v17.zip" `
  -DestinationPath (Get-Location).Path -Force

& $py -B -m unittest discover -s tests -p "test_ppo_menu_handoff_v17.py" -v
& $py -B -m tools.install_ppo_rollout_menu_handoff_v17 --check
```

**Only if all tests and the source-hash check pass:**

```powershell
& $py -B -m tools.install_ppo_rollout_menu_handoff_v17 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

The patch only replaces `train_rs_v612_ppo.py` and `tools/ppo_reward_timing_audit.py`, and adds `tools/ppo_rollout_menu_handoff_v17.py` and tests. The two original files are backed up as `*.pre_ppo_menu_handoff_v17.bak`.

## Controlled live test — 1,024 steps, not overnight

Local Practice server must be running and Chromium must remain visible, focused, unobstructed.

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_menu_handoff_v17_" + (Get-Date -Format "yyyyMMdd_HHmmss")
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

Do **not** set `--recover-rollout-menu` for this test; that legacy mode resets without preserving death rewards.

## Review

```powershell
Select-String -Path $log -Pattern "\[PPO MENU HANDOFF\]|\[PPO PAUSE\]|\[PPO RESYNC\]|Safe stop|Traceback"

& $py -B -m tools.ppo_reward_timing_audit --log $log --min-resumes 7 --min-steps 1024

$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit --reward-steps $rewardLog --min-steps 1024
```

The log audit now handles Windows PowerShell `Tee-Object` UTF-16LE correctly, counts independently confirmed boundary-MENU handoffs as verified optimizer continuations, requires terminal reward evidence after every completed handoff, and keeps all existing pause/fresh-frame checks. For a successful boundary handoff, look for:

- `[PPO MENU HANDOFF] Boundary MENU confirmed ...`
- `[PPO MENU HANDOFF] MENU reverified after optimizer; menu_idle_seconds_excluded=...`
- `[V612 GUARD] MENU confirmed ... NO key sent`
- `[PPO MENU HANDOFF] Terminal recorded by guarded env.step; death_reward=-1.0000; NO movement key sent.`

If no MENU occurs at a rollout boundary, the existing pause/resume/resync behavior is unchanged. If the run stops for any other reason, do not restart from the partial checkpoint until logs are reviewed.

## Rollback

```powershell
Copy-Item -LiteralPath "train_rs_v612_ppo.py.pre_ppo_menu_handoff_v17.bak" `
  -Destination "train_rs_v612_ppo.py" -Force
Copy-Item -LiteralPath "tools\ppo_reward_timing_audit.py.pre_ppo_menu_handoff_v17.bak" `
  -Destination "tools\ppo_reward_timing_audit.py" -Force
```

The helper and tests can remain on disk after restoring the original sources. Do not push the whole working directory; it contains backups, demo data and experimental files.
