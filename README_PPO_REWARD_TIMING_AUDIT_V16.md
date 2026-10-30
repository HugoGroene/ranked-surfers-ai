# Ranked Surfers PPO v1.6 — reward-clock regression + stability audit

**Scope:** no lane-change work. This is an opt-in, *observation-only* addition to
working v1.5.3. It does not modify PPO actions, reward coefficients/formula,
optimizer pause/resume logic, screen-state classifier, restart policy, or
checkpoints. The original repository/pretrained weights are not modified.

## What was independently checked offline

- The existing `rs_v61_rewards.ProgressRewards` implementation was run on
  synthetic **active-gameplay-only** timelines and on equivalent timelines with
  three long UI pauses. The two timelines produced identical per-step rewards,
  milestones, and death penalties after `compensate_pause_clocks`.
- Reward milestones were not awarded from a pause alone, including when a
  pause occurred close to a 0.5-second milestone.
- `episode_start`, the V6.1 action timer, and SB3's Monitor clock were adjusted.
  Old YOLO detection timestamps stayed old (not falsely refreshed).
- Reward reset after a death was tested; invalid clock inputs failed without
  partially mutating state.
- A read-only log audit correctly accepted the successful v1.5.3 512-step
  transcript, and rejected the earlier v1.5.2 safe stop. It recognizes a
  final frozen rollout without a corresponding resume because `model.learn`
  ends after the final update.

**Limitations:** None of this independently measures Unity's *internal* game
clock. The V6.1 reward's intentional `max_rewarded_step_seconds=0.15` cap
can under-credit long active intervals. PPO's gamma discount is per decision,
not per physical second. This package does **not** silently change either.
The following controlled run quantifies real capped intervals.

## 1 — Install audit files (in Claude clone ONLY)

Keep training stopped. Work from:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"

Expand-Archive `
  -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_reward_timing_audit_v16.zip" `
  -DestinationPath (Get-Location).Path -Force

& $py -B -m unittest discover -s tests -p "test_ppo_reward_timing_audit_v16.py" -v
& $py -B -m tools.ppo_reward_timing_audit --self-test
& $py -B -m tools.install_ppo_reward_step_audit_v16 --check
```

Review dry run; if it reports exactly the new opt-in audit hook and CLI flag:

```powershell
& $py -B -m tools.install_ppo_reward_step_audit_v16 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

Installer refuses unknown sources and double installation. Exact-source backup:
`train_rs_v612_ppo.py.pre_reward_timing_audit_v16.bak`. Default `--audit-reward-timing`
is **OFF**. No keybinds or checkpoint loading change.

## 2 — Controlled 2048-step PPO trial (NOT overnight)

Keep Practice server running, Chromium focused and unobstructed. Resume from
original read-only 4.61-second pretrained checkpoint, **never** from a short
experimental smoke test:

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_reward_clock_v16_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = "ai_data\ppo_pause_$run.log"

& $py -B -u train_rs_v612_ppo.py `
  --mode train `
  --pause-ppo-optimizer --resync-rollouts --audit-reward-timing `
  --resume $checkpoint --run-name $run `
  --steps 2048 --max-minutes 8 `
  --fps 30 --rate-mode uncapped --press-delay-ms 5 `
  --state-model $state --detector $detector `
  --imgsz 960 --detect-every 3 --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log
```

Do not enable `--recover-rollout-menu` in this test: a death during resume must
remain visible as a safe stop, not an abandoned episode without death reward.

## 3 — Run two independent checks on the completed files

```powershell
& $py -B -m tools.ppo_reward_timing_audit `
  --log $log --min-resumes 15 --min-steps 2048

$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit `
  --reward-steps $rewardLog --min-steps 2048
```

Target outcomes:

- `PASS_STABILITY_CANDIDATE`: 15 optimizer resumes, 15 fresh 4-frame
  observations younger than 100ms, 16 freeze verifications (final one does
  not require resume), no safety stops or abandoned episodes.
- `PASS_REWARD_REPLAY`: for **each actual PPO action**, independently replay
  `rs_v61_rewards.ProgressRewards` from the logged *observed duration* and
  terminal/truncation flags; survival/milestone/death breakdown and reward
  sent to PPO match within rounding tolerance. Count long, capped steps.

These are necessary checks, **not** statistical proof of learning improvement
or independent verification of Unity's internal clock. If either audit fails,
stop and share the output and original log. Do not launch overnight training.

The 2048-step trial saves outputs in a new `ai_data/rl_v2_runs/$run` directory;
the original checkpoint stays untouched.
