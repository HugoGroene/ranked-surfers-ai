# PPO auditor correction v1.7.5.1 (auditor only)

Your v1.7.5 run completed all 2,048 steps with 15 optimizer continuations, two independently confirmed MENU handoff terminals, fresh image stacks and a passing reward replay. Its stability audit failed solely because it required `pre-pause GAMEPLAY checks == verified freezes`. This is too strict when a button/menu transition happens after an initial valid GAMEPLAY screenshot and is independently confirmed as a terminal MENU by the existing v1.7.3 handoff.

This patch permits at most **one extra pre-pause GAMEPLAY check per confirmed rollout-boundary MENU**. It retains every other stability validation, including required optimizer continuations, independent MENU confirmation and real terminal reward, frame freshness, stop detection, and requested-step completion. It does not change training or gameplay code, checkpoint, classifier, rewards, or Unity clock adjustment.

Install from the repository root, with training stopped:

```powershell
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_reward_precheck_v1751.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_reward_precheck_v1751.py" -v
if ($LASTEXITCODE -ne 0) { throw "Audit tests failed; do not install" }
& $py -B -m tools.patch_ppo_reward_precheck_v1751 --check
if ($LASTEXITCODE -ne 0) { throw "Source check failed; do not install" }
& $py -B -m tools.patch_ppo_reward_precheck_v1751 --apply
if ($LASTEXITCODE -ne 0) { throw "Installation failed" }
```

Then audit the **existing** v1.7.5 log, with no new training run needed:

```powershell
$run = "v612_guard_hud_v175_20261028_023117"
$log = "ai_data\ppo_pause_$run.log"
& $py -B -m tools.ppo_reward_timing_audit --log $log --min-resumes 15 --min-steps 2048
$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit --reward-steps $rewardLog --min-steps 2048
```

Expected: `PASS_STABILITY_CANDIDATE`, `PASS_REWARD_REPLAY` if the full log matches the pasted audit. The new logic is an audit fix, **not** proof the tunnel-camera pan is solved. The earlier run had `visual_gameplay_recoveries: 0`.
