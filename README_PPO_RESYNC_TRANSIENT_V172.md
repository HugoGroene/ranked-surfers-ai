# PPO v1.7.2 — bounded post-resume `UNCERTAIN` frame recovery

## Incident

A 2,048-step training smoke test on v1.7.1 stopped after 1,920 steps and 15 verified freezes/resumes. The first screenshot 16.7 ms after optimizer resume classified `UNCERTAIN`, with ML GAMEPLAY 0.804, ML MENU 0.003, and independent MENU template score 0.000:

```
[PPO RESYNC TRACE] frame=1/4 since_resume_ms=16.7 state=UNCERTAIN menu_score=0.000 ml_gameplay=0.804 ml_menu=0.003
[PPO RESYNC] Safe stop: Cannot verify gameplay after PPO update ...
```

This is a legitimate safety stop; one near-GAMEPLAY uncertain frame does not prove gameplay or death. An immediate hard stop is unnecessarily strict if fresh follow-up captures can establish positive gameplay evidence. The screenshot itself was not provided, so the exact visual reason for the `UNCERTAIN` state is still unknown.

## What this patch changes

**Exactly one method**: `PPOV61Env.refresh_after_ppo_optimizer()` in `train_rs_v612_ppo.py`. The installer verifies the full SHA-256 of the installed v1.7.1 source. This is an OPT-IN refinement: its path only executes with existing `--resync-rollouts`, and your `--pause-ppo-optimizer` flag still defaults OFF.

* After resume, permits up to **8 no-input screenshots** to establish **4 *consecutive*, independently ML-confirmed GAMEPLAY** images.
* Only a narrowly defined `UNCERTAIN` frame may be skipped: `GAMEPLAY` probability >= 0.70 but below the *unchanged* configured gameplay confidence, MENU probability <0.10, and menu template score <0.30 × configured menu threshold. Up to 4 such skipped frames are allowed in the 8-capture window. A skipped frame **does not count as gameplay**.
* A skipped frame invalidates any partially accumulated PPO image stack. The next action is not permitted until four *new* valid consecutive images are captured.
* Any `MENU`, `LOADING`, `OTHER`, high menu-template score, low GAMEPLAY probability, browser focus loss, or persistent ambiguity **fails closed**, with no movement, no R, and no reset. When available, it saves the exact offending screenshot to `ppo_resume_failure_epNNNN_frameN.jpg`.
* Logs `[PPO RESYNC TRANSIENT]` for each discarded uncertain frame and exactly four `[PPO RESYNC TRACE] frame=N/4` lines **only for the final validated stack**. Existing v1.7 reward/stability audit thus remains compatible, including if an uncertain frame interrupts a partial stack.
* Keeps the original four-frame `self._pack()` path, native frame order, YOLO scheduling, 100 ms post-capture staleness check, and existing per-action gameplay guard. An increased no-action observation window may increase the existing capped interval after an optimizer; it is intentionally bounded and must be evaluated in the reward logs.

**Unchanged:** PPO policy/actions, hyperparameters, rewards, pause accounting, menu-handoff terminal detection, classifier/threshold, replay auditor, checkpoint files and guarded restarts. No Unity runtime was available for the authoring environment: **live PPO confirmation is still required**.

## Installation (Claude clone, PowerShell)

With training stopped, in the same repo that passed v1.7.1 tests:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_resync_transient_v172.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_resync_transient_v172.py" -v
& $py -B -m tools.patch_ppo_resync_transient_v172 --check
```

If and only if all pass:

```powershell
& $py -B -m tools.patch_ppo_resync_transient_v172 --apply
& $py -B -m unittest discover -s tests -p "test_ppo_resync_transient_v172.py" -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

Backup: `train_rs_v612_ppo.py.pre_ppo_resync_transient_v172.bak`. The trainer will *not* be replaced if its hash differs; do not bypass this check or restore the tracked Git version.

## Repeat the 2,048-step controlled test

Make sure `& $py server.py` is running in another terminal, port 8000. Keep Chromium visible and unobstructed. The exact original checkpoint is used **only as a read-only input**; unique experiment directories receive any trained outputs.

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_resync_v172_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = "ai_data\ppo_pause_$run.log"
& $py -B -u train_rs_v612_ppo.py `
  --mode train `
  --pause-ppo-optimizer `
  --resync-rollouts `
  --audit-reward-timing `
  --resume $checkpoint `
  --run-name $run `
  --steps 2048 `
  --max-minutes 10 `
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

Do **not** add `--recover-rollout-menu`; the existing v1.7 boundary-menu handoff and classifier guard must remain authoritative.

After training terminates:

```powershell
& $py -B -m tools.ppo_reward_timing_audit --log $log --min-resumes 15 --min-steps 2048
$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit --reward-steps $rewardLog --min-steps 2048
Select-String -Path $log -Pattern "\[PPO RESYNC TRANSIENT\]|\[PPO RESYNC TRACE\]|\[PPO RESYNC\]|\[PPO PAUSE\]|\[PPO MENU HANDOFF\]|Safe stop|Traceback"
```

Targets: `PASS_STABILITY_CANDIDATE` and `PASS_REWARD_REPLAY`, 2048/2048 steps, 15 continuations, 15 final fresh four-frame stacks (and no safety stop). There may be one final freeze with no matching resume because SB3 saves after the last rollout.

If a retry occurs, expect `[PPO RESYNC TRANSIENT] skipped=1 ...` then four normal `[PPO RESYNC TRACE] frame=1/4 ... frame=4/4` followed by a fresh observation age under 100 ms. If a retry *doesn't* occur, the successful run verifies only the ordinary path, not the newly added transient path live.

If it stops again with `UNCERTAIN`, `MENU` or focus loss, do **not** resume the partial checkpoint. Supply the exact error and diagnostic screenshot for the affected episode. The prior run's `ppo_resume_failure_ep0027_frame1.jpg` would also help establish the visual cause.

## Safety limits

The patch does not lower ML confidence or infer gameplay from motion alone. It does not fix independent PPO reward-per-step discounting, the 0.15-second survival cap or policy lane-change imbalance. No overnight training until the longer smoke test and reward replay pass, and enough additional stability has been established.
