# Ranked Surfers PPO v1.7.3 — post-resume death-menu handoff

This **opt-in** update applies to the already installed **v1.7.2** trainer
(v1.7.2 SHA256 `91fe4baa87e32d7b82f8edf4f753445d15a80fc7669f73a12ea6299791306e64`).
It does not alter policy weights, reward coefficients, lane-change behavior,
PPO hyperparameters, Unity pause timing, state classifier, original pretrained
checkpoint, or ordinary guarded death/restart logic.

## What the live failure showed

During the 4,096-step run, the seventh pause/resume succeeded, but resync
found `MENU` at capture 4/8 (~210 ms after resume), with
`ml_menu=0.998`, `menu_score=0.572`, and `ml_gameplay=0.000`.
The v1.7.2 safe stop was correct: PPO must never use that stale cached image
to send a new movement key. Unlike a pre-optimizer MENU, this *post-resume*
MENU did not yet have a guarded terminal handoff.

## New behavior

- When fresh resync first observes **both** ML MENU confidence >= existing
  confidence threshold and independent menu-template score >= 80% of the
  existing threshold, it records a candidate time. The first capture is **not**
  enough to authorize any action or death.
- The existing `confirm_post_optimizer_menu()` then requires **four consecutive
  independently corroborated MENU screenshots** within eight bounded captures.
  Focus is checked for every one. Anything unclear or inconsistent fails closed.
- If confirmed, the already-existing v1.7 guarded terminal handoff is armed.
  The optimizer has completed and no keyboard input, R, or direct reset occurs.
  The next normal PPO env step first verifies MENU using the existing
  `rs_v612_gameplay_guard.py`, records the **actual** `-1` death penalty, and
  lets the existing guarded restart proceed for the next episode.
- The terminal survival duration is anchored to the **first corroborated MENU
  observation**, not the later confirmation or idle-menu interval. The existing
  `compensate_pause_clocks` implementation excludes the separately verified
  idle menu time from episode, action, and Monitor clocks.
- No successful gameplay frame is counted as a new PPO observation after a
  MENU detection; no mixed/partial four-frame stack is published.
- The reward auditor distinguishes a confirmed post-resume death handoff from a
  normal four-frame refresh, and **fails** if the corresponding real terminal
  reward is missing. It continues checking ordinary pause/observation events.

**Known limitation:** the handoff records the terminal transition on the first
step of the next rollout. SB3 has already finalized the previous rollout, so
this is not exact last-action credit attribution. Offline tests do not prove
Unity clock behavior or overnight reliability.

## Install (Windows PowerShell, training stopped)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_post_resume_menu_v1732_verified.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_post_resume_menu_v173.py" -v
& $py -B -m tools.install_ppo_post_resume_menu_v173 --check
```

Only when both pass:

```powershell
& $py -B -m tools.install_ppo_post_resume_menu_v173 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

Backups are `train_rs_v612_ppo.py.pre_ppo_post_resume_menu_v173.bak` and
`tools/ppo_reward_timing_audit.py.pre_ppo_post_resume_menu_v173.bak`.
The installer **refuses unknown file hashes** and does not remove user changes.
It is not necessary to push source or checkpoints to GitHub.

## Controlled live verification — start with 2,048 steps

Start local `server.py` in another PowerShell window; keep Chromium visible,
focused, unobstructed, with the local Practice game. In the repo directory:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_post_resume_menu_v173_" + (Get-Date -Format "yyyyMMdd_HHmmss")
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

Keep `--recover-rollout-menu` OFF; it uses the older abandoned-episode
recovery and could omit the real death reward.

## Audit after training

```powershell
& $py -B -m tools.ppo_reward_timing_audit --log $log --min-resumes 15 --min-steps 2048
$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit --reward-steps $rewardLog --min-steps 2048
Select-String -Path $log -Pattern "\[PPO POST RESUME MENU\]|\[PPO MENU HANDOFF\]|\[PPO RESYNC TRANSIENT\]|Safe stop|Traceback"
```

**Success:** `PASS_STABILITY_CANDIDATE`, `PASS_REWARD_REPLAY`, and 2,048
completed steps. A successful post-resume MENU handoff produces:

```text
[PPO POST RESUME MENU] Confirmed MENU after 4 fresh ML+template captures; menu_idle_seconds_excluded=...; deferred guarded terminal with NO movement.
[V612 GUARD] MENU confirmed from ... captures; NO key sent ...
[PPO MENU HANDOFF] Terminal recorded by guarded env.step; death_reward=-1.0000; NO movement key sent.
```

If MENU never occurs, this verifies ordinary stability but not the new live
handoff. Keep overnight training on hold until longer runs with both audits
pass; a successful 2,048-step run does not establish overnight reliability.


## v1.7.3.2 — verified deterministic test harness

The first v1.7.3 test file imported the installed v1.7.2 auditor rather than
checking the proposed v1.7.3 auditor. The next test revision corrected that
but attempted to simulate elapsed time by replacing a captured, immutable
candidate timestamp tuple. The real callback had already stored the previous
tuple, so the Windows test still saw a zero elapsed interval.

In v1.7.3.2 the callback test simulates advancing a consistent monotonic,
perf-counter and wall clock by 210 ms after MENU confirmation. This is
independent of Windows clock granularity and involves **no real sleeping**.

In a clean copy of the exact v1.7.2 code, all **19 pre-install tests passed**;
`--check` and `--apply` passed; post-install tests passed with one expected
skip of the pre-install-only installer test. The **production trainer, auditor,
and installer payloads are unchanged** from the original v1.7.3 proposal.
The patch has not yet been confirmed in live Unity play.

**Do not use older v1.7.3 ZIPs.** Extract this v1.7.3.2 ZIP with `-Force`,
rerun tests and `--check`, then apply only if both pass.
