# Ranked Surfers LOCAL Practice — pause/HUD/clock diagnostic v1.2

**Observation-only, NO PPO training or weight changes.**

The previous pause v1.1 probe measured moving scene ~45 pixels, paused scene 0, and resumed scene ~32. It did not establish what would happen to the environment's time-based rewards.

This independent new `tools.probe_unity_pause_clock` module reuses the project's real local Practice browser/crop and existing V6.1 screen-state checks. It clicks the site's Pause button, verifies the overlay, samples the *same existing `_meter_mask` distance HUD ROI* and central track while running, paused and resumed, resumes by clicking the **visible Resume button inside the overlay**, and checks the overlay disappears. It also measures how far the environment's *unmodified* `time.monotonic() - episode_start` clock advances while paused.

## Install and run (Windows PowerShell, Claude clone only)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_clock_probe_v12.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_pause_clock_probe.py" -v
& $py -B -u -m tools.probe_unity_pause_clock --state-model $state --detector $detector
```

Your local Practice server must already be running at `http://localhost:8000`. Keep Chromium visible and unobstructed. Do not press keys while the probe runs. The probe intentionally starts one local Practice episode using the *existing* classifier; it does not create keys/action policy training.

## How to interpret

- `Running scene change > 1`, `Paused scene change ~ 0`, `Resumed scene change > 1` reproduces the reliable visual pause cycle.
- `Running HUD change` / `Paused HUD change` / `Resumed HUD change` show `yellow_pixels_*` and `changed_pixels`. `PASS_HUD_CANDIDATE` means a detectable yellow distance HUD changed while running and after resuming, but **not while paused**. This is useful indirect evidence, not proof of internal Unity clock semantics.
- `PASS_SCENE_HUD_INCONCLUSIVE` means the prior scene-freeze result was reproduced but the distance HUD could not be validated; do not interpret as HUD progress confirmed.
- `REJECT_HUD_MOVED_DURING_PAUSE` or `INCONCLUSIVE` means stop and inspect screenshots.
- `Trainer raw episode duration advanced while paused` prints elapsed seconds. This **deliberately does not adjust** `raw.episode_start`. V6.1 reward calculation and watchdog currently depend on wall-clock `info['duration']`, and the 557-feature observation includes wall-clock `episode_started`, so pause accounting MUST be corrected before pause integration into PPO.

The report and six audit screenshots are written into a separate `ai_data/rl_v2_runs/v612_pause_clock_probe_.../` folder. `pause_clock_probe_report.json` is the summary.

**No changes are made** to `train_rs_v612_ppo.py`, the restart/state classifier, game files, PPO policy, rewards, or original checkpoint.
