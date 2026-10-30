# Ranked Surfers v1.7.9 — explicit runtime PAUSED exclusion

**Scope:** A narrow source-matched safety fix for the first four images after PPO optimizer resume. No PPO weights, policy actions, survival rewards, ML state-classifier labels, detection thresholds, death penalty, or checkpoint code are changed.

## Evidence

A prior overnight failure occurred at `capture=1/8`, `UNCERTAIN` with `ml_gameplay=0.547`, 19.7 ms after resume. The exact screenshot shows the React **Paused** panel still rendered even though the trainer had reported the overlay hidden. The current nine-class learned screen classifier does not include a dedicated `PAUSED` label.

## What changes

- After the site Pause panel and three frozen track differences are independently verified, cache a small visual fingerprint of the panel.
- During each post-resume screenshot capture, check the Playwright Pause panel **before and after** the capture and compare the central screenshot crop with that verified paused fingerprint.
- Either indicator marks the frame `runtime_state=PAUSED`, excluding it from PPO's image stack **even when ML predicts `GAMEPLAY`**.
- Clear any partial fresh stack and wait passively, bounded by the existing maximum eight captures and at most four PAUSED frames. Never click a button or send a movement action during the check.
- Continue only after **four consecutive fresh high-confidence `GAMEPLAY` images**, with no DOM or visually stale Pause panel. Still honor strict independent MENU confirmation and stop on unresolved states.
- No tenth class is added to the ML model; this is only an independent runtime UI safety state.

**Limit:** The visual fingerprint is layout-dependent and detects close matches to the prior verified Pause panel. It is a conservative extra exclusion signal, not a universal visual Pause classifier. If the site redesigns the modal, revalidate it. The visual reference is cached within the running process, never written to disk.

## Installation (Windows PowerShell)

Stop training. Extract this archive into your repository's working directory, then run:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -m unittest discover -s tests -p "test_ppo_pause_state_v179.py" -v
& $py -B -m tools.patch_ppo_pause_state_v179 --check
& $py -B -m tools.patch_ppo_pause_state_v179 --apply
& $py -B -m unittest discover -s tests -p "test_ppo_pause_state_v179.py" -v
& $py -B -m unittest discover -s tests -p "test_ppo_current_runtime.py" -q
& $py -B -m tools.ppo_reward_timing_audit --self-test
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
```

The installer checks the normalized SHA-256 of the *exact v1.7.8 trainer* and packaged helper, and refuses unexpected changes or a second application. It preserves Windows CRLF line endings, creates `train_rs_v612_ppo.py.pre_ppo_pause_state_v179.bak` and replaces the trainer atomically. The new helper is supplied directly by the archive.

## Live gate

Resume from the successful v1.7.8 **test** checkpoint in a **new** run directory, test 2048 steps with pause/resync/reward auditing enabled, then run both reward audits. If you see `[PPO RESYNC PAUSED]`, the first frames were excluded as intended. Lack of such a line means the exception path was not exercised in that run. Keep strict UI stops; do not start overnight until the run finishes safely.
