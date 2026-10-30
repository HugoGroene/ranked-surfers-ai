# PPO post-resume uncertain-first-frame grace — v1.7.8

## Why

An 8-hour v1.7.7 training attempt stopped after 19,072 new PPO steps (about 25 minutes). At PPO rollout boundary 149, the very first screenshot after resume was:

`state=UNCERTAIN menu_score=0.000 ml_gameplay=0.547 ml_menu=0.004 capture=1/8`

The existing `tools/ppo_resync_transient_v172.py` permits passive retry only for UNCERTAIN frames with `p_game >= 0.70`, so this single ambiguous frame stopped training. We have NOT received the actual `ppo_resume_failure_ep0278_frame1.jpg` screenshot and cannot assert whether it shows tunnel camera motion, a death transition, or something else.

## Fix

- **Only** `tools/ppo_resync_transient_v172.py` is changed. The trainer, reward logic, PPO, model, original checkpoint, optimizer clock, pause click/reclick and death-hand-off path are unchanged.
- Low-confidence `UNCERTAIN` with weak ML MENU (<0.10) and weak template (<30% of menu threshold) now receives the *existing passive retry*, bounded by 4 uncertain skips in 8 captures. No confidence threshold for GAMEPLAY is lowered.
- Every uncertain frame invalidates any partial fresh RGB stack; the trainer still requires **four consecutive independently captured, high-confidence GAMEPLAY frames** before any action.
- Explicit MENU, strongly menu-suspicious, OTHER/LOADING, focus loss, invalid probabilities and exhausted retries still abort safely. Confirmed MENU is still passed to the existing corroborated terminal-death handoff.
- This change does not invent a death/restart or send movement on unknown screens.

## Install (training stopped, Windows PowerShell)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_resync_uncertain_v178.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_resync_uncertain_v178.py" -v
if ($LASTEXITCODE -ne 0) { throw "Candidate tests failed; do not install" }
& $py -B -m tools.patch_ppo_resync_uncertain_v178 --check
if ($LASTEXITCODE -ne 0) { throw "Source mismatch; do not install" }
& $py -B -m tools.patch_ppo_resync_uncertain_v178 --apply
if ($LASTEXITCODE -ne 0) { throw "Installer failed" }
& $py -B -m unittest discover -s tests -p "test_ppo_resync_live_v178.py" -v
if ($LASTEXITCODE -ne 0) { throw "Actual runtime regression failed" }
& $py -B -m unittest discover -s tests -p "test_ppo_current_runtime.py" -q
if ($LASTEXITCODE -ne 0) { throw "Existing runtime tests failed" }
& $py -B -m tools.ppo_reward_timing_audit --self-test
if ($LASTEXITCODE -ne 0) { throw "Reward self-test failed" }
git diff --check
```

The installer accepts exact original helper content with LF or CRLF, makes a `.pre_ppo_resync_uncertain_v178.bak` backup and refuses an unexpected helper or repeat installation. Candidate tests before install use the candidate code, not the old helper. The clean-clone test file `tests/test_ppo_resync_live_v178.py` exercises the actual currently installed trainer/helper after install and does not depend on the installer or its payload.

## Test status

On a local source-matched simulation: 17 new candidate/installer tests passed (before install); 17 passed after install; 6 clean-clone live-helper tests passed after install; 18 pre-existing current-runtime regression tests passed; synthetic reward clocks passed. **Not yet verified on Windows live Unity.**

## Evidence to inspect

Please also upload (or visually inspect) the screenshot saved in the stopped run at:

`ai_data/rl_v2_runs/v612_overnight_480min_v177_20261028_040130/ppo_resume_failure_ep0278_frame1.jpg`

The screenshot could show a camera pan/tunnel, death animation or a different overlay. This patch addresses premature abort on a weak MENU/unknown frame, not the fundamental classifier's ability to classify the image.

## Training and next steps

Do not immediately start an unattended 8-hour run. Test 2,048 or 4,096 live steps from the stopped run's *new output* `latest.zip`, check both audits, and keep the original 7-hour source checkpoint untouched. The new branch is bounded, so it may not occur during a short test. Commit only the modified runtime helper and the clean-clone test after live validation; no backup or ZIP payload belongs in Git.
