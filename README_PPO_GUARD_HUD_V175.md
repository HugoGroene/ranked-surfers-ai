# Ranked Surfers PPO V1.7.5 — browser HUD position fix

## Evidence from the uploaded V1.7.4 run

The fail-closed screenshot `v612_guard_episode_0004.jpg` shows live gameplay in a tunnel, a visibly jumping player, the 179 m HUD, and the Pause icon. The six guard readings were ML `UNCERTAIN` with P(GAMEPLAY)=.778–.894, P(MENU) ≤.0031, zero template scores, genuine track motion around 10–11, track texture around 31–33. The HUD pixel count was **zero**.

The original inherited V5 `_meter_mask` samples a **fullscreen-only** ROI: x=10.1–15.1%, y=1.1–4.8% of screen. In the uploaded screenshot the Chromium tab/address bars are visible; the yellow distance digits are at approximately y=111–121/1080, not y=12–52. Checking a 96×44 px crop from the uploaded screenshot confirms the original ROI has zero matching yellow pixels and the Chrome-visible ROI recovers the HUD.

## Exact change

An opt-in `PPOV61Env._meter_mask` override uses original fullscreen sampling when valid and falls back to x=10.1–15.1%, y=9–13% for the *observed* browser-chrome layout. It returns the exact same 64×24 binary mask shape. It does **not** change screen-gate thresholds, menu evidence, motion threshold, action mapping, reward formula, capture mechanics, or PPO pause/resync/eval code. This patch does not change the inherited V5 code.

**Crucially:** recognizing yellow digits by itself never authorizes a keypress. The unchanged guard still requires a *change* in the distance mask, independent moving textured track, lack of strong menu-template evidence, and two consecutive independent positive gameplay observations. If no supported HUD is found, the mask remains zero and the guard stops safely.

The supplied 96×44 test fixture is only the cropped distance counter from the user-provided screenshot and contains no address bar or private system details. Offline tests check the actual recovered HUD pixels, unchanged fullscreen layout, stagnant HUD fail-closed behavior, moving HUD evidence, menu preemption, source-hashed installation, and rejection of unknown files. A **single** screenshot cannot prove the digit delta over successive live captures, so the recovery path requires live verification.

## Installation (PowerShell, training stopped)

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_guard_hud_v175.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p 'test_ppo_guard_hud_v175.py' -v
if ($LASTEXITCODE -ne 0) { throw 'v1.7.5 offline tests failed; do not install' }
& $py -B -m tools.patch_ppo_guard_hud_v175 --check
if ($LASTEXITCODE -ne 0) { throw 'v1.7.5 source mismatch; do not install' }
& $py -B -m tools.patch_ppo_guard_hud_v175 --apply
if ($LASTEXITCODE -ne 0) { throw 'v1.7.5 apply failed' }
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

## Next validation

Use the existing v1.7.4 **2,048 step** controlled training command with a fresh `--run-name`, the **original** pretrained `--resume` checkpoint, `--pause-ppo-optimizer --resync-rollouts --audit-reward-timing`, and the same FPS/YOLO settings. Afterward run both audits requiring `--min-steps 2048` and 15 optimizer continuations. A completed run establishes basic integration; a line `[V612 GUARD] Resumed after ... two positive gameplay confirmations` is needed to confirm this exact formerly-uncertain HUD path worked live.

Do not start an overnight run based solely on offline tests. If the screenshot geometry changes again, the guard may still correctly stop. Original checkpoint remains untouched.
