# Ranked Surfers PPO pause v1.5.2 — pause-overlay classifier fix

## Observed issue

Your 512-step v1.5.1 smoke run stopped at the first 128-step rollout boundary with:

`[PPO PAUSE] Safe stop at rollout-end: Not verified GAMEPLAY during pause: UNCERTAIN, template=0.000`

The local Practice site draws a fullscreen pause overlay; **the unoccluded GAMEPLAY-trained screen-state classifier is not valid for this overlay**. The pre-Pause GAMEPLAY check still passed, but the code then incorrectly demanded a second GAMEPLAY classification *while the overlay was on screen*.

## Scoped fix

Replace only this inappropriate second check. **Before Pause**, the classifier must still confirm GAMEPLAY and the menu-template score must be low. **During Pause**, the site overlay must remain present, the central track must show three consecutive frozen frame intervals within 16 captures (v1.5.1 threshold 0.25 unchanged), and independent strong MENU template evidence still aborts. **After Resume**, the existing resync captures and verifies four unobstructed GAMEPLAY frames and the PPO input guard continues unchanged. No assumption of success based on overlay alone.

Original source and checkpoint untouched. Only `train_rs_v612_ppo.py` is modified *inside the Claude development clone* by the installer. Backup is created. Both flags remain opt-in. No edits to game code, rewards, PPO weights, classifier files, or restart logic.

## Install, check

Run in the Claude clone with training stopped and Python environment set:

```powershell
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_overlay_v152.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p 'test_ppo_optimizer_pause_overlay_v152.py' -v
& $py -B -m tools.patch_ppo_optimizer_pause_overlay_v152 --check
& $py -B -m tools.patch_ppo_optimizer_pause_overlay_v152 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

Only if all checks pass, with local Practice server running, test 512 steps using the **same original 4.61s checkpoint and settings**, a new run name, and `--pause-ppo-optimizer --resync-rollouts` but **without** `--recover-rollout-menu`.

## Validation

Expect 3 boundary cycles with `[PPO PAUSE] Freeze verified ...`, `[PPO PAUSE] Resumed after optimizer with pause-aware clocks ...`, and `[PPO RESYNC] refreshed ...` with age <100 ms, no safe stop, all 512 steps, and no abandoned partial episodes. If any uncertainty occurs, stop, inspect logs; do not proceed to overnight training. This still does not establish that the independent reward-step cap and PPO discounting bias is resolved.
