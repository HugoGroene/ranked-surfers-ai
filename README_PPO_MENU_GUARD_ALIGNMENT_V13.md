# PPO optimizer-menu guard alignment v1.3 (experimental)

## Why this patch exists

The first resynchronization of the 4-frame observation **worked in the real game**: a cached ~1,040ms-old image was refreshed to an 8.2ms-old image before the next PPO decision.

At the second PPO update the screen classifier repeatedly returned `MENU`, with the same template score `0.5771913909581744` each time. Experimental recovery v1.2 requires a threshold of `0.578` (0.85 × 0.68), so it stopped safely by a margin of ~0.00081.

The already-tested `rs_v612_gameplay_guard.screen_gate` and `.sequence_decision` instead require **four consecutive confident ML MENU classifications corroborated by independent template evidence**. The *moderate* corroboration threshold is 0.80 × the main template threshold (here 0.544), and the sequence must be terminal. This aligns the new recovery with established game safety checks; it does **not** simply accept ML MENU by itself or disable the safety guard.

## Installation in the Claude clone

Leave Practice training and the local game stopped during installation. Extract the ZIP into the clone root where v1.2 is already installed.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_menu_guard_alignment_v13.zip" -DestinationPath (Get-Location).Path -Force
& $py -B tools\align_ppo_menu_recovery_guard.py --check
& $py -B tools\align_ppo_menu_recovery_guard.py --apply
& $py -B -m unittest discover -s tests -p 'test_ppo_menu_guard_alignment.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_optimizer_menu_recovery.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_resync_hardening.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_rollout_resync.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_pipeline_audit.py' -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

The `--apply` makes `train_rs_v612_ppo.py.pre_menu_guard_alignment.bak` before changing anything. `--resync-rollouts` and `--recover-rollout-menu` remain opt-in. Nothing touches the original 4.61s PPO checkpoint.

## Live verification

Repeat the **same short local Practice smoke test** used for v1.2; do not start overnight. Use a fresh `--run-name` and `--log` and supply both `--resync-rollouts --recover-rollout-menu`. Preserve fps 30 uncapped, 5ms keypress, YOLO every 3 steps, image 512x288 and the original checkpoint. After run:

```powershell
Select-String -Path $log -Pattern '\[PPO GAP\]|\[PPO RESYNC\]|Safe stop|Traceback|ABANDONED'
```

Success is repeated fresh image refreshes (<100ms), or a confirmed menu-safe reset followed by continued training. Any ambiguity still produces a safe stop.

## Important: not a complete training-system correction

**If the game dies while PPO trains, its missing death transition is NOT included in the already-completed rollout.** A recovered abandoned episode omits its terminal reward. Count `[PPO RESYNC] Verified MENU... ABANDONED` events; repeated instances make long-run PPO training questionable. The patch fixes fresh next-decision input and conservative restart after verified MENU; it does not eliminate live-game progression during optimizer pauses or correct the previous rollout's missing death signal. Prefer investigating optimizer pause avoidance or gameplay clock freezing before long training.

No demo recording is involved. Don't commit `ai_data/` or any `.bak` files.
