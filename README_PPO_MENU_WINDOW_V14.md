# PPO optimizer-pause MENU confirmation: rolling window v1.4

## Why

The v1.3 live smoke test showed **two successful observation refreshes** (1,034 ms -> 8.1 ms and 858 ms -> 8.2 ms). The third boundary stopped: 4 consecutive classifier predictions were MENU (p >= .9992) but the *first* template score was .47064, below corroboration level .544; following scores .72788, .68168, .58733 passed. v1.3 samples exactly four frames, so it cannot see a fifth corroboration after a transient first frame.

v1.4 samples **up to eight** fresh captures. It only accepts when the **latest four consecutive readings** are independently corroborated by the existing `rs_v612_gameplay_guard` rules. A weak template reading or gameplay prediction restarts the four-frame streak. Browser focus is checked anew on every sample, and ambiguous scenes **still cause a safe stop**. No threshold is lowered; no keyboard commands are sent in the confirmer.

## Installation (Claude development clone only)

**Stop trainer and game before applying.** Extract the ZIP into the clone root after v1.3 is installed. Do NOT install in the original working/trained repo.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_menu_window_v14.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m tools.extend_ppo_menu_recovery_window --check
& $py -B -m tools.extend_ppo_menu_recovery_window --apply
& $py -B -m unittest discover -s tests -p 'test_ppo_menu_guard_alignment_v14.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_menu_guard_alignment.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_optimizer_menu_recovery.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_rollout_resync.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_resync_hardening.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_pipeline_audit.py' -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

Invoke installer as `-m tools.extend_ppo_menu_recovery_window` to avoid Windows import path trouble. It checks for the **exact v1.3 method**, refuses mismatches, and saves `train_rs_v612_ppo.py.pre_menu_window_v14.bak` when applying. Dry-run makes no changes.

## Live smoke test

Repeat the prior **512-step, max 5-minute** run from the original checkpoint with both `--resync-rollouts --recover-rollout-menu`, using a new run name. Keep the local Practice server running and check `[PPO RESYNC]` log lines. We want repeated fresh observations and at least one **verified MENU restart**, or a safe stop if no four-frame agreement occurs.

**Do not begin an overnight PPO run.** There is still a fundamental limitation: Unity continues during every ~0.8–1.0 second optimizer pause; a death during that gap is never included as a terminal transition in the rollout PPO has already learned from. The optional recovery abandons a partial episode without a death reward. Fixing that requires changes beyond menu detection and needs a controlled design/evaluation. Neither improved survival nor complete training correctness has been established by these patches.

Do not stage `ai_data/`, backups, model ZIPs, or log files in git.
