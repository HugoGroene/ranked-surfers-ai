# Ranked Surfers AI â€” current V6.1.2 PPO

Local Practice-only reinforcement-learning experiment. Runs against the locally served
Unity WebGL game (`http://localhost:8000`), **not** the public/ranked game.

## Current setup

- **Trainer/evaluator:** `train_rs_v612_ppo.py`
- **UI safety gate:** `rs_v612_gameplay_guard.py` and `test_rs_v612_guard.py`
- **State classifier:** `rs_v4_screen_states_v2.py`; active local model:
  `ai_data/v4_screen_states_rebuild_01/screen_state_candidate_v2.pt`
- **Hazard detector:** `ai_data/v5_hazards/runs/segment_round04/weights/best.pt`
- **Inputs:** 4 RGB frames + nine-class YOLO features; PPO chooses NONE/LEFT/RIGHT/JUMP/ROLL
- **Current keypress configuration:** 5 ms; 512x288; `--rate-mode uncapped`
- **Trained policy (Git LFS backup):** `checkpoints/v612_after_120min_press5_20261026.zip`
- **Local train/eval reports:** `ai_data/rl_v2_runs/` (intentionally Git-ignored)
- **Archived evaluation reports:** `reports/v612_learning_proof_20261026_153745/`

The latest saved 100-episode evaluation at 5 ms reports mean survival 3.11 s
before the 120-minute training run and 3.84 s afterwards (do-nothing reference:
3.48 s). These are results, not a guarantee of performance on every randomized map.

## Start the local Practice server

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory .\rankedsurfers-downloaded
```

Keep the game visible and unobstructed for MSS screen capture; disable display
and PC sleep while training. The saved safety guard must remain enabled.

## Checks after modifying code

```powershell
.\.venv\Scripts\python.exe -m unittest test_rs_v612_guard test_rs_v61 -q
.\.venv\Scripts\python.exe train_rs_v612_ppo.py --mode check --resolution 160x96
```

## Current and historical files

Files still at the root are active entrypoints or Python dependencies imported by
them. Older versions and standalone benchmarks are under `development/archive/`.
Local diagnostic ZIPs and backups are in `development/local/` (not committed).
Previous historical source is also preserved in Git history.

Moving legacy dependencies merely because they have names beginning `v2`, `v3`,
or `v4` will break imports in the current trainer. Use Python import graph checks
before further moves.

For the updated live guard and dataset tools, see
`README_RS_V612_GUARD.md`, `README_RS_V4_SCREEN_STATES.md`, and
`README_RS_V5_FIXED_SPLIT.md`.

Checkpoints are backed up with Git LFS. `git push` does not automatically back up
new files under ignored `ai_data/rl_v2_runs/`: copy selected weights and JSON
reports to `checkpoints/` and `reports/` explicitly when preserving new runs.

The downloaded Unity assets are retained in their existing path so the local
server works. Before sharing the repository publicly, confirm you have the
necessary permission to distribute those third-party assets.
