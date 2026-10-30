# Ranked Surfers PPO timing audit v1 (non-invasive)

## Purpose

The active `train_rs_v612_ppo.py` uses Stable-Baselines3 PPO with an actively
running Unity WebGL world. At a rollout boundary, SB3 trains on the last rollout
before choosing the next action. The previous environment observation may be old.
This package **measures that age**; it deliberately does not change observations,
actions, rewards, reset logic, guard rules, models, checkpoint formats, or policy.

### What gets logged

A new file inside each *new* PPO training run directory:
`ppo_rollout_gaps.jsonl`. Each line contains:

- `rollout_end_to_next_start_ms`: wall time from SB3 on_rollout_end to the next
  on_rollout_start (mostly PPO optimizer time).
- `last_env_step_to_next_start_ms`: time since the last completed env step.
- `cached_frame_to_next_start_ms`: time since the environment's last physical
  capture, if available.
- `last_env_step_done`: whether SB3 observed an episode ending on the last step.
- `potentially_stale`: diagnostic flag at >=250 ms, **not proof of a failed action**.

No extra browser screenshots are taken and no browser calls happen in the callback.

### Install safely on the Windows `claude-development` clone

Do not apply to your original repo. Extract the ZIP into the clone root,
keeping the `tools/` and `tests/` paths. Back up or commit your existing changes
first. Never `git add .` while untracked ai_data exists.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"

git status --short
& $py -B tools\apply_ppo_timing_audit.py --check
& $py -B tools\apply_ppo_timing_audit.py --apply
& $py -B -m unittest discover -s tests -p 'test_ppo_pipeline_audit.py' -v
& $py -B tools\ppo_reward_timing_probe.py
& $py -B train_rs_v612_ppo.py --help

git diff -- train_rs_v612_ppo.py
```

The installer makes `train_rs_v612_ppo.py.pre_ppo_timing_audit.bak` and refuses
if that backup already exists. `--check` previews the exact diff without writing.
No package installation is needed beyond the existing venv. The offline tests
only require Python stdlib + the repository's `rs_v61_rewards.py`.

### Run later, only after verifying the game and explicitly deciding to do so

Run a short, controlled Practice training continuation from a **copy** of the
4.61-second checkpoint in a **new** run directory. Use the established training
command and identical hyperparameters. Do **not** use the old `--run-name` or
overwrite any checkpoint. The instrumentation adds timing lines only.

Afterward, from the clone root, inspect:

```powershell
& $py -B tools\analyze_ppo_rollout_gaps.py `
  "ai_data\rl_v2_runs\NEW_UNIQUE_RUN_NAME\ppo_rollout_gaps.jsonl"
```

If the median age is many hundreds of milliseconds or seconds, it supports
investigating a safe method to refresh observations at each update boundary.
That *behavior-changing* fix should be implemented only after confirming the
logged age and testing the guard/terminal/reset behavior in local Practice.

### Reward diagnostic

`tools/ppo_reward_timing_probe.py` evaluates the **existing, unchanged**
`ProgressRewards` for equal 5-second episodes at different synthetic decision
intervals. It shows that the current 150 ms cap and per-decision discounting
can change return despite equal wall-time survival. This is evidence of a
possible learning incentive and **not** evidence that any specific action
actually has a longer execution time. Do not change reward coefficients yet.

### Rollback

Keep the `.bak` file as a safe point. To undo this instrumentation in a clean
uncommitted repo, first review `git diff -- train_rs_v612_ppo.py`. Restore the
specific file from its `.bak` copy if and only if it contains the state you
want. The patch adds no persistent model changes.
