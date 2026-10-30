# Local Practice pause/resume probe v1.1: overlay button fix

The v1 pause probe demonstrated an unpaused scene difference of 42.4533 and a paused difference of zero, but could not unpause: Playwright reported that `.game-pause-layer` intercepted pointer events on the page-level `Resume game` button.

This package changes exactly one source location in `tools/probe_unity_pause.py`: it chooses `overlay.get_by_role('button', name='Resume', exact=True)`, the visible **Resume** action inside the pause overlay. It is used for both the normal resume and finally/cleanup. The patch does not touch the PPO trainer, runtime guard, environment, rewards, checkpoint files or main game.

With training stopped, the **local Practice server** running and your current working directory at the Claude clone repo root:

```powershell
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_resume_fix_v11.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m tools.apply_pause_probe_resume_fix --check
& $py -B -m tools.apply_pause_probe_resume_fix --apply
& $py -B -m unittest discover -s tests -p 'test_ppo_pause_resume_fix_v11.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_pause_probe.py' -v
& $py -B -u -m tools.probe_unity_pause --state-model $state --detector $detector
```

An unmodified v1 probe is required. The installer creates `tools/probe_unity_pause.py.pre_resume_fix_v11.bak`. If an anchor or backup is missing/unexpected, it refuses to write. Run it using `-m` to ensure Python resolves `tools` imports.

Expected healthy result: **Unpaused scene change > 1**, **Paused scene change nearly 0**, **Resumed scene change > 1**; report status `PASS_CANDIDATE`. The saved `pause_probe_report.json` contains `pause_overlay_confirmed`, `resume_overlay_hidden`, `freeze_supported_by_pixels` and `resume_motion_seen`.

Visual freezing alone is not proof that Unity's simulation clock or reward clock stops. Before integrating pause around PPO updates, confirm the game timer/distance HUD does not progress during pauses and that the `GAMEPLAY` classifier is not run while the overlay is visible. Do not start overnight training from this result.
