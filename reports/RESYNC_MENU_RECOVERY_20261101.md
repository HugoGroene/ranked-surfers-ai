# Post-resume uncertain-menu recovery

Run `v612_detectoroff_3workers_24h_20261101_095630` stopped after 3660.32 seconds,
324 shared updates and 124,416 optimized transitions. Finalization errors were
empty; final checkpoint transitions were 124,416. Source checkpoint was unchanged.

Worker 0 episode 445 frame 3 showed the actual menu (logo and FIND RANKED MATCH).
Recorded probabilities: GAMEPLAY .088, MENU .889; template .656. At confidence
.90, the classifier returned UNCERTAIN. Resync only nominated confident MENU for
independent confirmation, so this supported candidate instead stopped the cohort.
This is distinct from the previously fixed Electron wall-clock timestamp failure.

`ResyncWindow` now also nominates UNCERTAIN with MENU probability at least
max(.80, .90 * configured confidence), GAMEPLAY below .10, and template at least
.80 * configured template threshold. Nomination STILL aborts gameplay refresh;
it authorizes only the existing bounded movement-free confirmation path. Four
fresh consecutive confident MENU frames with independent template support remain
required before the existing terminal handoff. No terminal acceptance threshold,
reward, PPO setting, image history or input timing was changed.

47 focused offline tests passed, including exact failure replay through the live
refresh method, no packing/keys, rejection of weak/ML-only candidates, unchanged
four-frame confirmation, pause exclusion and runtime regressions. No live test
has yet exercised this exact new path. Inconclusive confirmation will still stop.

Run `tools/restart_ppo_24h_20261101.ps1` in a separate PowerShell terminal. It starts
from this run's final checkpoint, performs a three-minute three-worker smoke run,
then starts 24 hours only if completion and all three learning audits pass. The
long stage continues from the smoke checkpoint, saves every ten minutes, has no
dashboard and opens its plot at completion. A smoke pass does not guarantee the
rare candidate occurs. The separate temporal-history prototype is not enabled.
