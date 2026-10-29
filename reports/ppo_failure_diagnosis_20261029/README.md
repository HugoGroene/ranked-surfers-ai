# Existing-artifact diagnosis: 29 October 2026

Read-only inspection of the completed corrected PPO learning-rate comparison;
no new gameplay, evaluation, training, checkpoint changes or capture fixes.

## Learning statistics and behavior

Both arms used the preserved small BC pilot, rollout 512 and 32,768 transitions.
The larger BC batch was not their initialization. The completed transition
audits passed with no discarded partial collection. See the original
[comparison results](../ppo_lr_comparison_20261029/primary_results.json).

| Statistic | Learning rate 0.0001 | Learning rate 0.0003 |
| --- | ---: | ---: |
| Median approximate KL | 0.0098 | 0.0274 |
| Median clipping fraction | 0.0705 | 0.148 |
| Median explained variance | 0.794 | 0.795 |
| Median value loss | 0.253 | 0.188 |
| Sampled LEFT / RIGHT during training | 25 / 84 | 57 / 63 |
| Recorded deaths | 319 | 327 |
| Time-limit truncations | 0 | 0 |
| Post-resume passive terminal transitions | 6 | 2 |

Optimizer statistics were parsed from 63 printed update summaries per arm;
the ledgers independently record 64 optimized rollouts. The printed log is not
a complete per-update tensor dump. Higher learning rate produced larger updates
without established survival improvement. Critic aggregates do not show an
obvious collapse, but do not establish accurate values at individual obstacles.

All three deterministic 100-episode evaluations (pilot and both continuations)
selected zero lateral actions. The ledgers record only 109 and 120 sampled
lateral actions respectively during training. Insufficient useful action
coverage is a hypothesis; causality has not been established.

The transition ledger records selected actions, dispatched commands, timing,
reward substeps and terminal transitions, but not historical value predictions
or every policy input tensor. Sparse early source previews and menu screenshots
cannot identify each collision cause. Last sampled actions on terminal rows
must not be interpreted as causal actions: some were suppressed by the guard.

## Confirmed image framing mismatch

All 30 saved detector-source images across baseline, low/high training and
low/high evaluations have dimensions 1920x983. Visually inspected examples
include approximately 28 pixels of Chromium address bar at the top and a small
taskbar strip at the bottom. PPO and detector use the shared cropped source in
`rs_v5_segmentation_live_benchmark.py::_native_frames`; this is not merely a
full-desktop diagnostic-preview issue.

The normalized crop is [0, .055, 1, .965]. Applied to a 1920x1080 desktop it
selects y=59:1042, producing 983 rows. All 321 demonstration manifests (151
approved, 170 candidate) instead describe a calibrated canvas region at
left=0, top=87, width=1920, height=945. The recorder applies the same fractions
inside that canvas: relative y=52:912, producing 860 rows. Equal fractional
parameters therefore do not establish equal physical inputs.

At 512x288, these shapes imply 262 versus 229 rows of resized content and
different letterbox padding. The saved frame 100 in the pilot recording
`demo_20261028_145438_471297/frames.npz` confirms nonzero rows 29 through 257.
Both paths are width-limited here; this is a framing/coordinate mismatch, not
evidence of a proportional object-scale change. It also changes normalized
detector coordinates used for auxiliary features. The plateau's cause remains
unproven, and not every historical policy tensor is available for inspection.

Several inspected early GAMEPLAY-labelled images show the opening animation
at zero metres. This warrants checking startup-state/control timing, without
assuming from screenshots alone that all such frames were uncontrollable.

The loaded screen-state checkpoint has four labels (MENU, LOADING, GAMEPLAY,
OTHER), a stored fractional crop and 192x112 input. Nine classes belong to the
hazard detector. Screen-state, pause-panel and diagnostic image paths have
separate purposes; do not crop every image identically or silently change the
classifier input distribution. Root cause of the desktop/canvas discrepancy
has not been conclusively isolated; fullscreen state and physical calibration
must be checked together.

## Decisions and outstanding work

No learning-rate winner, architecture change, recurrence or integrated BC loss
was adopted. Preserve the pilot, larger BC candidates and both PPO outputs.
Recordings remain reusable, but BC/PPO capture compatibility needs correction
before claiming a consistently aligned training dataset.

Recommended targeted work is shared verified canvas alignment for policy and
detector/BC images, offline framing checks, and a short synchronized diagnostic
of policy inputs, probabilities, dispatched actions and pre-collision gameplay.
No such new live diagnostic or crop fix has been completed in this inspection.

The user subsequently requested a three-hour autonomous PPO continuation from
the eight-hour plus larger-BC checkpoint. Its saved settings are rollout 128,
batch 32, five epochs, learning rate 0.0003, gamma .995 and lambda .95. An initial
command selected rollout 512 and rate .0001; rate .0001 was not an adopted winner.
Run configuration and effective-setting reports must confirm the actual launch.
Training completion and gameplay improvement remain unverified at publication.

## Publication verification

Before publication, the 12 offline discovery patterns listed in the retained
`offline_validation_pass.json` were rerun against the current tree: 104 tests
executed, zero failures or errors. This is a fresh observed count, distinct from
the historical 204-test record retained with the earlier validation artifacts.
No live browser, training or evaluation was launched for publication.
