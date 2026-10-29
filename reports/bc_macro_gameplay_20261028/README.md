# Macro-selected BC candidate: 100-episode gameplay evaluation

The candidate is bc_control_macro_lr3e4_20261028/latest.zip. All 100 episodes
completed. Chromium foreground activation succeeded automatically. Configurations
match the 4.83-second pilot except checkpoint and run name. Source, pilot,
previous larger BC and candidate checkpoint SHA-256 hashes remained unchanged.
No runtime source, rewards, observation schema, guards or training settings were
changed during this evaluation. No PPO continuation was launched.

| Metric | Small BC pilot | Previous larger BC | Macro BC candidate |
| --- | ---: | ---: | ---: |
| Mean survival | 4.8276 s | 4.4800 s | 3.6653 s |
| Median survival | 4.275 s | 4.200 s | 3.485 s |
| Best survival | 11.55 s | 10.05 s | 9.31 s |
| At least 5 seconds | 47/100 | 40/100 | 10/100 |
| At least 10 seconds | 4/100 | 1/100 | 0/100 |
| LEFT / RIGHT selected | 0 / 0 | 0 / 0 | 91 / 9 |
| Guard aborts | 0 | 0 | 0 |
| Guard hard stop | No | No | No |

Action counts describe policy selections. They are not independent evidence of
Unity response or successful avoidance; some selections may be suppressed when
the subsequent guard confirms a death/menu. The new candidate has escaped the
zero-lateral-selection pattern, but survival is worse. Do not promote it over
the pilot for a longer continuation.

Candidate minus pilot mean is -1.1623 seconds; independent episode bootstrap 95%
interval [-1.6278, -0.7135]. Versus previous larger BC: -0.8147 seconds,
[-1.2299, -0.4156]. These are separate sessions, not paired identical maps.
Intervals omit session/map-distribution and training-seed variation, so they do
not isolate the causal effect of particular actions. No death-cause attribution
was performed; jumping/coin timing as the principal learned skill remains a
hypothesis. Offline imitation improvements did not translate into better
survival in this evaluation.

## Timing and the underlying system

Median gameplay-only interval was 51.77 ms (about 19 observations/second), p95
65.35 ms and p99 69.66 ms. Previous larger BC: 51.65 / 65.34 / 69.78 ms. The
candidate regression did not coincide with a broad slowdown in these metrics.
Pilot median was 53.36 ms. These periods are not capture-to-Unity reaction times.

YOLO source age at observation packing was 112.7 ms median, 188.9 ms p95;
reported fresh fraction was 0.969. The previous larger run was 106.0 / 182.1 ms,
fresh fraction 0.975. Fresh means age below the 300 ms gate, not latest imagery.
Statistics are kept separately from raw RGB; the policy also receives a newer
four-frame RGB stack. Detector age statistics retain the latest 1,000 fresh
samples, not necessarily all decisions. Full frame/feature ages at the moment
Unity actually responds have not been established by these reports.

Source inspection found this order:

1. The evaluation loop predicts an action from the previously returned RGB/aux
   observation (train_rs_v612_ppo.py, evaluate).
2. The runtime guard captures and classifies a newer screenshot before keys
   (same file, make_env_class._step_browser).
3. It dispatches the already selected action when gameplay is verified; that
   guard image does not recompute the chosen action.
4. TimedEnv sends a tap, captures again and the environment packs the next
   policy observation (rs_v2_timing_probe_ab.py; train_rs_v612_ppo.py._pack).

The legacy frame_age_at_command_ms uses last_capture_end after the guard
capture; it can therefore understate the age of the actual policy input. That
timestamp is also assigned after menu/resize processing, rather than at desktop
presentation. A five-millisecond requested key hold is not end-to-end latency.
These are measurement distinctions, not proof of a broken controller.

The architecture has RGB history plus timing/action/hazard context and can
plausibly learn substantially better anticipation without a rewrite. More PPO
steps are not guaranteed to discover steering after jumping is mastered. The
current BC experiments freeze the visual/aux extractor, so they only change
how existing features map to action probabilities. Ordinary PPO continuation
can also train those features. These tests do not isolate which learning
component is limiting performance.

The optimizer pause mechanism prevents optimization time from counting as
active game time, but it does not provide slow-motion perception during active
play. Training-only resynchronization passively collects four frames after
resume; supplied overnight traces show roughly 213-277 ms before the fourth
accepted frame. That first-action gap merits measurement separately from cached
frame age. This evaluation does not invoke optimizer pause/resync.

Finally, the 15-second episode cap and currently short survival do not validate
sustained high-speed play. Reacting early based on motion can compensate for
finite latency, but that predictive behavior has not been demonstrated here.

## Recommended next step

Retain the 4.83-second pilot as the best observed survival reference. Before
another overnight run, perform a focused latency audit on the unchanged policy:
track exact policy-input frame IDs/timestamps, policy inference, guard work,
key dispatch, browser receipt and observable Unity movement; report median,
p95/p99 and outliers. Distinguish RGB and YOLO source age, and measure the
post-resume first-action gap in a separate short training test. Preserve guards.

If measurements identify a bottleneck, A/B one scheduling/capture setting at a
time on the same checkpoint. Otherwise proceed with a short, evaluated PPO
continuation from the pilot rather than assuming this weaker BC model is a
better starting point. No major pipeline rewrite or more demonstration
collection is justified by this evaluation alone.

The general learning caution is consistent with SB3's discussion of poor
trajectories perpetuating poor learning and the need for quantitative custom
environment experiments:
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

See comparison.json, pipeline_assessment.json, checkpoint_hash_verification.json
and the copied unmodified raw reports/configurations in this directory.
