# Controlled autonomous PPO learning experiment

Status: **COMPLETE**. All seven stages completed in one owned Chromium window,
with six reuses and clean closure. All 98,304 requested training steps and all
400 evaluation episodes completed. No uncertain-state abort or hard guard stop
occurred. Original checkpoint hashes and runtime code hashes are unchanged.

The pipeline is faster, but **none of these continuations demonstrated improved
survival**. Keep the preserved pilot as the reference; do not promote an arm on
the basis of training throughput alone.

## Evaluation results

Each evaluation used 100 deterministic episodes and a 300-second limit. All
400 episodes ended through confirmed death; none reached the time limit.

| Model | Mean | Median | Best | Reached 5 s | Reached 10 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh baseline | 4.6494 s | 4.525 s | 10.06 s | 48 | 3 |
| A: rollout 128 | 3.9852 s | 4.160 s | 9.94 s | 23 | 0 |
| B: rollout 512 | 4.3799 s | 4.165 s | 10.08 s | 35 | 1 |
| C: rollout 512, KL 0.03 | 4.1714 s | 3.810 s | 10.03 s | 29 | 1 |

Independent episode-bootstrap 95% intervals for mean differences (20,000
resamples, seed 123):

| Comparison | Mean change | 95% interval |
| --- | ---: | ---: |
| A minus baseline | -0.6642 s | [-1.1206, -0.2185] s |
| B minus baseline | -0.2695 s | [-0.7573, +0.2248] s |
| C minus baseline | -0.4780 s | [-0.9704, +0.0096] s |
| B minus A | +0.3947 s | [-0.0260, +0.8218] s |
| C minus B | -0.2085 s | [-0.6635, +0.2600] s |

These intervals condition on the trained models and evaluation sessions. They
exclude training-seed and procedural-map/session uncertainty and are not
multiple-comparison adjusted. A regressed in this sample; the other comparisons
remain inconclusive. This is not proof that PPO cannot improve with more steps.
All deterministic evaluations selected only NONE/JUMP/ROLL. This is recorded as
a diagnostic, not a requirement to reward or force lateral actions.

## Efficiency and optimizer behavior

| Arm | Training wall time | Decisions/s | Reported paused time | KL median / max | KL early stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2,039.77 s | 16.06 | 298.573 s | 0.0292 / 0.1479 | 0 |
| B | 1,757.64 s | 18.64 | 208.812 s | 0.0359 / 0.0761 | 0 |
| C | 1,654.09 s | 19.81 | 82.825 s | 0.0238 / 0.3880 | 63 |

B's observed throughput was about 16% higher than A's. Survival/reset frequency
also differs, so the entire wall-time difference cannot be attributed to rollout
length. C skipped optimizer work and was faster again, but did not improve
evaluation survival. Target KL is an approximate stop criterion, not a hard
upper bound; C's logged maximum shows why it should not be treated as one.
Some stops occurred in the first epoch. The displayed optimizer summaries omit
the final update because SB3 prints the preceding training statistics at rollout
logging; model metadata and full early-stop messages are preserved separately.

Saved total timesteps are 1,695,195 for each arm. Saved `_n_updates` values are
66,205 / 65,245 / 65,048 from a common 64,925. These count optimization epochs,
not minibatch optimizer steps; different rollout sizes make them incomparable
as direct measures of training work.

## Runtime and safety verification

- All three arms passed `PASS_STABILITY_CANDIDATE` and `PASS_REWARD_REPLAY`.
  Continuation counts were 255 / 63 / 63, matching effective rollout sizes.
- The shared-observation guard recorded 31,769 / 31,836 / 31,950 prepared hits.
  Fallback and pre-dispatch expiry checks remained active; no threshold changed.
- Arm A reached 17.09 seconds during training, demonstrating that the old
  15-second cap no longer limits episodes. This is not evidence of minute-long
  model capability.
- Arm A's log warns that its final rollout ended on MENU without a subsequent
  step to record that final terminal penalty. This known boundary limitation is
  retained in the audit; no post-hoc reward or checkpoint adjustment was made.
- Replay validates reward arithmetic and clock checks validate internal
  consistency. Neither establishes Unity internal time, unbiased reward caps or
  time-normalized gamma. Capped intervals remain recorded for review.
- A screenshot gate still has a finite capture-to-key race. The shared-observation
  path remains opt-in; legacy fallbacks need not use the same policy image.

## Recommended next experiment

Keep the faster runtime and persistent-browser support available. Treat rollout
512 as a throughput candidate, not a validated survival improvement. Preserve
the original pilot and all three output checkpoints.

Before another unattended eight-hour continuation, compare a smaller learning
rate (for example 0.0001 versus 0.0003) from the same preserved source with
rollout 512 and the other settings fixed. Frequent early KL stops and survival
regression make update size a stronger next question than adding memory or
forcing specific actions. Repeat across training seeds or repeat evaluations
before adopting a winning setting. This is a proposed next experiment, not an
additional run launched automatically.

No BC loss, new demonstrations, recurrent model, detector removal or cloud
deployment was introduced. No source changes were made during this trial, and
no Git commit or push was performed.

## Reproducibility

- [Full numerical results and hashes](results.json)
- [Exact plan and source hash](plan.json)
- [Manifest](manifest.json), [effective stage preview](effective_manifest_preview.json)
- [Batch state and stage report paths](batch/batch.json)
- [Analysis script](analyze_results.py)
- [Pipeline implementation and latency validation](../ppo_pipeline_foundation_20261028/README.md)

Training checkpoint outputs are `ai_data/rl_v2_runs/ppoctrl_20261028_02_train/latest.zip`,
`.../ppoctrl_20261028_04_train/latest.zip`, and `.../ppoctrl_20261028_06_train/latest.zip`.
The reference remains `ai_data/demonstrations_v13/bc_runs/bc_20261028_152607/model.zip`.

## Fixed protocol

The exact-observation gameplay guard is enabled throughout. Rewards, observation
schema, architecture, source checkpoint, detector, key taps, 300-second episode
limit, learning rate 0.0003, minibatch 32, epochs 5, gamma 0.995 and entropy 0.002
are fixed. No BC updates, movement rewards/cooldowns or memory changes are used.

| Arm | Additional PPO steps | Rollout size | KL stopping threshold |
| --- | ---: | ---: | ---: |
| A | 32,768 | 128 | None |
| B | 32,768 | 512 | None |
| C | 32,768 | 512 | 0.03 |

A fresh 100-episode baseline precedes training because the optimized runtime and
longer episode cap differ from historical measurements. Every arm starts from
the same preserved pilot checkpoint, not the preceding arm's weights, and gets
its own 100-episode deterministic evaluation. All training requires pause/clock
and reward-replay audits before the next stage. Source hashes and effective PPO
settings are recorded. Original source models/recordings remain untouched.

This comparison tests rollout length, then KL stopping at the larger rollout.
Learning rate and exploration are held fixed rather than changing everything
at once. One training seed and sequential unpaired procedural maps limit the
conclusions. Equal step counts do not imply equal wall time or active-gameplay
exposure. Action counts remain diagnostics; success is survival/control and
learning efficiency, not requiring LEFT/RIGHT frequency.

The results above include before/after survival, bootstrap uncertainty, runtime
throughput, optimizer traces and audits. The historical 4.83-second pilot is
context, not the matched runtime baseline for this experiment.
