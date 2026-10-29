# Equal-budget PPO continuations from two BC checkpoints

Both branches completed 4,096 additional PPO steps and a 100-episode deterministic
local Practice evaluation. No demonstration-loss integration, detector ablation,
architecture, reward or runtime control change was made. Every source checkpoint
and recording was preserved; new models were saved in unique run directories.

## Outcome

| Starting point | Mean before | Mean after PPO | Median after | At least 5 s after | At least 10 s after | LEFT / RIGHT after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small BC pilot | 4.8276 s | 4.6238 s | 4.525 s | 46/100 | 0/100 | 0 / 0 |
| Newer macro-selected BC | 3.6653 s | 4.5058 s | 4.180 s | 38/100 | 3/100 | 1 / 0 |

Best episode after continuation: pilot 7.53 s; newer BC 11.50 s. All four before/
after evaluations used 100 episodes with the same configuration apart from
checkpoint/run name. The older pilot's before result remains the highest mean
among these models; the continued models have no established difference.

Independent-episode bootstrap mean differences (20,000 resamples, seed 123):

| Contrast | Difference | 95% interval |
| --- | ---: | ---: |
| Pilot after minus before | -0.2038 s | [-0.7351, 0.3082] |
| Newer BC after minus before | +0.8405 s | [0.4045, 1.2840] |
| Newer BC after minus pilot after | -0.1180 s | [-0.6135, 0.3926] |
| Newer BC after minus original pilot | -0.3218 s | [-0.8650, 0.2302] |

These are sequential unpaired map sessions and one training seed. Intervals
condition on the observed episodes and omit session/map-distribution and training-
seed variability. They do not isolate the causal benefit of BC. In particular,
the pilot comparison branch also started with a BC checkpoint: this experiment
compares two BC starting points, not BC versus no BC.

The newer BC checkpoint recovered survival after PPO, supporting the possibility
that a weaker immediate BC score need not doom subsequent RL. It has not shown
a clear survival advantage over the pilot. Its 91 LEFT / 9 RIGHT deterministic
selections before continuation fell to 1 / 0 afterward. Counts are policy
selections, not independent proof of Unity movement or successful avoidance.

Mean lateral probabilities remain larger in the newer BC continuation:
LEFT=0.0320 and RIGHT=0.0474, versus pilot continuation 0.0019 / 0.0033.
They declined from newer BC's 0.0656 / 0.0661. Thus lateral options remain in the
stochastic policy, but almost never become its deterministic choice. The newer
branch sampled both directions during training; this is not reliable steering.

## Equal-budget checks

The starting models differed in exactly four actor tensors; the other 40 saved
policy tensors were identical. Both had the same saved PPO settings and began
at 1,662,427 total steps and 64,925 SB3 updates. Both outputs independently load
at 1,666,523 total steps and 65,085 updates: 32 rollouts of 128 steps, each with
five PPO epochs. All policy parameters were trainable for the continuations.

Training and evaluation configurations compare equal apart from source/run name.
Both used 512x288 RGB, 557 aux features, deterministic 100-episode evaluation,
15-second episode cap, seed 123, uncapped rate, 5 ms requested key hold,
YOLO imgsz 960, detection every 3 decisions and a 300 ms freshness cutoff.

Timing remained comparable. Evaluation median/p95/p99 observation intervals:
pilot continuation 53.91 / 65.40 / 69.73 ms; newer BC continuation
55.36 / 65.43 / 69.81 ms. These are not Unity input-to-visible-response delays.

## Optimizer pausing and reward timing

Both training runs explicitly enabled --pause-ppo-optimizer, --resync-rollouts
and --audit-reward-timing. Pausing clicks the game's Pause control, verifies a
frozen track, then clicks Resume; it intentionally avoids Escape because it can
exit fullscreen. Episode/action/monitor clocks are corrected for the verified
pause. Four checked fresh gameplay frames are required before the next rollout.
No optimizer runs during evaluation, so evaluation does not pause periodically.

Both branches passed PASS_STABILITY_CANDIDATE with 31 optimizer continuations
and PASS_REWARD_REPLAY for all 4,096 steps. There were no live guard aborts or
hard stops in either completed training or evaluation. Reported active-clock
drift was zero, an internal consistency check rather than an independent Unity
timer reading. Reward replay validates trainer arithmetic; it does not establish
that reward caps or per-step discounts are unbiased. Full audit warnings remain
in the copied JSON reports. Eighteen pause-runtime tests and the synthetic
reward-clock self-test passed before live work.

## Startup interruption and recovery

The initial newer-BC launch was refused foreground focus by Windows. The guard
stopped before any episode, PPO action or training update. Its folder has startup
screenshots/configuration only and was preserved. The initial automation then
stopped because there was no completed training report. This infrastructure
failure was not counted as a model safety failure or included in the step budget.

The newer branch was retried from its original unchanged BC source in a new
training directory; focus was verified, and the full requested budget completed.
The pilot branch was not repeated. experiment.json records the resolved startup
error and recovery. There are no training/evaluation processes left running.

## Preserved outputs

Pilot continuation:

`ai_data/rl_v2_runs/v612_ab_pilot_train_20261028/latest.zip`

Newer BC continuation:

`ai_data/rl_v2_runs/v612_ab_macro_bc_train_retry1_20261028/latest.zip`

Evaluation reports:

- ai_data/rl_v2_runs/v612_ab_pilot_eval_20261028/v61_eval_report.json
- ai_data/rl_v2_runs/v612_ab_macro_bc_eval_20261028/v61_eval_report.json

Source/output model hashes are in checkpoint_hash_verification.json. All 77
recorded local source-file hashes remained unchanged throughout the comparison.
Raw logs, configuration commands, audits and copied original reports are kept
here. Original recordings were never curated or used for BC updates in this task.
No Git commit or push was performed.

## Next-step assessment

Keep the original pilot as the best observed survival reference and preserve the
newer BC-to-PPO checkpoint as a viable continuation candidate. The latter has
recovered survival and retains more lateral probability, but has not demonstrated
reliable steering or superiority. This 4,096-step diagnostic cannot settle the
user's longer-term BC hypothesis.

If continuing learning, a longer monitored continuation with periodic matched
evaluations is more informative than assuming this short test proves either BC
success or failure. Track lateral choices and outcomes alongside survival. Demo-
loss integration remains a later separate experiment as requested; the current
pipeline already resumes BC-produced PPO checkpoints.

Detector removal also remains a separate controlled ablation. YOLO is
asynchronous: 114 ms feature age is not a 114 ms blocking delay in every decision.
An image-only/no-YOLO model could reduce GPU work but would lose learned context
and needs retraining/evaluation, not just a speed claim from disabling inputs.
No detector setting was changed here. See ../ppo_reaction_latency_20261028 for
the measured capture-to-command latency and detection-frequency tradeoff.
