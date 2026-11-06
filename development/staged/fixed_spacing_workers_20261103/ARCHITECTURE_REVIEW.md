# Review of auxiliary inputs and spacing advice

Source reviewed: `road_models.py`, `road_adapter.py`, `live_adapter.py`, `hybrid.py`,
`rs_v6_ppo_features.py`, `rs_v61_features.py`, `train_rs_v3_ppo.py`,
`train_rs_v612_ppo.py`. The benchmark does not change this architecture or PPO settings.

## What the implementation already does

The spatial extractor produces 128 visual features through three strided convolutions,
an 8x8 adaptive average-pooling grid and a linear layer. A separate auxiliary MLP maps
the input vector to 96 then 64 features. Their concatenation is 192-dimensional and
feeds the actor and critic, each with a 64-unit head before its output. Offline checks
verify that both heads use the same feature-extractor instance.

This is already the proposed late-fusion design, including critic access. The old tiny
encoder also had separate visual/auxiliary branches. Sharing is the SB3 PPO default;
separate actor/critic extractors could also work if both received the relevant inputs.
[SB3 custom policy documentation](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).
Providing inputs does **not guarantee** that the learned policy uses them meaningfully.
Matched ablations would be needed to demonstrate that.

The vector contains four recent action entries, one-hot across five actions (20 values),
three recent observation intervals normalized by 0.5 s, and time since each of the five
action types clipped/normalized to one second. These are action/command context, not
proof that Unity executed every requested movement. There is no full timestamped list
of all actions from the last second. Four action entries are a count-based window,
whose time span varies with decision speed. The three legacy interval values describe
recent observation updates, not the gaps between the new sparse image slots.

The temporal adapter separately appends **each selected image's actual age and validity**.
For the four-frame schedule, ages are divided by 0.75+0.06 seconds and clipped to [0,1].
There are eight additional values, making 565 auxiliary values total. The original
five-frame schedule has ten additional values (567 total). Detector-derived slots
remain unavailable with the detector disabled. No new detector or lane-odometry
assumption is introduced.

## Which recommendations need qualification

1. **Action history need not be exactly the image-history duration.** Adding bounded
   history can help address partially observed state; it does not itself violate PPO.
   Neither window length nor a one-second cutoff guarantees a Markov representation.
   The relevant distinction is state versus partial observation, described in
   [Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).
   Extending to a full timestamped action sequence would be a separate schema experiment.

2. **Watch entropy, but do not automatically raise it.** More entropy pressure encourages
   broader action probabilities; it does not specifically teach lateral maneuvers or
   ensure useful exploration. Low entropy can mean learned competence or premature
   collapse. Evaluate behavior and return alongside it. Keep the existing 0.002
   coefficient for this performance comparison. SB3 defines `ent_coef` as the entropy
   coefficient in the loss: [PPO parameters](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

3. **The proposed offsets are irregular, increasingly spaced schedules, not strictly
   logarithmic.** Their appeal is combining near-term motion and older context with
   a small stack. There is no special mathematical benefit established for these numbers.
   Four sparse samples cover the same oldest-to-newest horizon as many dense samples
   but do not contain the same temporal information; intermediate events can be missed.

4. **CNN depth does not automatically separate temporal frequencies.** The first
   convolution mixes all stacked RGB channels. Deeper layers do not automatically
   specialize in older frames, nor do early layers exclusively handle recent frames.
   Useful temporal processing must be learned from the inputs and objective.

5. **Keep actual ages; do not substitute slot indices.** Indices would hide capture
   jitter and missing history. Raw `log(age+epsilon)` has an arbitrary large negative
   value near age zero and is not inherently better aligned with CNN features.
   Our bounded linear ages plus validity flags are already finite and interpretable.
   A normalized log1p transform could be tested later, but it adds another experimental
   variable and is not needed for this worker-count benchmark.

## Performance expectations

At a fixed resolution/encoder, changing five frame ages leaves the tensor shapes and
CNN computation unchanged. A shorter retained buffer, selection behavior, missing
slots and different gameplay can still change observed performance slightly. Another
full five-frame sweep was therefore dropped at the user's request.

The new 384x384 five-frame test reached roughly 22 optimized transitions/s with three
workers; the older small-encoder setup reached roughly 33-35/s. These are different
systems/checkpoints and not a controlled regression comparison. 384x384 has exactly
the same pixel count as 512x288; five frames therefore hold 25% more image values than
the old four-frame input, despite a smaller width. The new spatial encoder retains
larger intermediate feature maps, and the asynchronous history pump performs additional
capture/crop/copy work. Profiling and different reset rates also matter. The previous
worker-scaling evidence suggests three workers as a candidate, not a guaranteed optimum
for the four-frame schema. This short suite will measure it with actual optimizer work.
