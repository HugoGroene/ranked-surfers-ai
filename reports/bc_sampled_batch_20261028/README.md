# Larger BC experiment using a sampled review policy

The user explicitly requested inspection of about ten recordings followed by
batch use if reasonable, including useful short/medium runs and previous data.
This supersedes individual review for this particular experimental selection.
It is not a claim that every retained recording was visually reviewed or optimal.

## Review and selection

Seed 20261028: three random eligible short, three medium, three long recordings,
plus the longest eligible run with an interior gap. Inspected session runs:
64, 173, 56, 257, 282, 77, 132, 2, 209, 58. Review used full-run contact sheets
and dense frames around the proposed cut boundary, not continuous full-video
expert annotation. No obvious retained restart/death transition was found in
those sampled views. Earlier suboptimal actions may remain.

Rules for the first larger batch: a suggested retained prefix of at least three
seconds, at least two non-NONE action labels, at least 98% valid labels, matching
saved/recomputed labels and preview length, no interior >130 ms capture gap
between 2.15 seconds and the trailing-cut region, and valid four-frame histories.
Opening trim: 2.15 seconds. Trailing trim: 2.5 seconds. Actual bounds snap to
captured frames; label intervals cannot cross the end. These are heuristic cuts,
not independently confirmed collision times. Suspected mixed attempts were
excluded by the interior-gap rule, but a restart with no detected gap can remain.

142 new runs passed: 59 long, 66 medium and 17 short. Retained duration is about
36.38 minutes. New selected labels: LEFT=450, RIGHT=440, JUMP=638, ROLL=227.
Combined with the nine previously approved pilot runs: 151 runs and 47,978 valid
observations, including LEFT=491 and RIGHT=478. The 149 excluded new runs remain
candidates. Original manifests are backed up under the session review directory.
Only selected manifest approval/bounds metadata changed; raw recordings remain
intact. Selection JSON records every included/excluded run and its basis.

## Actor-only training

Source is the original eight-hour PPO checkpoint, not the smaller BC pilot.
121 whole training runs / 30 whole holdout runs, seed 77. After the existing NONE
subsampling: 2,389 training observations (NONE=836, LEFT=402, RIGHT=392, JUMP=555,
ROLL=204). Natural-distribution holdout: 8,812 observations (NONE=8467, LEFT=89,
RIGHT=86, JUMP=129, ROLL=41).

Architecture, observation schema, PPO rewards and gameplay control are unchanged.
The new offline helper caches outputs of the frozen extractor once. Actual
checkpoint checks verified equivalent probabilities and an equivalent actor
update versus the uncached path. It refuses stateful/stochastic layers. Training
uses globally shuffled cached examples rather than the old run-local minibatch
ordering; this implementation difference is recorded in config.json.

Adam learning rate 0.00002, batch size 12, up to 100 epochs, minimum 10, patience
10. Selection criterion is lowest natural-distribution holdout cross-entropy,
including the unchanged source as epoch zero. Per-action metrics are reported.

Training stopped after epoch 15; epoch 5 was best. Holdout cross-entropy improved
from 1.33423 to 1.16620, accuracy from 41.76% to 54.95%. These values should not be
compared directly with the small pilot's different holdout. The selected model
still has zero deterministic LEFT/RIGHT choices on the training/holdout demo
observations. Later epochs achieved some RIGHT recall (up to 12.8%) while overall
holdout loss worsened; they were not selected. LEFT recall remained zero.

Because 96% of the natural holdout is NONE, lowest overall loss does not establish
rare-action learning. This conflict is visible in the report; no claim is made
that this checkpoint solves lane changes. More recordings or epochs alone are
not established as the fix. Gameplay evaluation is still required before any
PPO continuation; the small pilot's 4.83-second live result remains the reference.

## Saved checkpoint and verification

`ai_data/demonstrations_v13/bc_runs/bc_sampled_batch_20261028_1740_v2/latest.zip`

SHA-256: `9963b7229d2aa5dae4514cad359c1f904d6ee24f721137ab298d55d7618c5936`.
Four actor tensors changed; all 40 other saved policy tensors remain bitwise
identical to source. Independent ZIP CRC and PPO.load compatibility checks pass:
12x288x512 image, 557 auxiliary features, five actions. Source and pilot hashes
remain unchanged. The subsequent matched 100-episode evaluation completed; see the results below.

The first attempt completed optimization but its save failed because the atomic
helper permits only latest.zip. The corrected run used the same verified caches
in a new directory and reproduced the epoch metrics, then saved successfully.
The failed-at-save directory is preserved as diagnostic output, not a checkpoint.

Five cache-integrity tests passed. The existing seven trim-planner tests passed;
actual source-policy cache/update parity and independent frozen-weight checks
also passed. The normal PPO optimizer can resume the saved checkpoint; it is
not a separate incompatible actor format.

## Gameplay evaluation completed

Mean survival was 4.48 seconds versus 4.8276 seconds for the small pilot;
40/100 reached five seconds, versus 47/100. There were no LEFT/RIGHT choices
and no guard aborts. The larger model has not established an advantage.
See [evaluation and next-step assessment](../bc_large_evaluation_20261028_180438/README.md).
