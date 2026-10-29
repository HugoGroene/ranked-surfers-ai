# Controlled BC optimization comparison

## Scope

Existing approved/trimmed recordings only, including short/medium runs and the
nine pilot recordings. No raw recording, manifest, reward, observation schema,
actor architecture or gameplay guard was changed. NONE examples remain in the
training data. No penalty, quota or reward for lateral actions was introduced.

## Small-subset optimization controls

The same three fixed balanced subsets (80 observations, 16 per action) were
fitted in RAM for up to 10,000 updates. Source was the existing larger BC model;
all non-actor tensors remained frozen. No diagnostic weights were saved.

| Learning rate | Seed 77 accuracy | Seed 78 | Seed 79 |
| --- | ---: | ---: | ---: |
| 0.0001 | 71.25% | 73.75% | 75.00% |
| 0.0003 | 82.50% | 87.50% | 88.75% |
| 0.0010 | 82.50% | 82.50% | 85.00% |

All five actions were learnable as outputs, but no trial reached the predefined
95% accuracy plus minimum 80% per-class recall target. More updates helped
relative to the 3,000-update diagnostic. This does not prove that the frozen
representation is incapable, nor does memorization establish good gameplay.

## Full-dataset comparison

All three experiments start from the preserved eight-hour PPO source, not from
a previously BC-updated actor. Same 121 training / 30 whole-run holdout split,
seed 77, 2,389 training samples, batch size 12, same NONE subsampling and cached
frozen features. The holdout contains 8,812 observations; 8,467 are NONE.

To prevent this 96% NONE fraction from determining selection almost alone,
selection uses the arithmetic mean of the five per-action cross-entropies.
Natural-distribution loss and all per-action metrics are also reported. This
metric is an experimental BC selection option, not a change to PPO rewards or
the original trainer default. The learning-rate comparison varies only learning
rate within this common setup. Relative to the preceding BC experiment, the
selection criterion and allowed optimization budget also differ; causal claims
cannot attribute the entire difference to learning rate.

Budget: at most 100 epochs, minimum 20, patience 20. The 0.00002 and 0.0003 runs
used the full budget; convergence is not established. Results below are for each
selected checkpoint, not the final epoch unless they coincide.

| Learning rate | Selected epoch | Macro holdout loss | NONE recall | LEFT | RIGHT | JUMP | ROLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00002 | 100 | 1.7475 | 73.6% | 3.4% | 10.5% | 20.2% | 0.0% |
| 0.00010 | 62 | 1.6264 | 87.0% | 0.0% | 0.0% | 29.5% | 0.0% |
| 0.00030 | 99 | 1.5574 | 78.7% | 2.2% | 1.2% | 44.2% | 7.3% |

Recall measures exact agreement with recorded actions, not the set of all safe
actions. Multiple safe routes and human timing variation can reduce agreement.
The holdout has only 89 LEFT, 86 RIGHT and 41 ROLL labels, so rare-action metrics
are imprecise. These comparisons share one split/seed and use that holdout for
model selection; they are not independent confirmation of generalization.

## Event alignment check

Three training recordings were re-analyzed from original timestamps and input
events. All saved action/valid labels reproduce exactly. All 61 Windows keydowns
had corresponding browser events and assigned labels (11, 15, 35 per run).
Assigned key offsets after their observation timestamp range from 0.03 to 62.38
ms across those recordings, within the existing 130 ms labeling bound. These
are whole-recording diagnostics, including excluded tails; they do not certify
every training sample. No new visual judgment that the moves were optimal was
made. Clock agreement/event assignment does not independently measure rendered
frame latency or Unity input application time.

## Outcome and next step

The 0.0003 candidate has the lowest macro holdout loss and strongest JUMP/ROLL
recall among these three selected checkpoints. It is an offline candidate, not
a demonstrated survival improvement. Lateral recall remains weak; NONE is still
recognized, and no frequent lane changes are required by this experiment.

Next useful check is the same guarded 100-episode gameplay evaluation against
the 4.83-second pilot, considering survival, action counts and timing together.
Do not infer better gameplay from holdout accuracy or launch overnight PPO on
these metrics alone. No live evaluation or PPO continuation was launched here.

Selected candidate:

`ai_data\demonstrations_v13\bc_runs\bc_control_macro_lr3e4_20261028\latest.zip`

## Verification and artifacts

Independent loads of all three saved candidates verified exactly four actor
tensors changed and all 40 other policy tensors match source bitwise. Model
hashes match their training reports; the eight-hour source hash is unchanged.
Original pilot and prior larger BC checkpoints remain preserved. Five existing
cache-integrity tests passed. A selection-ranking check verified that natural
and macro metrics distinguish the expected imbalance tradeoff, and that missing
holdout classes are rejected for macro selection.

See full_dataset_comparison.json, small_subset_comparison.json and
alignment_audit.json. Complete training configurations, per-epoch metrics,
feature caches and checkpoints reside in their unique bc_control_macro output
directories. The small-subset trials retain exact indices and input hashes in
the separate bc_fit_control_lr reports. No Git commit or push was performed.

## Gameplay evaluation completed

The selected candidate averaged 3.6653 seconds over 100 episodes versus the
pilot's 4.8276 seconds. LEFT/RIGHT selections were 91/9, but survival worsened;
there were no guard aborts. It is not promoted over the pilot. See the
[evaluation and pipeline assessment](../bc_macro_gameplay_20261028/README.md).
