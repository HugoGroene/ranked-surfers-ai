# Larger BC: matched 100-episode gameplay evaluation

All three evaluations used the same evaluation configuration apart from run name
and checkpoint: deterministic policy, 100 episodes, 15-second episode limit,
512x288, fast capture, uncapped rate, 5 ms press delay and the same state/hazard
models and guards. These were separate gameplay sessions, not paired seeds.

| Metric | Eight-hour PPO | Small BC pilot | Larger BC |
| --- | ---: | ---: | ---: |
| Mean survival | 4.297 s | 4.8276 s | 4.48 s |
| Median survival | 4.19 s | 4.275 s | 4.20 s |
| Best survival | 10.08 s | 11.55 s | 10.05 s |
| At least 5 seconds | 32/100 | 47/100 | 40/100 |
| At least 10 seconds | 2/100 | 4/100 | 1/100 |
| LEFT / RIGHT choices | 0 / 0 | 0 / 0 | 0 / 0 |
| Guard aborts | 0 | 0 | 0 |

The larger model is not an observed improvement over the pilot. Mean difference
is -0.3476 seconds; an independent-episode bootstrap 95% interval is
[-0.8809, 0.1726] seconds. Against PPO it is +0.1830 seconds, interval
[-0.2768, 0.6352]. Neither comparison establishes a clear advantage. These
intervals omit session and training-seed variability.

Capture timing was comparable: median 51.91 / 53.36 / 51.65 ms respectively;
fresh detector fractions were 0.973 / 0.976 / 0.975. All 100 larger-model
episodes completed; no uncertain-state abort or guard hard stop was reported.
Automatic window activation failed; manual focus succeeded before gameplay.

Larger-model mean LEFT/RIGHT probabilities increased to 0.0027 / 0.0251,
from the pilot's 0.0006 / 0.0054. They never became the deterministic choice
in live evaluation. Increased probability alone is not useful lane behavior.

## Assessment and next experiment

Keep the pilot as the best observed survival reference and preserve all models.
Do not begin a long PPO continuation based on this result.

The larger BC selection used natural-distribution holdout cross-entropy, with
8,467 NONE observations out of 8,812 (96%). It selected epoch 5 with zero
LEFT/RIGHT recall, while later epochs showed some RIGHT recall as overall loss
worsened. This is evidence that the selection objective can obscure rare-action
learning; it does not prove that a later epoch would play better. LEFT recall
remained zero. More demonstrations or training time alone are not established
as the remedy.

Recommended next step: a controlled offline BC diagnostic on the existing
reviewed data. First test whether the frozen representation and actor can fit a
small balanced action subset. Then compare action-balanced training/selection
using whole-run holdouts, reporting per-action recall and cross-entropy as well
as natural-distribution loss. Keep changes confined to a separate BC experiment.
Check label timing around representative lateral events if fitting fails, and
inspect held-out lateral decisions before a new live evaluation. Only consider
short PPO continuation after a BC candidate demonstrates useful lane choices
and competitive survival. No additional training was launched by this evaluation.

Raw evaluation reports, comparison statistics and checkpoint hash verification
are included here. All three checkpoint hashes remained unchanged.
