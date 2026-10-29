# First actor-only BC pilot: gameplay evaluation

Nine explicitly approved, trimmed human runs; seven training and two held out.
183 BC training examples after NONE subsampling. Three epochs, batch size 12,
learning rate 0.00002, seed 77. Source: latest eight-hour v1.7.9 PPO checkpoint.
Only actor tensors changed; source and BC checkpoint SHA-256 values remain
unchanged after evaluation. Raw reports/configurations are copied unmodified.

| Metric | Source PPO | BC pilot |
| --- | ---: | ---: |
| Deterministic episodes | 100 | 100 |
| Mean survival | 4.2970 s | 4.8276 s |
| Median survival | 4.190 s | 4.275 s |
| Best survival | 10.08 s | 11.55 s |
| At least 5 s | 32 | 47 |
| At least 10 s | 2 | 4 |
| LEFT / RIGHT | 0 / 0 | 0 / 0 |
| NONE | 1836 | 2263 |
| JUMP | 3987 | 4426 |
| ROLL | 1576 | 1502 |
| Guard aborts | 0 | 0 |
| Median gameplay interval | 51.91 ms | 53.36 ms |
| Detector fresh fraction | 0.973 | 0.976 |

Observed mean difference: +0.5306 s (about 12.35%). Independent episode-bootstrap
percentile 95% interval: [+0.0283, +1.0471] s, 20,000 draws, default_rng(123).
This is encouraging survival evidence, but only one evaluation per checkpoint.
It does not include training-seed, procedural or session variability. Actual
capture timing differed slightly. CLI configurations differ only by source
checkpoint and run name. Both use the existing 15-second episode cap.

Mean LEFT+RIGHT probability increased from 0.45% to 0.60%, but neither policy
selected a lateral action. Therefore this experiment has not demonstrated the
intended lane behavior. Offline BC metrics also show zero lateral recall on
both training and held-out demo observations. This means insufficient dataset
size is not the only plausible limitation; fitting/optimization and frozen
representation remain unresolved. More data could improve generalization, but
we should first check whether the actor can learn the current training examples.

Recommended next checks: repeat the matched gameplay comparison to assess
session variability; then use a separate, short offline BC fitting diagnostic
with per-action metrics before committing to extensive new recording or a long
PPO continuation. Preserve both source and BC checkpoints. No new learning run
was launched as part of this gameplay evaluation.

The startup foreground and browser-focus checks passed in this live evaluation.
That is one successful startup, not a guarantee that Windows will always allow
automatic foreground activation.
