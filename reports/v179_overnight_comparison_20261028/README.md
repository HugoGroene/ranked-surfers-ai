# v1.7.9 overnight continuation: 28 October 2026

Local Practice only. Raw evaluation reports/configurations are copied unchanged
into this directory. Model weights, screenshots, recordings and training logs
are not included.

## Runtime and training

v1.7.8 permits bounded passive retries of weak-menu UNCERTAIN captures without
counting them as gameplay. v1.7.9 excludes the live Pause DOM panel and close
visual matches to a verified paused screenshot from every accepted resync frame.
A partial stack is discarded; four consecutive confident GAMEPLAY captures are
required. Persistent ambiguity stops safely. The visual matcher is specific to
the current panel layout, not a general learned Pause classifier.

Training run: `v612_paused_state_v179_8h_20261028_045909`.
386,829 environment steps, 5,553 completed episodes, 28,803.94 wall seconds;
training mean 4.18 s, maximum 15.0 s; zero reported uncertain-state aborts or
guard hard stops. The 480-minute limit ended the run before its deliberately
large step target. Reward replay passed all 386,829 steps. The stability auditor
reported FAIL_SAFE_AUDIT solely because requested steps were not completed;
this is not an unconditional stability-audit pass. Reward replay validates
arithmetic, not Unity internal time. The survival cap and per-decision discount
remain unchanged.

## Matched deterministic evaluations

| Metric | Before | After |
| --- | ---: | ---: |
| Episodes | 100 | 100 |
| Mean survival | 3.9933 s | 4.2970 s |
| Median survival | 3.445 s | 4.19 s |
| Best survival | 11.44 s | 10.08 s |
| At least 5 s | 23 | 32 |
| At least 10 s | 1 | 2 |
| NONE | 1269 | 1836 |
| LEFT / RIGHT | 0 / 0 | 0 / 0 |
| JUMP | 1827 | 3987 |
| ROLL | 3317 | 1576 |
| Guard aborts | 0 | 0 |
| Median gameplay interval | 56.59 ms | 51.91 ms |

Configurations differ only in run name and source checkpoint. Actual timing
still varied between sessions, so identical flags do not eliminate timing or
procedural variation. Both evaluations used the existing 15-second episode cap.

Independent episode bootstrap: NumPy default_rng(123), 20,000 resamples of 100
observations per model, percentile 95% intervals. After-minus-before mean:
+0.3037 s, interval [-0.1575, +0.7503] s. The observed improvement is inconclusive;
this interval does not measure independent training-seed variability.

## Preserved local checkpoints

- Before: `ai_data/rl_v2_runs/v612_paused_state_v179_test_20261028_045143/latest.zip`
- After: `ai_data/rl_v2_runs/v612_paused_state_v179_8h_20261028_045909/latest.zip`
- After SHA-256: `1188e943f42b70e4e5ef94d4e7c23662d1ee7191d5e87f4e4aa0921fa638b55c`

Neither checkpoint was overwritten. Git does not include these working models.

## Lane-action investigation

Both deterministic policies selected no lateral action. Average LEFT+RIGHT
probability was 0.24% before and 0.45% after. The policy shifted from ROLL toward
JUMP. Training sampled some lateral actions, which does not demonstrate useful
lane decisions under deterministic evaluation.

Source inspection finds the expected action mapping: 1 -> ArrowLeft,
2 -> ArrowRight. This establishes the intended mapping, not successful Unity
movement. The current observations use image columns rather than verified track
lanes. No reward is tied to lateral actions. The cause of suppressed lateral
probabilities is unresolved; insufficient exploration, observations, learned
value and physical action delivery remain hypotheses. Preserve both models;
verify actual lateral input delivery in a separate controlled local probe before
changing rewards, exploration or using approved human demonstrations. No policy
behavior was changed for this release.
