# Overnight readiness — 31 October

## Completed checks

- Applied detector-off/input-contract/independent-seed/evaluation-clock/finalization
  audit patch. Supports up to 2,880 minutes; enough requested steps are also needed.
- Earlier patched smoke: 1,536 optimized transitions, three passing learning audits,
  no guard stops or finalization errors. Source checkpoint unchanged.
- Matched deterministic evaluations (100 each), identical flags except model/output:
  source mean 4.44452 s, final mean 4.37650 s. Difference -0.06802 s;
  independent episode-bootstrap 95% interval [-0.60268, 0.45671] s, 20,000 resamples,
  seed 731. This does not establish equivalence or improvement; sequential episodes
  need not be independent. Both completed without guard stops and selected no lateral actions.
- Action diagnostic `live_action_checks_20261031_202347`: ten trials completed.
  Both LEFT and RIGHT trials visibly move toward the requested side; both JUMPs
  are airborne. ROLL lowers the character but is partly occluded. LEFT_TWICE does
  not establish two distinct shifts from the centre lane. Browser events match.
- Both template-only terminals in that diagnostic (episodes 2 and 6) visibly end
  on the menu. This is sampled evidence, not proof against all false deaths.
- Startup recording shows forward camera alignment around 1.0–1.25 s after reset
  completes. HUD progresses from 0 to 2, 5 and 13 m. This confirms visible progress,
  not an independent measurement of Unity simulation time or input consumption.
- Overall-mean trajectory added alongside rolling-100 post-training plot, with
  backward compatibility. 60 focused offline tests passed in this readiness pass.

## Remaining limits and decisions

Final short stability check `overnight_readiness_20261031_01`: COMPLETE, 11 updates,
4,224 optimized transitions in 129.16 seconds (32.70/s); all three audits passed.
Periodic checkpoint at 1,920 transitions and final checkpoint at 4,224 saved.
No guard stops, discarded partial transitions or finalization errors. Source
unchanged; overall-mean plot saved. This run did not enable detailed reaction
profiling; the proposed user-run 15-minute check enables it.

No BC or exploration intervention for this continuation. Detector remains off.
BC recorder validation is required before future recording/BC work, not this run.
Physical-browser crop parity remains untested and is not needed to continue with
the same headless canvas path. Independent simulation-clock fidelity under load
remains unverified. The startup classifier can accept the opening animation;
we have not established when inputs become effective during it. Training retains
existing startup behavior. Do not count a diagnostic wait as a training fix.

Long-run reliability is empirical: short passes cannot certify 8–48 hours. Preserve
periodic atomic checkpoints, keep safety stops enabled, and inspect outcomes.
More training is not guaranteed to recover lateral exploration or improve survival.
Use the completed 15-minute readiness checkpoint as the overnight continuation
source; preserve the two-hour reference and separate scaling branches. This is
a latest-weight continuation, not a model promotion supported by evaluation.
No architecture/reward/threshold
or learning-rate changes are introduced for the overnight run.

## Final readiness and scaling results

`v612_readiness_15m_20261031_203221`: COMPLETE, 83 updates, 31,872 optimized
transitions, 911.97 seconds, 34.95 transitions/s. All three audits passed; no
guard stops, discarded partial transitions or finalization errors. Periodic and
final saves exist; latest.zip passed ZIP CRC verification. Source unchanged.
319 episodes, mean 4.2755 s, recent100 4.2620 s, max 13.2653 s. These are training
statistics, not a frozen-policy improvement claim. Last 1,000 command samples per
worker: policy request-to-command median 31.95–32.34 ms, p95 37.71–38.73 ms.
All 90 saved plot points include the cumulative overall mean.

`training_scaling_20261031_204916`, approximately three minutes per configuration:

| Workers | Optimized transitions/s | Worker latency median ms | Worker latency p95 ms |
| --- | ---: | ---: | ---: |
| 3 | 32.68 | 32.05–32.32 | 38.64–39.03 |
| 4 | 36.51 | 33.88–34.00 | 68.03–71.50 |
| 5 | 35.52 | 37.62–39.01 | 80.95–83.26 |
| 6 | 30.66 | 71.52–73.75 | 88.65–89.57 |

All stages completed and all audits passed, with no finalization errors or guard
stops. Latencies are each worker's retained window, not pooled full-run quantiles
or Unity response measurements. All started from the same two-hour source.
The fixed 128-step per-worker rollout changes aggregate update size as workers
increase. Sequential short stages are not repeated controlled learning trials.
Three workers remain preferred for reaction speed; four gains about 12% throughput
at substantially worse tail latency. Six is functional but slower than three here.

Checklist: smoke/saves/latency complete; matched evaluation complete; lateral and
jump input effects observed; template-only terminal samples reviewed. Independent
Unity-clock validation and long-duration reliability remain open. Detector-off
recorder validation is deferred until BC work. Do not describe this as all possible
live checks passing. The 12-hour continuation is the next reliability experiment.
