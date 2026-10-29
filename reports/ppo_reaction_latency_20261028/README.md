# PPO reaction latency: actual policy frame to key dispatch

## Result

Two guarded local Practice audits completed, 30 episodes each, using the
preserved small BC pilot. Baseline detection every three decisions was compared
with detection every decision. This was the sole runtime-setting difference
apart from the unique output name. No training or weight saving occurred.
Both runs had zero uncertain-state aborts and no guard hard stop.

| Measurement | Detect every 3 | Detect every 1 |
| --- | ---: | ---: |
| Decisions recorded | 2,804 | 2,111 |
| Movement commands recorded | 2,037 | 1,538 |
| Median observation interval | 51.85 ms | 61.14 ms |
| Mean observation interval | 53.04 ms | 55.90 ms |
| Observation interval p95 | 65.38 ms | 67.33 ms |
| Policy inference median | 3.88 ms | 5.41 ms |
| Policy RGB age at prediction median | 3.91 ms | 4.49 ms |
| Policy RGB age at command median | 28.23 ms | 31.50 ms |
| Policy RGB age at command p95 | 35.16 ms | 36.35 ms |
| Policy RGB age at command p99 | 36.93 ms | 41.01 ms |
| Guard-to-inherited-step median | 20.59 ms | 21.02 ms |
| Keypress round trip median | 17.54 ms | 17.89 ms |
| Policy RGB age at press return median | 48.00 ms | 49.71 ms |
| Guard RGB age at command median | 6.99 ms | 8.65 ms |
| YOLO source age at prediction median | 114.05 ms | 65.78 ms |
| YOLO source age at prediction p95 | 190.83 ms | 71.62 ms |

All metrics from inference through YOLO age above use actual movement-command
rows. Observation intervals use the environment's gameplay-only report. Rounded
medians of separate stages should not be summed into an exact median total.

The normal pipeline supplies recent RGB to the policy and inference itself is
fast. A new guard capture plus classification is the largest measured stage
between prediction and command. This work verifies gameplay and must be
preserved. More frequent YOLO yields much newer context but also worsened the
median action/observation interval and RGB-to-command age in this test. This is
an observed scheduling/load tradeoff, not a free speed improvement. The normal
configuration remains unchanged at detect-every 3.

## What was measured

The audit wraps the unchanged evaluation process in memory. It records actual
MSS grab start/end and IDs, the capture used by observation packing, policy
inference, guard captures, inherited controller entry and Playwright
keyboard.press start/end. It adds no model forward passes, browser requests,
sleeps, action changes or thresholds inside the loop. Lightweight Python
instrumentation has some unmeasured overhead.

```text
Policy source desktop grab
  -> observation packing
  -> policy prediction
  -> newer guard grab + screen classification
  -> Playwright press request
  -> requested key hold and API return
  -> post-action desktop grab / next policy observation
```

The immutable policy-source record prevents guard captures from silently
replacing the source timestamp. Both timestamps are retained. Capture ID and
timestamp ordering were independently checked for all 3,575 movement commands:
policy source precedes guard source, which precedes post-action capture.
Three offline tests passed, including a regression where a newer guard image
would otherwise understate policy-image age.

## Bounds and limitations

RGB age is measured from MSS grab completion, not the game's rendering or
physical display presentation time. Source grabs themselves take about 12-13 ms.
These measurements cannot establish exactly when Unity applies an input or when
its response becomes visible. Keypress round trip includes the requested 5 ms
hold and protocol/scheduling overhead; it is not the keydown-to-Unity delay.
No browser-event clock calibration or character-motion response detector was
added. The pilot selects JUMP/ROLL/NONE, so this is not an independent test of
Unity's lateral movement response or valid double-tap timing.

A newly visible hazard can also appear just after a policy-source capture and
wait until another decision. Thus a 28 ms image age at dispatch does not mean
every new visual event receives a response within 28 ms. Active gameplay is not
slowed down by optimizer pauses. The 15-second episode cap does not validate
sustained late-game speed.

Maximum RGB-to-command ages were 107.87 ms and 101.33 ms, both at the first
recorded movement in episode 1. After the first 32 decisions, maxima were
44.02 ms and 57.01 ms; full and startup-inclusive distributions remain saved.
This distinction is descriptive, not an excuse to omit startup latency.

YOLO age is measured from the detector's recorded source timestamp, assigned by
the existing pipeline after capture processing. It is therefore approximate
relative to true source grab time. The metric tracks the detector snapshot
actually present in the policy input, not an unrelated newer worker result.
Missing snapshots are excluded from age quantiles and counted in raw records.

These were sequential sessions, not randomized identical maps. Mean survival
was 5.32 s versus 4.32 s in only 30 episodes. Those survival figures must not be
used to promote a setting or replace the pilot's 100-episode 4.83 s reference.
Scene complexity and action mix can affect runtime distributions as well.

## Recommendation

Latency was worth checking: earlier guard-frame timing could make the real
policy image appear newer than it was. The measured baseline does not show a
severely stalled policy, nor prove latency adequate at every game speed.

Keep fast repeated inputs available. The observed harmful repeated lateral
moves do not justify a global lane-change cooldown that blocks valid patterns.
A useful next performance prototype would align safety classification and policy
inference with the same recent frame, while retaining all gameplay checks and
measuring actual browser/Unity response. That requires careful environment-loop
and transition-alignment tests; it was not implemented in this audit. Do not
remove guard work or shorten key holds blindly for a lower stopwatch reading.

For learning, the new BC model's initially worse score does not rule out a better
starting point for subsequent PPO. A controlled next learning experiment is
short equal-budget PPO continuations from the pilot and the newer BC model,
using the same preserved timing/reward settings, followed by matched evaluations.
This directly tests the user's useful-exploration hypothesis.

Integrating a small, weighted demonstration-imitation term into PPO updates is
also a plausible later experiment. Keep demonstration loss separate from PPO's
on-policy rollout objective, retain useful NONE examples, and perform extra
optimization during verified optimizer pauses. Tune/decay the demo influence
rather than requiring permanent exact imitation. Once PPO updates its feature
extractor, the old frozen-feature BC caches cannot simply be reused: demonstration
observations must be encoded with the current extractor, or an explicitly frozen
architecture must be tested separately. No demo loss or PPO code was changed.

Demonstration-augmented policy-gradient research supports this direction in
other tasks, not a guarantee for this game:
[Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and
Demonstrations](https://arxiv.org/html/1709.10087v2), especially IV-C2.

## Artifacts

Full raw per-step timestamps remain in each run's reaction_latency_steps.jsonl
under ai_data/rl_v2_runs. Summary audits, original evaluation reports and
configurations are copied here. comparison.json includes the exact configuration
difference, additional distributions and independent capture-link checks.

Pilot SHA-256 remained
ca99e521298e783eec6f5b11de11fa21eaa03631842801457e4ea3974643c5bc.
Normal training/evaluation source files, model weights and recordings were not
modified. New tooling: tools/audit_ppo_reaction_latency.py and
 tests/test_ppo_reaction_latency.py. No Git commit or push was performed.
