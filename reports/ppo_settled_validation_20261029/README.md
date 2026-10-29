# Corrected PPO learning: implementation and live gates

29 October 2026. Local Practice only. Implements the five corrections identified
in [the learning correctness audit](../ppo_learning_correctness_20261029/README.md).
These tests validate transition mechanics and safe operation, not improved skill.

## Delivered behavior

1. **Boundary deaths settle before targets.** The collector invokes the strict
   pause/menu gate before final buffer insertion and GAE. If the continuing
   episode is terminal, a guarded, no-key settlement is combined with that
   episode's last actual transition. It is not assigned to a newly sampled
   action. DummyVecEnv's reset produces a distinct episode, which is frozen
   before optimization. An already-terminal last step is not penalized twice.
   The final requested rollout uses the same protocol, so no next rollout is
   needed to emit its known terminal penalty.
2. **Resync is explicit.** The pre-pause endpoint remains the previous rollout's
   bootstrap endpoint. After optimization, four-frame refresh is represented by
   a passive transition from that endpoint to the newly observed state. It has
   a value target and elapsed-time reward, but no sampled action, dispatched
   key, actor loss, entropy loss or KL contribution. If the game dies during
   that interval, its terminal belongs to this passive transition. Shared
   feature weights can still learn through the value loss. Actor-only Adam
   momentum is not applied to passive-only minibatches.
3. **Discounting has time units.** Stored gamma and lambda are applied as
   `gamma ** (verified_active_seconds / 0.05)` and the analogous lambda power.
   GAE and timeout bootstrapping use the same transition duration. The reference
   unit is explicit/configurable; gamma .995 and lambda .95 retain their values.
   This preserves a physical-time discount horizon across cadence changes,
   subject to the external clock's limits and reward-emission discretization.
4. **Reward timing is verified and conservative.** Survival advances on confident
   gameplay captures, not on later death-confirmation waits. Pause UI
   transactions, including uncertain entry/verification, are excluded from the
   reward clock. The per-decision .15-second survival cap is removed ONLY on
   this verified-time path. Genuine long verified gameplay/resync intervals can
   receive their elapsed reward. Death confirmation itself adds no survival
   or milestone time. Coefficients remain .8/second, .15 per .5-second milestone,
   and -1 death. Historical `ProgressRewards` is unchanged for legacy runs.
5. **Completion and accounting are explicit.** A wall-time budget finishes the
   current rollout, processes optimization, then stops without resuming. It can
   exceed the requested wall time by that remaining work. Reports distinguish
   raw environment operations, collected transitions, processed rollouts,
   unique optimizer-used transitions, repeated optimizer sample presentations,
   discarded partial collection, policy selections and dispatched keys.
   Guard stops/interruptions remain fail-closed and report incomplete work.
   Periodic checkpoints are deferred until AFTER the corresponding optimizer
   phase, while the game remains frozen, so saved weight progress is explicit.

Architecture, 12x288x512 image / 557-D auxiliary schema, action mapping, key hold
duration, classifier thresholds and keyboard gates remain unchanged. No BC
updates, movement quotas/cooldowns, recurrence or detector ablation were added.

## Enablement and compatibility

Training with `--pause-ppo-optimizer --resync-rollouts` now enables corrected
learning by default. `--settled-learning` is an explicit equivalent. The
monitored manifest includes it. `--legacy-learning` reproduces historical
training behavior; it is not the corrected path. Abandoned-episode menu
recovery is forbidden with corrected collection.

The feature extractor/actor/critic and checkpoint format remain load-compatible
with ordinary PPO evaluation. Training returns and verified training episode
durations have changed meaning and must not be compared directly to historical
training curves. The five-episode compatibility evaluation used the normal
evaluation clock and return reporting. Compare survival only under matched
evaluation settings; establish a fresh baseline for learning experiments.

The old reward JSONL remains a diagnostic, but cannot replay a merged/passive,
verified-time transition with the historical cap. The authoritative file is
`learning_transitions.jsonl`; batch validation uses `audit_settled_ledger`.
New runs also save coefficients, time units and source hashes in
`learning_protocol.json`. Both reporting and the new audit stream data rather
than retaining an overnight ledger in RAM.

## Offline verification

**204 selected tests passed**; see [offline_validation_pass.json](offline_validation_pass.json). The selected regression
suite includes historical core/runtime/pause/focus/session tests and corrected
collector tests using real installed SB3 with a synthetic environment.

New coverage includes pre-optimization boundary settlement, final-budget death,
post-resume passive death, explicit masked refresh, true death versus timeout
bootstrapping, duration-aware GAE against a hand calculation, reward/discount
cadence subdivision, passive-only Adam behavior, full-rollout deadline stop,
corrected CLI defaults and rejection of abandoned-episode recovery.

Synthetic environment results establish collector ordering and math, not Unity
rendering or collision behavior. The exact final-boundary MENU exception was
exercised offline; it did not occur in the completed live gates.

Earlier combined-suite attempts exposed a pre-existing Windows clock-resolution
issue in the legacy test fixture's artificial 8 ms sleeps. Those waits are now
20 ms, above a clock tick; production capture waits and fail-closed clock checks
were not relaxed. Failed-attempt JSON/text diagnostics remain alongside the
final passing report. Resume tests also verify a saved time unit is inherited
unless explicitly overridden and stale prior-run capture metadata is cleared.

## Live validation

All outputs use NEW directories. The original pilot was independently hashed
afterward and remains
`ca99e521298e783eec6f5b11de11fa21eaa03631842801457e4ea3974643c5bc`.
All three new checkpoint ZIPs pass CRC checks. No prior checkpoint or report
was rewritten.

| Run | Collected | Unique optimizer-used | Passive refreshes | Outcome |
| --- | ---: | ---: | ---: | --- |
| `v612_settled_smoke_20261029_144837` | 256 | 256 | 1 | Initial smoke passed |
| `v612_settled_gate_20261029_train` | 1,024 | 1,024 | 7 | Full corrected ledger audit passed |
| `v612_settled_deadline_20261029_145734` | 128 | 128 | 0 | Six-second soft deadline ended at optimized rollout boundary |

All three report zero discarded partial transitions and zero uncertain-state
guard aborts/hard stops. The deadline intentionally did not complete its 1,024
requested transitions; its report says so, with stop reason
`wall_time_at_optimized_rollout_boundary`. It is not a failure to reach a fixed
step target being disguised as completion.

The initial smoke preceded refinement of exact packed-frame lineage and
dispatch timestamps. The 1,024-transition gate includes those refinements,
ten terminal transitions and a real death during post-resume refresh. The
transition ledger records that exception at rollout 8, index 0:

```text
kind=passive_resync, sampled_action=null, actor_mask=false, command=null
active_seconds=0.0872862
survival=0.06982896, death=-1, environment_reward=-0.93017101
done=true, buffer_reward=environment_reward
```

The wall span includes menu confirmation and a guarded reset, but those waits
do not inflate this verified duration. No policy action was invented to receive
the penalty. Existing paused-frame rejection also ran during the test.

The saved 1,024-transition checkpoint was then loaded and evaluated for **5/5
episodes**, in the same browser (one launch, one reuse, clean closure). There
were no guard stops. Mean survival was 5.03 seconds in this tiny diagnostic;
that is not evidence of a new best model or better learning.

Evidence: [validation summary](validation_summary.json),
[exact manifest](manifest.json), [batch completion](batch/batch.json),
[1,024-transition audit](batch/v612_settled_gate_20261029_train_audits.json).
Full ledgers/checkpoints and startup images remain under their local ignored
`ai_data/rl_v2_runs/` directories.

## Limits and next step

The verified clock is an external, conservative estimate. It may omit the
final interval between the last confidently playable frame and actual death,
and excludes uncertain pause entry even if Unity advanced briefly. It does not
read Unity's internal simulation clock. Capture-to-key races remain finite.
Positive endpoint classification is not infallible. These limits are recorded,
not replaced by a claim of exact engine-time rewards.

The optional faster guard remains optional. Corrected PPO collection is now
the pause-aware default, but the long-run learning quality of this new protocol
is unestablished. The old pilot's critic was trained on the old objective and
needs adaptation; historical reward curves are not a matched control.

Next use a monitored fixed-budget experiment under these corrected semantics,
with a fresh matched evaluation baseline. Do not mix additional BC, recurrence,
detector changes or new reward coefficients into that first comparison. No
multi-hour training was started as part of this implementation.
