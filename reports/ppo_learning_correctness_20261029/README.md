# PPO learning correctness audit — 29 October 2026

## Decision

Do not use the next long run to tune around the current rollout-boundary
handling. The normal capture/action path has supporting evidence, but the
pause/resync/menu callbacks introduce transition and credit-assignment gaps that
reward replay alone does not detect. Address collector integration before a
learning-rate comparison. This audit does not identify the cause of yesterday's
survival regression or prove that the policy cannot learn.

No runtime, reward, checkpoint, observation schema, guard or PPO hyperparameter
was changed. No browser, live gameplay or working model training was launched.
The only new files are this report and its offline diagnostic script/results.
Tiny synthetic PPO models run on CPU with vector observations; their observations
and gap lengths are artificial, not measurements of Unity dynamics.

## Evidence and reproducibility

- Inspected the installed SB3 **2.9.0** collector, PPO learn loop and DummyVecEnv
  step/reset implementation, rather than assuming callback ordering.
- Ran the actual repository `build_callback` with actual SB3 collection/buffering
  in seven synthetic scenarios: normal resync, boundary death before another
  rollout, boundary death at training completion, post-resume death, ordinary
  death, timeout and time-budget stop. Instrumented buffer additions and return
  computation without changing trainer code. Assertions check ordering,
  resets/episode-start flags, deferred penalties, resync inputs, time-limit
  bootstrapping and callback-stop omission.
- Analyzed **485,133 recorded reward rows** across the eight-hour run and three
  controlled training arms.
- Analyzed **6,780 recorded decision traces / 4,973 movement commands** from
  four latency evaluations, including the optional same-observation guard.
- Runtime/reward/helper source hashes were unchanged across the synthetic audit.

Authoritative numerical output: [results_final.json](results_final.json).
`results.json` and `results_v2.json` preserve earlier audit passes with fewer
checks. None is an original training/evaluation result being rewritten.

To reproduce without overwriting anything, run from the project root with the
existing Python environment and choose a NEW output filename:

```powershell
& $py -B -s reports/ppo_learning_correctness_20261029/audit_learning.py `
  --output reports/ppo_learning_correctness_20261029/results_REPEAT.json
```

The script refuses an existing output. It uses temporary files for synthetic
callback logs and writes only its new JSON result. It needs the local ignored
reward/latency artifacts and installed dependencies. It never loads the working
PPO ZIPs or instantiates the live browser environment.

## 1. Rollout-boundary death: penalty is late, or absent at completion

Confirmed ordering in installed SB3:

```text
predict action -> env.step -> callback.on_step -> buffer.add
finish rollout -> value bootstrap -> compute returns/advantages
callback.on_rollout_end -> verify/freeze or discover MENU
PPO.train
next callback.on_rollout_start -> resume/resync or re-confirm MENU
predict next action -> guarded env.step -> deferred terminal reward
```

The boundary-menu gate is safe for keyboard input, but discovery happens AFTER
the preceding rollout's targets were computed. For an ongoing episode whose
cached last frame looked playable, those targets bootstrap from a nonterminal
value. The preceding rollout is then optimized before the known death penalty
is recorded. The callback does not recompute those targets.

In the next rollout, a new action and log probability are sampled from the
cached observation. The guard suppresses movement, then returns the death
penalty on that new transition. It is an actual recorded death, not a missing
arithmetic penalty, but the selected action did not execute and the death was
already independently known before that action was sampled. The new negative
advantage can train that selected action; GAE does not retroactively traverse
the previously trained rollout. Later value learning can propagate information,
so this does not mean death can never affect preceding behavior.

At final training completion there is no next rollout, so the pending terminal
reward is never emitted. Arm A demonstrates the real case: its last reward row
is `done=false`, duration 5.08, reward +0.048; the final log subsequently confirms
MENU and defers the penalty, then saves the checkpoint. No terminal row follows.
The synthetic final-boundary scenario reproduces the pending flag, absence of
the penalty and nonterminal final bootstrap.

Do not blindly subtract one from a previous reward: a boundary can also occur
after a genuine terminal and DummyVecEnv auto-reset. In that case the current
episode is new. A correction must identify the exact episode/transition and
finalize it before return computation, preserving genuine reset boundaries.

## 2. Passive resync crosses an unrepresented state/time interval

`refresh_after_ppo_optimizer` gathers four confident images while the game is
running, then `replace_cached_observation` replaces `_last_obs` only. No action,
reward, terminal or buffer transition describes that refresh interval. The prior
rollout was bootstrapped from the older pre-refresh endpoint; the next rollout
starts from the refreshed endpoint without an episode boundary.

Reward duration still starts at the previous emitted step. Consequently the
next action's reward includes pre-action time from that interval, subject to the
unchanged 0.15-second cap. This is separate from verified optimizer pause time,
which is compensated. Pause entry/exit and verification uncertainty also affect
what remains in the measured duration; these files do not independently measure
Unity's simulation clock.

The synthetic trace makes the discontinuity explicit: old endpoint code 2 is
used for bootstrapping; passive resync changes the code to 102; the next action
uses 102 and receives time credit including an artificial 0.25-second passive
gap. This is an ordering demonstration, not a claim the real gap is 0.25 seconds.

Recorded first-after-optimizer duration increments have medians of **0.47 s**
overnight and **0.50 s** in A/B/C, despite ordinary increments around 0.04–0.05 s.
These categories include boundary-menu handoffs as well as successful resumes.
All 63 such rows in both B and C exceed the cap. The gap is systematic, not
evidence that optimizer time itself leaked through the pause compensation.

## 3. Normal input/observation path: checks passed, with a measurement limit

Across four saved latency runs:

- Every one of 4,973 recorded movement commands matched the sampled action's
  Arrow key; prediction/command/post-action-capture ordering checks passed.
- **6,700** comparable consecutive, same-episode observation links matched the
  preceding post-action capture exactly. No cross-frame mismatch was found.
- **36 nonzero selected actions** had no command; every one was a death terminal,
  not an unexplained suppression during recorded ongoing gameplay. This is
  appropriate keyboard safety, but it confirms selected-action counts are not
  dispatch counts. Their transition rewards still attach to the selected action.
- Remaining no-command choices were NONE, as expected.

These are recorded evaluation traces, not full overnight PPO-buffer traces.
Playwright command request/return is not independently timed browser receipt
or Unity applying/responding to a key. Lateral command paths were not exercised
by these four zero-lateral policies. No claim of physical lateral registration
or death-cause attribution follows from these checks.

## 4. Reward cap: arithmetic matches, interpretation needs care

Reward remains: 0.8 times min(duration increment, 0.15), plus 0.15 for each new
0.5-second milestone on a non-death step, minus 1 for death. A terminal cancels
milestones newly crossed on that terminal step. Reset clears the reward state.

| Run | Rows | Exact increments >0.15 s | First-after-optimizer median | Sum of observed increment above cap |
| --- | ---: | ---: | ---: | ---: |
| Eight-hour v1.7.9 | 386,829 | 8,354 (2.160%) | 0.47 s | 2,397.72 s |
| A: rollout 128 | 32,768 | 566 (1.727%) | 0.50 s | 200.24 s |
| B: rollout 512 | 32,768 | 346 (1.056%) | 0.50 s | 128.07 s |
| C: rollout 512 + KL | 32,768 | 385 (1.175%) | 0.50 s | 133.66 s |

The existing auditor allows 0.012 seconds tolerance because logged durations
are rounded to 0.01 seconds, so its counts are lower (8,345 / 555 / 341 / 381).
The exact counts here exclude floating-point epsilon around 0.15 seconds.

Terminal rows account for about **64% / 59% / 85% / 84%** of observed excess
duration respectively. Ordinary terminal duration is measured after confirmation
work in the inherited controller, so it includes some post-death recognition
latency. The deferred-menu path has a separate first-menu anchor. Therefore the
excess cannot honestly be called entirely lost valid survival credit. Removing
the cap could reward confirmation latency; do not remove it as an audit fix.

Nonterminal excess remains meaningful: 872.28 s overnight, 82.43 s in A, 18.81 s
in B and 20.97 s in C, spread across their recorded intervals. A larger rollout
reduces the frequency of boundary gaps and changes both timing overhead and the
distribution of capped increments. This is a learning-signal confound in addition
to optimizer statistics. Category totals overlap; do not sum first-after-reset,
first-after-optimizer and terminal groups as disjoint sets.

## 5. Faster cadence changes the physical discount horizon

Gamma is **0.995 per transition**, not per second; GAE lambda is 0.95. Using the
measured median intervals as constant-interval illustrations:

| Illustration | Decisions/s | Discount on reward five seconds away | Gamma half-life | GAE (gamma*lambda) half-life |
| --- | ---: | ---: | ---: | ---: |
| Normal guard, 51.92 ms | 19.26 | 0.617 | 7.18 s | 0.639 s |
| Faster guard, 40.19 ms | 24.88 | 0.536 | 5.56 s | 0.495 s |

Faster decisions shorten the discount horizon in physical time. Reward density
is also divided across more decisions, while milestone and death scales retain
their absolute values. Keypress overhead makes actual cadence action-dependent;
one median cannot reconstruct a trajectory's discount schedule. The GAE figure
describes trace weighting, not a hard limit on how far a critic can learn.

This is a changed effective learning objective when timing changes, not a proven
implementation defect or proof that discounting caused regression. A possible
later experiment is cadence-calibrated fixed gamma/lambda or duration-aware
discounting, but either changes learning semantics and needs separate approval,
buffer/GAE design and validation. Neither was applied here.

## 6. Time limits and wall-time stop have different meanings

The installed DummyVecEnv preserves the true terminal observation on a timeout
and returns a fresh reset observation. SB3 adds gamma*V(terminal_observation)
to the stored timeout reward before buffer insertion. The environment audit
runs before that addition, which is legitimate: raw reward and PPO-buffer reward
need not match on timeouts. The synthetic test observes raw +0.04 becoming
approximately +0.26261 in the buffer. Death does not receive this bootstrap;
the reset state's episode-start flag is true. Existing controller tests cover
300-second truncation, reset reload and true death winning simultaneous timeout.

By contrast, `callback.on_step` returning false for the wall-time training budget
returns from SB3 collection BEFORE that step is added. The partially collected
rollout is not optimized. The synthetic stop records one environment audit row
and zero buffer additions. In the eight-hour run, **13 trailing steps** follow
the last full 128-step rollout; those partial-rollout experiences did not produce
another PPO update. `num_timesteps` and audit counts measure interaction, not
proof that every row trained the model. All three fixed-budget arms end on full
rollouts and do not have that incomplete-tail issue.

## Recommended correction order (no fixes applied)

1. Design episode-aware collector integration so boundary/death resolution and
   the observation endpoint are settled before return/advantage computation and
   optimization. Define how active resync time belongs to the preceding action
   or an explicit transition, without inventing a sampled action/log probability
   or silently discarding the death. Preserve the pause and keyboard gates.
2. Add a transition ledger that distinguishes sampled action, dispatched key or
   suppression reason, exact observation identity, pre/post times, episode ID,
   raw reward, buffer reward including timeout bootstrap, terminal flags, and
   whether a transition was collected and subsequently optimized. It must not
   insert extra model forwards/captures into the live path.
3. Make planned wall-time completion end at a validated rollout boundary or
   explicitly report its discarded partial rollout. Do not rewrite old reports.
4. Revalidate pause, actual death, post-resume death, reset, timeout and final
   boundary ordering offline, then run a short live gate before long training.
5. Only then compare update size/learning rate. Keep cap/gamma/lambda unchanged
   during the integration fix; study their cadence dependence separately.

## Limits

These are code-ordering proofs, existing-log measurements and synthetic
reproductions. They do not measure Unity internal time, establish which action
caused each death, or isolate the cause of worse survival. No original result
or model was edited. The experiment remains paused pending review of the
collector/transition correction, not blocked on collecting more demonstrations.
