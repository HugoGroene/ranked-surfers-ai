# Ranked Surfers AI: project and learning-pipeline audit

Audit date: 31 October 2026. Reviewed commit: `9ba46167d13c2aa7e6a353d28f7636795255643d`, branch `headless-parallel`.

This is a source and existing-artifact audit. No training code, settings, checkpoints, recordings or experiment results were changed. No new gameplay, evaluation, training or test suite was started. This report is the only new artifact created for the audit.

The two-hour experiment `v612_optimized3_detectoroff_2h_20261031_142111` was RUNNING when inspected. Its configuration was read, but its partial results are not treated as a completed experiment. The findings below do not require interrupting it.

## Assessment

The project now has a substantially stronger collection and optimization pipeline than the original overnight trainer. The code explicitly handles rollout-boundary deaths before return calculation, represents post-resume time as passive transitions, applies elapsed-time discounting, prevents trajectories from different workers being joined, and saves validated checkpoints atomically. The recent completed diagnostics support those engineering improvements.

The remaining concern is whether the experience being collected consistently represents the intended game and actions, and whether subsequent evaluation can measure improvement fairly. Faster capture and passing arithmetic audits cannot answer those questions by themselves.

The highest-priority findings are:

| Priority | Finding | Evidence status |
| --- | --- | --- |
| High | Pre-action and post-action paths can confirm death using different evidence rules | Confirmed source inconsistency; false deaths not established |
| High | Checkpoint input semantics can change without an observation-shape error | Confirmed missing compatibility enforcement |
| High | Evaluation survival/reward reporting differs from settled training, and incomplete evaluations can return normally | Confirmed source behavior |
| High | A fresh safety recapture can dispatch an action sampled from an older observation | Confirmed control path; practical impact needs measurement |
| High | Current evidence verifies command dispatch, not reliable Unity execution of each action | Confirmed measurement gap; missed actions not established |
| Medium | All workers load the same saved random seed without a worker-specific reseed | Confirmed setup; resulting trajectory correlation not measured |
| Medium | Final cleanup can fail before the final checkpoint/report is written | Confirmed failure path; periodic atomic saves mitigate it |
| Medium | BC tools do not yet share a complete framing/detector contract with current training | Confirmed compatibility gaps; relevant before revisiting BC |
| Medium | Learning audits validate important bookkeeping, but not all claims suggested by their name | Confirmed scope limitation |

I found no source evidence that a recurrent network, larger network, additional action rewards or another browser rewrite is presently necessary. None of these findings establishes that the current model cannot learn. Several provide more specific hypotheses to test than another broad hyperparameter comparison.

## 1. Scope and current system

The important paths reviewed were:

- Environment, observations, pause/resume, action dispatch, terminal handling and evaluation: `train_rs_v612_ppo.py`, `train_rs_v3_ppo.py`, `train_rs_v2_verify.py`, `train_rs_rl_early_death.py`, `rs_v2_timing_probe_ab.py`, relevant V5 capture code and `rs_v612_gameplay_guard.py`.
- Learning and parallel collection: `tools/ppo_settled_learning.py`, `tools/ppo_parallel_learning.py`, `tools/train_ppo_parallel.py`, shared rollout/frame transports, prepared-observation and frame-state caches.
- Rendering and framing: headless adapter, Electron Python/JavaScript bridge, `tools/ppo_framing.py`, background geometry handling, local Unity page wrapper and render-budget helper.
- Observation features and BC: `rs_v6_ppo_features.py`, `rs_v61_features.py`, recorder preprocessing, `rs_v612_demo_bc.py`, `tools/train_demo_bc_cached.py`.
- Reporting, persistence and validation: atomic checkpoint helper, survival aggregation, reaction metrics, relevant regression-test sources, saved validation summaries and recent completed experiment reports.

This is not an exhaustive verification of every historical patch installer, downloaded/minified web asset or compiled Unity binary. The original Unity C# gameplay/input implementation was not available for this audit. Important conclusions therefore distinguish Python/browser behavior from actual simulation behavior.

### Active configuration

The current two-hour run uses three headless Electron hosts, separate synchronized policy replicas, shared-memory screenshot and rollout transfer, dense observations, detector OFF, OpenCV threads 1 and exact-frame classifier reuse. The dashboard is disabled. The final plot is enabled.

The saved source model and run configuration confirm:

| Setting | Value |
| --- | --- |
| Policy input | Four RGB frames: `12 x 288 x 512`, plus 557 auxiliary values |
| Framing | `canvas_v1`, whole 1920 x 945 game canvas |
| Actions | NONE, LEFT, RIGHT, JUMP, ROLL |
| Rollout | 128 transitions per worker; 384 per shared update |
| Learning rate / epochs / minibatch | 0.0003 / 5 / 32 |
| Gamma / GAE lambda / reference duration | 0.995 / 0.95 / 0.05 seconds |
| Entropy coefficient / target KL | 0.002 / none |
| Saved seed | 123 |
| Episode limit | 300 seconds |
| Periodic checkpoint interval | 10 minutes, checked at completed shared updates |

The input checkpoint is `reports/v612_speed3_preview30_5m_20261031_140835/latest.zip`, SHA-256 `086c11c2570d3048830368e907b114e2327b34c44233401e20c43c546fba3024`. That preceding preview experiment was interrupted; its checkpoint is a continuation artifact, not evidence that the preview experiment completed successfully.

The 15-second ceiling is absent from this active training configuration. A 300-second limit permits runs up to five minutes, although an episode reaching that limit is a truncation, not proof that it would have died then.

## 2. What is already sound and should be retained

### Learning boundaries and sample ownership

`SettledPPO.collect_rollouts` settles the final transition before computing returns. Post-resume progression is represented by a passive transition rather than silently replacing the next observation. Passive rows are excluded from the actor loss, while remaining available to the critic. Time-limit transitions bootstrap; deaths do not. See [settled collector](../tools/ppo_settled_learning.py#L126) and its [regression tests](../tests/test_ppo_settled_learning.py).

Per-worker returns/advantages are computed before merge. The merger validates worker identity, policy version, policy hash, shapes, masks and numerical buffer fields, and does not run GAE across unrelated games. Shared rollout slots have explicit publish/commit ownership. See [parallel learning](../tools/ppo_parallel_learning.py#L50) and [rollout transport](../tools/ppo_rollout_transport.py).

This is one learner receiving several trajectories, not independently trained models whose checkpoints are later combined.

### Capture and safety

Electron demands a future paint event rather than reading an image FIFO. Frame serials, geometry and format are checked; stale replies are discarded under the existing bound. Shared-memory pixels are copied into owned storage and their header is checked before and after reading. Removing that copy without an ownership replacement could introduce corrupted observations.

The prepared gameplay check is tied to a specific image and timestamp. Exact-frame classifier caching does not manufacture independent confirmation votes. Pause resynchronization checks both overlay visibility and rendered pause evidence. The bounded MENU-tail recovery can only confirm a terminal or abort; it cannot authorize movement.

References: [Electron capture](../tools/ppo_electron_capture.py#L26), [renderer](../tools/electron_worker/main.cjs#L55), [shared frame validation](../tools/ppo_shared_frame.py#L29), [prepared reading](../tools/ppo_prepared_guard.py), [MENU-tail recovery](../tools/ppo_stream_menu_tail.py).

### Persistence and existing verification

The atomic checkpoint helper saves a candidate, checks ZIP integrity, flushes it, preserves `latest.prev.zip` and replaces `latest.zip` only after validation. Unique periodic snapshots are also retained. This is a useful protection against incomplete saves; it cannot preserve an update that was never saved before a process/OS failure.

The existing [publication validation summary](github_publish_validation_20261031_01/summary.json) records 322 passing offline tests and JavaScript syntax checks. These were not rerun during this audit. The tests include meaningful synthetic collector/optimizer cases, rather than only CLI checks. They still do not establish Unity input acceptance or long-run policy quality.

## 3. High-priority findings

### 3.1 Death confirmation is not one consistent protocol

**Confirmed:** `PPOV61Env._step_browser` checks gameplay before a command using the newer screen-state guard. However, the inherited `LocalPracticeEnv.step` subsequently calls `_menu_votes` / `_wait_menu_confirmation`, which can declare death from template scores alone. `PPOV61Env.step` retains the inherited `died` result; it only forces an additional terminal when its own guard confirmed MENU.

References: [pre-action guard](../train_rs_v612_ppo.py#L904), [inherited vote/confirmation rules](../train_rs_rl_early_death.py#L369), [inherited step](../train_rs_rl_early_death.py#L404), [V2 delegation](../train_rs_v2_verify.py#L423), [final reward/terminal handling](../train_rs_v612_ppo.py#L1076).

**Why it matters:** an incorrectly accepted template-only death could create a wrong negative reward, end the trajectory and initiate reset handling. Conversely, replacing it carelessly with a stricter path could add substantial confirmation latency. A passing reward replay would reproduce the supplied terminal flag; it would not prove that flag was correct.

**Not established:** that these template-only terminations are false, frequent, or responsible for short survival.

**Next action:** record a terminal-evidence source for every death and retain a small bounded set of associated frames. Review template-only terminals against the existing ML/visual rules. Then, if warranted, consolidate terminal acceptance into one tested protocol without weakening thresholds or adding extra captures to the ordinary gameplay path.

### 3.2 Input shape checks do not protect checkpoint meaning

**Confirmed:** the parallel trainer assigns the requested `framing_profile` and `detector_mode` onto the loaded model. It does not first enforce agreement with the saved values. The standalone evaluator builds an environment from CLI options and loads the model into it; its defaults are legacy framing and detector ON. These can have exactly the same tensor dimensions as canvas framing and detector OFF.

References: [parallel metadata assignment](../tools/train_ppo_parallel.py#L365), [evaluation load](../train_rs_v612_ppo.py#L1957), [CLI defaults](../train_rs_v612_ppo.py#L2034), [framing implementation](../tools/ppo_framing.py).

**Why it matters:** a valid checkpoint can be evaluated or continued with different visual/auxiliary semantics without a load error. The active run explicitly uses canvas/off, matching its source metadata; this finding does not indicate an accidental mismatch in that run.

**Next action:** establish one checkpoint observation contract covering framing, resize/letterbox method, image channel/stack order, feature version, detector availability encoding and relevant model identities. Default to the saved contract or fail on mismatch. Intentional migration should be explicit and recorded, including the old and new contract.

### 3.3 Evaluation does not currently report the same survival measure

**Confirmed:** settled training aggregates `reward_duration_s`, based on the last accepted gameplay observation and excluded optimizer pauses. Evaluation appends `info['duration']`, the inherited episode wall clock. Evaluation also cannot enable the settled training mode through its current CLI checks, so its reported returns use the other reward path.

References: [training episode statistic](../train_rs_v612_ppo.py#L1733), [evaluation duration](../train_rs_v612_ppo.py#L1999), [inherited episode clock](../train_rs_rl_early_death.py#L450), [mode restrictions](../train_rs_v612_ppo.py#L2131).

Death confirmation can increase the latter duration without additional survival. This is separate from stochastic training versus deterministic evaluation, and separate from different capture speeds. No universal numerical correction can safely be inferred from the code alone.

**Also confirmed:** evaluation catches a UI safety stop or keyboard interruption, writes partial statistics and returns normally. It does not set a clear evaluation completion/status field in that path. A caller using only process exit status can mistake a partial evaluation for a successful batch. Episode count can expose incompleteness, but callers must explicitly check it.

**Next action:** report both clocks with precise names, use one designated survival metric for comparisons, and add explicit requested/completed counts, completion status and stop reason. Preserve historical metrics for reproduction. Compare models under the same crop, detector mode, control backend and decision protocol before making a promotion decision.

### 3.4 Safety-frame freshness is not policy-decision freshness

**Confirmed:** an action is sampled before `env.step`. If the prepared gameplay reading expires, `_step_browser(action)` takes another capture or several confirmation captures. When gameplay becomes acceptable, it can dispatch the same previously sampled action. Its late prepared-reading fallback also recurses with the same action.

References: [sample before env.step](../tools/ppo_settled_learning.py#L147), [guard recapture/recovery](../train_rs_v612_ppo.py#L919), [late fallback and dispatch](../train_rs_v612_ppo.py#L1048).

**Why it matters:** “the latest screen is safe for movement” and “this action was chosen from a sufficiently recent obstacle layout” are different claims. Existing reaction telemetry correctly keeps policy-source and guard-source timing separate. Typical recent timings are good, but the source does not enforce an equivalent hard maximum on the age of the policy decision across all recovery paths.

**Next action:** inspect the current run's policy-source latency tails, particularly after guard recovery and startup. If problematic, use an explicit stale-decision protocol. Do not simply rerun the policy inside the guard: PPO must store the observation, action, value and log probability that actually produced the dispatched command, with intervening time represented correctly.

### 3.5 Actual action effectiveness remains unverified

**Confirmed measurement gap:** the code timestamps Playwright command issue/round trip. It does not directly verify Unity's consumption of LEFT/RIGHT/JUMP/ROLL or the resulting movement. The action context records dispatched commands, not visually confirmed lane changes. Current workers request 5 ms taps; Electron's offscreen frame-rate setting is 60. Those numbers alone neither prove nor disprove reliable action recognition.

References: [input tap](../rs_v2_timing_probe_ab.py#L43), [action context update](../train_rs_v612_ppo.py#L1067), [reaction metric scope](../tools/ppo_reaction_metrics.py), [Electron setup](../tools/electron_worker/main.cjs#L41).

The completed 15-minute detector-off run selected LEFT/RIGHT only 60 times out of 29,947 actor transitions, approximately 0.20%. This is a much more specific behavioral concern than the overall mean fluctuating. It does not prove the input system is broken: the policy may simply assign those actions very low probability.

**Next action:** after the current run, use existing evidence first; if insufficient, collect one short targeted diagnostic linking policy pixels, action probabilities, issued keys and visible movement/collision. Check all actions on the current Electron path. Distinguish an inappropriate choice, a missed/ineffective input, a late input and a misclassified terminal. Do not reward lateral actions merely to make their count increase.

## 4. Additional confirmed gaps and bounded risks

### 4.1 Worker random streams are not explicitly independent

Every worker loads the same checkpoint without a worker-specific seed override or subsequent reseed. The source checkpoint saves seed 123. Installed SB3 source inspection confirmed that loading restores saved attributes and invokes setup, which calls `set_random_seed(self.seed)`.

Reference: [worker model loading](../tools/train_ppo_parallel.py#L77). The environment's Gym reset seed also does not by itself establish a seeded Unity track generator.

**Consequence:** worker sampling streams start from the same seed. Different observations, episode lengths and passive transitions can cause divergence, so this is not proof that the workers play identical games. It is an avoidable risk to experience diversity and reproducibility.

**Improvement:** use documented, reproducible worker-specific sampling seeds after loading, distinct from the learner seed. Record them. Separately establish whether local Practice exposes a usable track seed; do not claim matched game layouts merely from a Python seed.

### 4.2 “Verified gameplay time” is still an external clock estimate

The settled reward clock advances to the time of an accepted gameplay observation, minus excluded pauses. It is more defensible than crediting terminal-confirmation time, but it is not a Unity simulation timer. A new Electron paint serial also does not prove a new simulation tick; explicit invalidation can repaint unchanged content.

References: [clock update](../train_rs_v612_ppo.py#L184), [paint serial and invalidation](../tools/electron_worker/main.cjs#L55), [reward calculation](../tools/ppo_settled_learning.py#L24).

Under sustained rendering/simulation stalls, accepted gameplay frames could still be separated by substantial wall time. The audit cannot establish whether all that time corresponds to intended game progression. Conversely, time since the last confident gameplay frame before death is conservatively uncredited.

**Improvement:** validate wall-time survival against an independent progression signal in a short diagnostic, especially under load. Distance/HUD progress or trustworthy local diagnostic events may help; keep them out of policy inputs unless deliberately changing the problem. Review the local no-coin objective and terminal reasons as part of that environment contract; the Python reward itself does not separately measure coins or collision types. Do not remove safeguards or change reward coefficients based on this uncertainty alone.

### 4.3 Checkpoint saving is robust; shutdown finalization is less robust

The coordinator's `finally` block performs sampler shutdown, process joins, connection closure, host closure, buffer cleanup and preview shutdown before the final model save and report write. Several of these operations can raise without an enclosing per-stage safeguard. A second interrupt can also interrupt finalization. `parallel_training.json` is written directly rather than by atomic replacement.

References: [shutdown sequence](../tools/train_ppo_parallel.py#L525), [report writer](../tools/train_ppo_parallel.py#L377), [atomic model helper](../tools/ppo_atomic_checkpoint_v176.py).

An earlier preview run retained a RUNNING report after interruption. That observation is consistent with incomplete finalization, but does not identify which operation caused it.

**Improvement:** finalize the last fully committed learner state and status independently of best-effort cleanup; retain the original failure plus any cleanup errors. Write reports atomically. Test interrupted collection, worker failure, cleanup failure and interruption during checkpointing with small synthetic cases. Preserve the existing atomic save and unique snapshots.

### 4.4 BC is not yet a drop-in operation for the canvas/detector-off policy

The recorder and original BC entry point have framing selection. However, the cached BC trainer calls `approved_runs(root)` without forwarding a framing profile, so it selects the default legacy recordings. The recorder loads the detector and stores its auxiliary outputs; a framing label alone does not make those features equivalent to detector OFF.

References: [cached split](../tools/train_demo_bc_cached.py#L35), [approved-run filtering](../rs_v612_demo_bc.py#L21), [recorder model loading](../rs_v612_demo.py#L1059), [recorder canvas preprocessing](../rs_v612_demo.py#L1337).

The BC implementation deliberately freezes the visual extractor and critic, and trains the actor layers. It can improve decisions using existing representations; it does not teach the frozen visual extractor new visual features during that BC phase. PPO subsequently unfreezes the parameters and can update them.

**Improvement before more BC:** unify the observation contract across recorder, both BC trainers and evaluation. Validate detector-off feature reconstruction and exclude the wrong framing. Preserve original recordings; any conversion should produce a separate version and demonstrate actual pixel/feature alignment. Old downsampled/cropped frames cannot recover pixels that were never saved, although retained full-canvas previews may offer a separate conversion route with their own quality/timing checks.

Keep whole-run holdout, manual approval, label-validity limits and independent checkpoint output. Also port the cached trainer's epoch-zero candidate protection to the original BC trainer if that older entry point remains supported: the latter initializes its best validation loss to infinity rather than allowing the untouched source to win.

### 4.5 Audit PASS has a narrower meaning than end-to-end correctness

`audit_settled_ledger` replays supplied duration/reward substeps, checks passive attribution, capture lineage, discount arithmetic, terminal bootstrap exclusion and processed sample accounting. It does not independently reconstruct Unity time, check actual image contents, verify accepted physical actions or recompute GAE from recorded old values and next-state values. Those values are not all present in its current ledger schema.

Reference: [audit implementation](../tools/ppo_settled_learning.py#L325).

It also validates used indices for uniqueness/range, but does not itself require their count to equal every collected row. The current no-KL-stop shared coordinator provides additional coverage assumptions/checks; a standalone PASS should not be interpreted as proving every sample was optimized in every possible mode.

**Improvement:** give the report explicit audit dimensions. Add a small sampled numerical return/advantage trace and an independent reference calculation, plus tests that intentionally corrupt each invariant. Retain the existing synthetic boundary tests. There is no need to save every full image indefinitely or add expensive validation to each policy decision.

### 4.6 Reproduction metadata does not yet cover the whole experiment

The run configuration records important flags and the source checkpoint hash. Worker learning protocols hash the main trainer and settled-learning source, but not the full capture, framing, guard, parallel optimizer or local game dependency graph.

Reference: [learning protocol metadata](../tools/ppo_settled_learning.py#L108).

**Improvement:** record Git commit plus dirty-source hashes, dependency versions, Electron/game release identity, classifier/detector identities, input contract and worker seeds in a compact startup manifest. Link a final checkpoint to the last committed shared update. This makes recovery after a restart much less dependent on conversation history and file timestamps.

### 4.7 Local game rendering preferences can affect comparability

The local Unity wrapper supports `renderScale` and optional adaptive resolution. When enabled, its render-budget helper changes canvas backing resolution in response to frame intervals. The geometry guard checks viewport and element bounds; identical CSS geometry does not establish identical internal render resolution.

References: [Unity wrapper](../rankedsurfers-downloaded/game/releases/ae44e7782a4c8bbe5b10/arena.html#L162), [render budget](../rankedsurfers-downloaded/game/releases/a470c570db438c3e31be/render-budget.js), [geometry checks](../tools/ppo_background_display.py#L211).

Adaptive resolution defaults off in the helper. This audit did not establish that it is enabled in the active run. The opportunity is to record effective rendering preferences/backing dimensions, not to assume a current fault or alter them during training.

## 5. Learning choices worth measuring, not changing blindly

### Discounting and credit assignment

The elapsed-time correction is present. For current settings, continuation discount is `0.995 ** (seconds / 0.05)`, with a half-life of about **6.91 seconds**. The combined GAE trace multiplier is `(0.995 * 0.95) ** (seconds / 0.05)`, with a half-life of about **0.62 seconds**. These are mathematical properties of the configured estimator, not a maximum episode length or proof of inadequate learning.

Long survival can emerge from repeated local decisions. Nevertheless, delayed consequences of a lane choice deserve attention. Inspect action-conditioned advantages and value errors around obstacles before changing gamma/lambda. Small-minibatch actor advantage normalization and critic training on passive rows are deliberate estimator choices, not automatically defects. Because the feature extractor is shared, critic gradients can still change the representation used by the actor even when a passive sample has no direct actor loss.

### Action coverage and visual dependence

The rare lateral choices deserve a focused diagnosis. Useful evidence includes action probabilities for representative obstacle scenes, probability changes under different real observations, action-conditioned outcomes and a short collision trace. A policy that mainly responds to action history or episode timing could look active while making little use of visual geometry; that possibility is currently unproven.

The architecture combines a CNN image branch with an auxiliary branch. Its CNN compresses to a 3 x 6 spatial grid before a 128-feature projection. That is a finite representation, but the source alone does not establish insufficient capacity. The four-frame stack also spans a variable amount of time; interval features help expose this, but do not prove invariance across headless/visible capture cadences.

References: [extractor](../train_rs_v3_ppo.py#L54), [CNN](../train_rs_rl_early_death.py#L106), [frame stack/timing](../train_rs_v612_ppo.py#L127).

### Worker count changes learning batches as well as speed

One worker collects 128 transitions per update; three collect 384. Per-worker temporal rollout length remains 128, but the shared update sees more samples and executes more minibatches. This is not a pure throughput toggle for learning comparisons.

The parallel learner currently fixes progress remaining to 1 and rejects non-null target KL. Constant current learning rate/clip settings are compatible with that behavior. A future scheduled learning rate or clip schedule would not advance as expected without additional work; target-KL experiments cannot simply be passed through this prototype.

Reference: [learner setup](../tools/ppo_parallel_learning.py#L99).

### Clean framing is intentional, but evaluation must match it

`canvas_v1` uses the complete 1920 x 945 canvas and letterboxes it into 512 x 288. This produces a 512 x 252 game image with approximately 18 black rows above and below. Those bars preserve aspect ratio; they are not accidentally captured browser chrome. The separate guard image includes synthetic padding for historical guard coordinates.

References: [canvas preprocessing](../tools/ppo_framing.py#L9), [letterboxing](../rs_v5_segmentation_live_benchmark.py#L77), [guard composition](../tools/ppo_headless_env.py#L21).

Do not remove the bars, stretch the image or change observation resolution during an ongoing comparison. The principal remaining issue is enforcing that same transform wherever this checkpoint is used.

## 6. Performance opportunities that preserve reaction speed

The recent completed three-worker detector-off diagnostic achieved 33.57 optimized transitions/s with steady policy-request-to-command median/p95 of 31.6/38.3 ms. One worker achieved 17.50/s and 25.6/33.6 ms. Three workers therefore delivered about **1.92 times** the aggregate throughput of one in those separate diagnostics; one delivered about **52%**, not two thirds, of the three-worker throughput. These are not matched learning-quality experiments.

References: [three-worker report](SPEED3_DIAGNOSTIC_15M_20261031.md), [one-worker report](SINGLE_WORKER_PREVIEW_20261031.md).

Potential improvements, in order of evidence and risk:

1. **Reduce avoidable hot-path logging cost.** The settled ledger opens/writes/closes a file per transition. Measure its tail latency before changing it. A bounded writer or persistent buffered handle could reduce overhead, but must preserve ordering, commit semantics and useful crash diagnostics.
2. **Make long-run aggregation incremental.** Survival aggregation rereads and sorts all episode events every five seconds and rebuilds the curve. It is off the optimizer thread, which is good, but its work grows with run length. Incremental offsets and a bounded recent-episode deque can retain every historical graph point without repeated full rescans. Account for late-arriving events before freezing points.
3. **Profile copies and optimizer/collection balance before further memory changes.** Shared rollout transport removes large pipe serialization, but dense collection still copies into shared slots and then the learner's buffer. Some copies enforce ownership. Compact storage and shared inference have already shown latency tradeoffs; their lower memory use alone does not justify adopting them.
4. **Keep preview work optional and separate.** A requested 30 FPS preview does not imply 30 fresh policy observations/s. Preview encoding/network/browser rendering can consume resources. Benchmark it enabled and disabled if needed; the current no-dashboard run avoids that variable.

References: [ledger writer](../tools/ppo_settled_learning.py#L118), [survival aggregation](../tools/ppo_survival_metrics.py#L54), [rollout publication](../tools/ppo_rollout_transport.py#L31), [merge copy](../tools/ppo_parallel_learning.py#L64).

There is no current measurement justifying a custom Chromium build or GPU shared-texture rewrite as the next priority. The observed steady command latency is already much better than the earlier headless screenshot path. It remains a command-latency measurement, not a complete Unity reaction-time measurement.

## 7. Interpretation of results and the graph

The survival curve samples the rolling mean of the last 100 completed episodes across workers, every ten seconds; early points contain fewer episodes. Consecutive points overlap heavily and may contain exactly the same episodes. They are not independent batches of 100.

The terminal `overall_mean` uses all completed episodes in that run. Neither statistic includes unfinished episodes. If survival improves greatly, a completion-based graph may temporarily lag because the longest new episodes have not ended yet. Time-limited episodes are included and must be identified as censored at the configured limit.

A rising recent mean is useful monitoring evidence, but it mixes changing policies, stochastic actions, differing situations and correlated windows. A falling or sinusoidal-looking mean does not by itself identify a PPO defect. Historical deterministic evaluation means are additionally affected by the clock and input-contract differences described above.

The two-hour run is a useful diagnostic and a source of candidate checkpoints. Its training curve alone should not promote the final checkpoint over preserved references. Evaluate a small predeclared selection, such as source, midpoint and final, with matched settings once evaluation semantics are checked; do not select the apparent best training-curve peak and treat the same evidence as independent validation.

## 8. Recommended next steps

1. **Let the active run finish.** Preserve its periodic/final checkpoints, report, worker logs, ledger audits and curve. Determine whether all committed shared updates and worker records agree. Do not infer completion from an attractive graph or an existing `latest.zip` alone.
2. **Use those artifacts to diagnose behavior.** Compare action probabilities/counts, terminal sources where available, policy-age tails, pause/resume overhead and progression over time. Existing logs cannot reconstruct every collision or value estimate; state those limits explicitly.
3. **Address the concrete reporting/contract defects separately.** Add checkpoint-input compatibility enforcement, consistent evaluation metrics/status, independent worker seeds and resilient finalization, each with focused tests. Document each new flag/default. No reward, architecture or exploration change is needed for these fixes.
4. **Resolve death and action-effect uncertainty.** Use saved frames first. If insufficient, collect one short targeted current-backend diagnostic with synchronized observations, probabilities, commands, terminal evidence and visible outcomes. Investigate the template-only post-action path before replacing it.
5. **Compare frozen checkpoints under the same protocol.** Use the current canvas/detector-off contract and a consistent survival measure. Report distribution/tail survival, action counts, failures and uncertainty, not only the mean. Test visible-browser transfer separately from within-backend learning improvement.
6. **Choose one learning hypothesis from the evidence.** Examples are insufficient useful lateral exploration, a stale-decision tail, poor visual dependence or a credit-assignment problem. Change only the supported variable. Revisit BC after its input contract is aligned; defer recurrent memory and another large infrastructure rewrite unless evidence calls for them.

The important success criterion is a policy that responds correctly to obstacles and improves survival under a reproducible evaluation. More optimized transitions per second helps only when those transitions accurately represent that task.
