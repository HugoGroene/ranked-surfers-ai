# Next experiment: input delivery, then a small imitation-learning branch

Research and local source review: 28 October 2026. This is an experimental
recommendation, not evidence that an intervention has already improved gameplay.

## Decision

Do not prioritize another unchanged eight-hour continuation. Preserve both
checkpoints. First verify physical lateral movement at the trained 5 ms tap
duration. If delivery works, curate demonstrations and test actor-only behavioral
cloning on a separate checkpoint. Consider a short controlled exploration
experiment if demonstrations are unavailable. Neither more training nor BC is
guaranteed to solve the problem.

## What the project actually establishes

- Matched deterministic tests: 3.9933 s before, 4.2970 s after; 100 episodes each.
  The after-minus-before mean difference is +0.3037 s, with episode-bootstrap
  95% interval [-0.1575, +0.7503] s. This interval includes zero and does not
  include training-seed variability. Timing also varied between sessions.
- Both policies selected zero LEFT/RIGHT actions. Mean lateral probability was
  0.24% before and 0.45% after. The continuation shifted ROLL toward JUMP.
- The 386,829-step training run sampled 3,901 lateral actions (about 1.01%).
  These are scattered decisions, not proof of successful lane-change sequences.
- Checkpoint metadata, read directly from ZIP `data`: n_steps=128, batch_size=32,
  n_epochs=5, learning_rate=0.0003, ent_coef=0.002, gamma=0.995,
  gae_lambda=0.95, target_kl=None. CLI defaults are not sufficient to establish
  resumed hyperparameters; PPO.load restores checkpoint settings.
- Six demonstration manifests are candidates, with 18 LEFT and 19 RIGHT valid
  labels in total, before any manual segment review. They are not approved data.
- Observations include four RGB frames plus 557 features. Image columns are not
  verified track lanes; disabling commanded-lane assumptions is intentional.

No source evidence establishes whether poor lane behavior comes from physical
input delivery, inadequate examples, representation, exploration or learned
value. Calling it a known exploration bug would overstate the evidence.

## What the primary sources support

1. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
   stresses sample inefficiency, evaluation separate from exploratory training,
   and consistent wrappers. More interaction can help, but a training reward
   curve does not settle deterministic skill or environment correctness.
2. [RLiable](https://arxiv.org/abs/2108.13264) shows why point estimates from few
   runs can mislead. Our episode interval is useful within these two trained
   policies; it is not a substitute for independent training runs.
3. [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
   exposes entropy regularization and target_kl. Clipping alone does not bound
   all policy updates. Neither increasing entropy nor adding a KL limit is
   established as the fix here. They should be isolated experiments, not a
   bundle of simultaneous changes.
4. [Behavioral cloning documentation](https://imitation.readthedocs.io/en/latest/algorithms/bc.html)
   describes supervised learning from observation/action examples and warns
   about error recovery/generalization. It directly addresses the lack of
   successful lateral examples but still requires live validation.
5. [DAgger](https://arxiv.org/abs/1011.0686) addresses the distribution shift
   created when an agent's own errors lead to unfamiliar states. It is a later
   option if clean BC can imitate examples but fails after its first mistake.
6. [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732)
   demonstrates benefits from combining demonstrations and RL in other games.
   Its DQN/replay algorithm is not a drop-in PPO result or proof for this runner.
7. [Primacy bias](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf)
   provides evidence that continued interaction need not undo early learning
   biases. Its tested methods/tasks differ from this PPO setup; do not reset
   weights based on this analogy.

## Ordered experiments and decision gates

### 1. Verify input delivery before training

Use the existing guarded input probe, starting in ordinary confirmed Practice
gameplay. Begin at 5 ms, then compare 25 ms only as a separate diagnostic if
needed. LEFT and RIGHT must be tested from a non-edge lane. LEFT_TWICE is useful
for repeat-event behavior but a second LEFT at the track edge proves nothing.

Record browser keydown/up counts and visually inspect before/after character
position. Browser delivery alone is not proof that Unity consumed the input.
The existing probe records two frames separated by several NONE decisions;
camera motion/occlusion can make a pair ambiguous. Such cases remain unverified.
Repeat ambiguous cases and inspect short time sequences rather than guessing.

If 5 ms inputs fail while longer pulses reliably work, address input semantics
in an isolated branch before any BC/PPO experiment. A 5 ms pulse being shorter
than a rendered frame is a reason to test, not proof of lost input.

### 2. Inspect demonstrations, then test a conservative BC copy

Manually review framing, lateral label timing, held keys, pre-death errors and
selected segments. Candidates remain excluded until explicitly approved. The
existing minimum of three approved runs is a loader check, not a sufficient
sample-size claim. Collect varied situations in both directions, including
successful lane moves and cases where staying put is correct.

Use the existing actor-only BC helper on a new output checkpoint, retaining
frozen extractor/critic initially and whole-run validation holdout. Ensure the
holdout contains lateral examples; overall accuracy can hide poor rare-action
recall. Add per-action holdout metrics before interpreting a BC result. No BC
training or automatic approval was performed in this research task.

Compare against its exact PPO source with the same 100-episode deterministic
protocol. Look for correct lateral decisions, survival distributions, >=5/10 s
rates and guard events. More lane changes alone is not success. If results are
close, repeat in alternating model order; session/procedural variation remains.

If holdout imitation improves but live recovery fails, gather corrective examples
on the agent's visited states (DAgger-style). If even reviewed labels cannot be
imitated with a frozen extractor, investigate representation before long PPO.

### 3. If demonstrations are unavailable, use a short exploration experiment

Keep an unchanged PPO control and one separate candidate from the same source.
Change only the entropy coefficient; verify the loaded model actually uses the
intended value. Use a fixed step budget rather than only wall time. Evaluate
both deterministic policies before committing more compute. Exact coefficient
selection requires a small pilot; there is no evidence-backed universal value.

Do not introduce a lane-change reward: that can reward movement without safety.
Do not simultaneously add LSTM, change detector size, alter gamma, change input
duration or reset the actor. A more complex model is premature until delivery
and example quality are established.

## Chromium startup focus implementation and limitations

`tools/ppo_browser_focus.py` is called by the PPO startup override after browser
setup and before the first reset/episode. It activates the Playwright tab,
temporarily assigns an unpredictable title to identify exactly one Chromium
top-level window, attempts SetForegroundWindow and one WScript AppActivate
fallback, and restores the title. Both the native foreground HWND and browser
focus/visibility must agree. Normal window geometry is not changed. There are
no simulated Alt/Tab keys, clicks, global Windows setting changes or focus
stealing retries during gameplay. A denied activation allows up to eight
seconds for manual foreground selection, then stops before gameplay input.

[Playwright bring_to_front](https://playwright.dev/python/docs/api/class-page#page-bring-to-front)
activates a tab. [Microsoft SetForegroundWindow](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow)
documents OS restrictions and possible denial. Foreground ownership does not
prove that no always-on-top overlay covers the game; MSS still needs an
unobstructed canvas. This feature cannot guarantee unattended launch everywhere.

Ten focus/integration tests passed, including denial, ambiguity, title restore,
OS/DOM disagreement, shell fallback, passive manual focus and actual startup
error propagation. The 36 existing current-runtime/pause regressions also passed
after the startup hook was added. Two live attempts (143231 and 143349) failed
foreground verification and safely stopped before gameplay input. The final
passive manual-focus grace is unit-tested, not yet live-verified. No input-pair
evidence was collected, so physical lane delivery remains unverified.
