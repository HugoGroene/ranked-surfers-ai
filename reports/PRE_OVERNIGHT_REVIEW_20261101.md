# Pre-overnight review: road vision, capture and temporal spacing

Scope: important training/evaluation/control/observation paths and the day's
experimental changes, reviewed before the next user-run batch. No new live game
or training test was launched during this review. This is not a proof that every
source file or game state is free of defects.

## Policy inputs: timing and actions

- Four RGB frames, in oldest-to-newest order, plus four normalized frame ages and
  four validity flags. Each age uses a monotonic screenshot-request timestamp,
  divided by span plus 60 ms and clamped to [0,1]. These are conservative capture
  timing estimates, not Unity simulation timestamps or absolute wall-clock dates.
- The original 557-element auxiliary vector remains, extended to 565 elements.
  It retains four recent action choices as one-hot values, five times-since-action
  features (bounded at one second), recent observation intervals and episode age.
- The historical images and action context are separate inputs. There is no full
  timestamped action sequence attached to each selected frame. A command is not
  verification of its in-game movement. Lane odometry is disabled by default;
  detector-derived geometry is represented as unavailable.
- The policy is feedforward. Frame stacks and action context provide short
  history, not an LSTM state or guaranteed learning of button combinations.

Relevant source: `rs_v6_ppo_features.py`, `rs_v61_features.py`, the live `_pack`
and `_step_browser` paths in `train_rs_v612_ppo.py`,
`development/experiments/temporal_history/live_adapter.py`, and
`development/experiments/road_vision/history_spacing.py`.

## Concrete corrections from this review

1. Increased the overnight evaluation cap from six to 15 minutes. The former cap
   could leave a stronger model short of its requested 50 episodes. Incomplete
   evaluations remain explicitly ineligible for the completed comparison.
2. Added a default 12-hour scheduling budget. Before a new stage, the runner
   reserves its configured duration plus three minutes of startup/finalization
   allowance. Insufficient remaining time produces `BUDGET_LIMIT`, preserving
   completed work. This is not a hard kill watchdog; a pathological running-stage
   stall may exceed the estimate. Resume continues in a new output folder with
   the saved plan and unchanged sources. Failed training restarts fresh so its
   effective training budget does not silently differ from other arms.
3. Fixed early model-setup finalization: failures before `_settled_stats` existed
   could trigger a second exception before cleanup. Reporting and environment
   cleanup now still run, with a regression test for that failure path.
4. Updated a historical v1.7.2 installer test to test its exact historical input
   fixture instead of asserting that today's trainer still contains the obsolete
   method. Existing tests exercise the current refresh/menu recovery method.
5. Documented Koffi 3.3.6 alongside Electron 45.2.1 for the existing native bridge.
   No dependencies were installed or updated during the review.

No rewards, press duration, action mapping, terminal acceptance thresholds,
freshness bounds or production defaults were changed in this review.

## Reviewed experiment design and implementation

The suite has seven uniform-history crop/encoder arms and three additional
spacing arms on road192_spatial: ten configurations, two independent Python/Torch
seeds each, fresh initialization, 25 minutes of training and 50 deterministic
evaluation episodes each. Smoke weights are never reused for full trials.
Unity track seeds remain uncontrolled. This is a screening experiment; episode
confidence intervals do not capture all between-training-seed uncertainty.

All offline gates and all ten smoke training/evaluation pairs precede any full
training. All full training precedes evaluation. Completed-stage reuse checks
source hashes and saved artifacts. Unknown cleanup failures or user interruption
stop continuation; a failed training model is not evaluated. Atomic checkpoint
handling and settled-learning transition accounting are retained. An independent
stage failure is reported rather than silently counted as a successful sample.

The 128/192 existing-versus-spatial pairs isolate the encoder change. Spatial
128/192/256/384 isolates input size at the same crop. The full128 versus road128
existing pair checks crop. Four spacing patterns share road192_spatial; this is
not every possible resolution-by-spacing combination. A promising combination
may still need confirmation of interactions after screening.

Uniform/golden/logarithmic patterns use the same requested 500-ms span, with
different placements. Consecutive is a variable-span observation-pack control,
not an exact replay of the old production stack: missing startup history is
zero-filled with validity flags, and the new capture pipeline remains active.
Reset/pause isolation, latest-frame inclusion, distinct-frame selection and
checkpoint contracts are retained. Nonuniform BC replay is explicitly refused
until matching replay support exists. BC remains deferred.

Full composed frames feed safety checks. Policy crops alone remove side pixels;
they retain full height and letterbox without distortion. The new encoder keeps
an 8x8 grid, versus the old 3x6 grid. Larger images do not create a new reward or
action rule. A 384x384 frame has the same pixel count as 512x288, so it should not
be advertised as reducing input storage relative to that previous resolution.

GPU capture imports the Electron shared texture through D3D11, completes readback
while retaining the lease, copies through shared CPU memory, validates geometry
and serials and uses bounded stale-frame recovery. It is not zero-copy GPU-to-
PyTorch inference. Full-resolution readback and rendering remain workload costs.
The prior wall-clock capture failure fix uses a monotonic request roundtrip
upper bound, not cross-process wall-clock synchronization. A new paint callback
still does not independently prove Unity advanced its physics simulation.

## Accumulated live evidence and limits

- Short suite: 38/40 stages passed. Both six-worker GPU runs failed waiting for
  fresh paint. Four-worker GPU collection achieved about 60-61 transitions/s and
  39-41 ms p95 request-to-command latency; these are frozen collection figures,
  not optimized training throughput. Six-worker bitmap was slower with worse
  latency. Do not retry six unchanged or weaken freshness bounds.
- The subsequent road192_spatial GPU trial completed 7,552 optimized transitions
  in 600.47 active seconds (12.58/s), with a passing settled-learning audit and
  no finalization errors. The final 1,000 timing samples had about 29.9 ms median
  and 73.1 ms p95 policy-request-to-command latency. Capture averaged 46.8 Hz
  including pauses despite a requested 60 Hz. Do not call requested rates achieved.
- Its ten-episode evaluation averaged 4.10 seconds and chose JUMP for all 722
  decisions. The earlier 50-episode learned controller also chose only JUMP;
  its 4.35-second mean versus 3.91 for constant JUMP is not proof of visual skill.
- Earlier short temporal trials do not establish that more than four frames
  cannot help. Likewise, half a second is an unproven history hypothesis.
- Smaller images may discard distant coin detail; 256/384 are useful controls,
  not proven improvements. The video used single-frame imitation learning and
  additional controller logic, so architecture alone does not reproduce it.

Next live gate: the batch's new spacing and 256/384 smoke tests. Longer-run
learning quality, Unity clock accuracy, true input-to-game reaction and general
crop coverage remain empirical questions. Tune three/four-worker throughput only
after identifying a promising model configuration; then collect compatible human
demonstrations and assess BC. No high-confidence winning configuration exists yet.

## Offline validation

- Road-vision suite: 31 tests passed.
- Temporal suite: 45 tests passed, including setup cleanup and pre-launch stopping.
- Capture suite: six tests passed.
- Focused production dependencies: 89 tests discovered, 88 passed and one exact
  historical installer test skipped because the current trainer is newer.
- Electron host/bridge JavaScript syntax checks and new batch plan generation passed.

Total: 170 passing tests, one intentional historical skip. No live performance
claim is inferred from these offline tests. Raw checkpoints, browser profiles,
screen recordings and private prompt history remain local; source and curated
review documents are the publishable project state.
