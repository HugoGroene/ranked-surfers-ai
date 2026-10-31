# Bounded MENU confirmation routing: 31 October 2026

## Recorded failure

`reports/v612_sharedmem_4workers_2h_20261031_112933` stopped after 969.70 active
seconds, 53 shared updates and 27136 optimized transitions. Worker 1 episode 112
exhausted the ordinary six-capture guard without sending a movement command.
The saved image shows the title/menu scene. The classifier's last five MENU
probabilities exceed .9989; this failure does not demonstrate a classifier error.

The last five template scores were .9990624023335765, .5309857835812407,
.4936935572656977, .7347898673175708 and .6525108082289421. With the existing
.68 strong threshold and .544 corroboration threshold, these contain two strong
hits and three corroborated readings. Both final readings are corroborated.
The previous extra-check candidate required four corroborations, so it refused
to collect additional evidence despite the sustained, independently supported
MENU predictions.

The original JSON is copied verbatim to
`tests/fixtures/menu_transition_20261031_ep0112.json`. Historical reports,
screenshots and checkpoints remain unchanged.

## Focused change

`tools/ppo_stream_menu_tail.py` retains its previous candidate route and adds a
second route. After at least six ordinary captures, its last five must all be
confident ML MENU without visual gameplay evidence, contain at least two strong
template hits and three corroborated readings, and have corroboration on both
final frames. This authorizes only additional passive evidence collection.

The existing helper still waits 45ms between up to six additional captures,
rejects new non-MENU or positive visual-gameplay evidence, and delegates all
terminal decisions to the unchanged `sequence_decision`. It never returns PLAY.
No new CLI option, retry clicks, classifier changes, reward changes, frame-age
changes or terminal acceptance thresholds are introduced. The existing scope
remains Electron/stream workers.

## Offline evidence and limits

The new regressions first reproduced the unpatched stop using the exact JSON.
After the change, a hypothetical additional MENU frame with template score .60
satisfies the original six-frame terminal rules; this is NOT a frame captured
in the failed run and does not prove that live recovery would have succeeded.

Actual checked-out `_step_browser` tests verify seven-capture confirmation with
zero movement/keypress or action attribution, and a 12-capture safe stop when
all six extra templates remain weak. Additional tests reject gameplay, loading,
pause/unknown states, non-finite scores and visual gameplay; capture failures
propagate. ML-only or insufficient/receding template support cannot enter the
new route. Previously eligible candidates retain their prior behavior.

Validation: 193 focused offline tests passed, including current runtime,
settled/shared learning, pause/resume, capture freshness and guard regressions.
Syntax and Git whitespace checks passed. Hash checks confirmed that the trainer,
terminal decision rules, screen-classifier code, capture adapters, settled-learning
implementation and both preserved latest checkpoints were unchanged by this fix.

Live recovery and a new long-run completion have not yet been tested. Another
inconclusive or invalid UI sequence must still stop safely. Detector-off work
is separate and remains paused.

## Preserved progress

The failed attempt's `latest.zip` passes ZIP CRC checks and contains 1846686
total timesteps and 68020 update epochs. SHA256:
`93cc8f240342dff3b1b418e547d56e1415e5b5704a98b6a5c168e5adf8b324dd`.
All four committed rollout prefixes independently pass `PASS_SETTLED_LEARNING`
at 6784 optimized transitions each. The incomplete rollout contributed 146
ledgered transitions that were not optimized; three additional interactions
were reported as completed but not ledgered when the cohort aborted.

The 30-minute source remains unchanged, SHA256
`fb97d304db4ef60d2caf2e135628615c7b68b7ef9a551c40093659fb4b86abd2`.
The interrupted run recorded 436 completed training episodes, mean 4.2487s,
best 13.5278s. These are stochastic training statistics, not policy evaluations.
