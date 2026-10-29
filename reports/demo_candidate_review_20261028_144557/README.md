# Candidate demonstration review — 28 October 2026

User clarification: these six runs were mainly input/capture tests. Keep them
as diagnostic candidates. Do not approve them automatically or train BC from
them. Fresh deliberate gameplay demonstrations are the next recommended step.

No replay, label array, manifest, checkpoint or approval status was modified.
`analysis.json` contains recomputed timing/label checks and original manifest
SHA-256 hashes. The lateral sheets show the exact stored model frames at
offsets -2, 0, +2 and +5 from each lateral label. This is sampled visual review,
not proof that every move in every full video was optimal.

| Run suffix | Duration | Valid LEFT / RIGHT | Technical findings |
| --- | ---: | ---: | --- |
| 211547_738959 | 5.96 s | 0 / 1 | Clean timing; little lateral coverage |
| 211627_127845 | 38.30 s | 9 / 10 | Strongest diagnostic candidate; no >130 ms gaps |
| 211822_036384 | 38.48 s | 6 / 5 | Two long capture gaps around 34 s affect a LEFT stack |
| 211841_149579 | 6.83 s | 1 / 1 | Clean timing; little lateral coverage |
| 211844_541805 | 2.20 s | 0 / 0 | No lateral labels; default trim leaves only NONE; preview speed check fails |
| 211845_625831 | 6.89 s | 2 / 2 | Clean timing; small lateral sample |

All six sets of saved labels reproduce from raw events. Every eligible keydown
was assigned. Win32/browser keydown counts match. Median intervals are about
49 ms. Browser-vs-Win32 event matching found no labeled non-NONE sample whose
matching browser keydown timestamp preceded its observation-end timestamp;
this supports timestamp alignment but does not independently measure Unity's
internal input handling or pixel acquisition instant.

## Visual and observation issues

- Contact sheets include the opening logo/chase/camera transition. These are
  unsuitable as assumed expert NONE gameplay. Each run has 17–20 samples marked
  valid in its first second; validity is an interval/label check, not a proof of
  good controllable gameplay. A reviewed leading trim is needed for future BC.
- Lateral sequences visibly show lane changes in several recordings, supporting
  human input delivery. They do not establish that autonomous 5 ms taps behave
  identically or that each move was the safest choice.
- Frame-gap issue: run 211822 has 493.458 ms and 201.160 ms gaps at frames 671
  and 672. Frames 672–674 are marked valid but their reconstructed four-frame
  histories cross a long gap. Frame 674 is LEFT at 34.355 s. `Replay.sample()`
  reconstructs histories directly and does not filter this history gap itself.
  Run 211844 has three analogous early-history samples after a 145.670 ms gap.
- Full-canvas source framing is 1920x945 windowed rather than the PPO session's
  fullscreen geometry. Crop/letterbox code is shared, but differing aspect ratios
  can still change visual contents. Prefer consistent framing for fresh demos.
- The videos use constant 20 FPS while capture timestamps vary. Use original
  timestamps/frame indices for label review; preview playback seconds are only
  approximate, particularly after a gap.

## Illustrative exclusions, not approval proposals

Before the user clarified the diagnostic intent, two conservative technical
windows were calculated. They remain illustrative only, not approved segments
or recommendations to turn tests into expert data:

- 211627: [2.15, 37.603) preserves 9 LEFT, 10 RIGHT, 6 JUMP and 2 ROLL labels.
- 211822: [2.15, 33.50), [34.70, 37.778) preserves 5 LEFT, 5 RIGHT, 9 JUMP and
  2 ROLL labels, excluding the affected history near 34 s.

These windows remove obvious startup/gap concerns but cannot establish that
all remaining decisions were good. Default last-0.7-second trimming likewise
does not establish that the action causing a death has been removed.

## Recommended fresh collection

Record varied, genuinely good local Practice play with both useful lane changes
and appropriate stay-put decisions. Use the same stable display framing as PPO;
enter fullscreen and finish calibration before playing. Autosave as candidates.
Review raw event timing and exact action sequences, then approve only the
selected good intervals explicitly. Never relax the 130 ms validity bound to
increase sample count.

The BC loader's three-approved-run minimum is a software guard, not an evidence-
based sufficient dataset size. Aim for diversity across independent runs and
enough LEFT/RIGHT examples to populate both training and whole-run holdout.
Initially 10–20 purposeful runs is a practical collection target, not a guarantee
of improved survival. Inspect per-action holdout performance before accepting
overall accuracy as evidence that lateral decisions were learned.

Then train the existing conservative actor-only BC implementation on a separate
copy of the latest PPO checkpoint and perform the same 100-episode deterministic
evaluation. No BC training was performed as part of this review.
