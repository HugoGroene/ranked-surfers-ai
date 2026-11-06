# Short hybrid-history capacity suite

Isolated process-local adapter. Production and earlier experiment files are not
modified. Nothing here downloads maps, records BC, or starts games unless explicitly
run with `--execute`. Keep other training/evaluation stopped during measurements.

## Cases

Every row runs at both **256x256 and 384x384**, eighteen cases total. All use the road
crop, spatial encoder, Electron GPU shared-texture capture, detector off, one worker,
60 Hz requested capture and 30 Hz maximum decisions. Achieved rates are measured.

| Layout | Recent decision observations | Older target ages | Total frames |
|---|---:|---|---:|
| recent4 | 4 | none | 4 |
| sparse4 | 1 (current) | 100, 500, 1000 ms | 4 |
| sparse5 | 1 (current) | 50, 100, 500, 1000 ms | 5 |
| recent3_context2 | 3 | 500, 1000 ms | 5 |
| recent4_context1 | 4 | 1000 ms | 5 |
| recent3_context3 | 3 | 200, 600, 1000 ms | 6 |
| recent4_context3 | 4 | 200, 600, 1000 ms | 7 |
| recent3_context4 | 3 | 200, 400, 700, 1000 ms | 7 |
| recent4_context4 | 4 | 200, 400, 700, 1000 ms | 8 |

`sparse4` is exactly the proposed target-age set {0, 0.1, 0.5, 1.0} seconds.
`sparse5` tests adding one nearby fixed-age image. The other five-frame variants
retain a consecutive recent group. Five frames were omitted from the old candidate
list, not rejected by measurements. The isolated adapter now supports them without
changing the earlier experiment's validator.

The recent slots preserve the last distinct policy observation timestamps. Their
spacing is measured, not asserted to be constant. Older slots select the closest
capture at or before each target, with at most 60 ms selection error. A slot that
overlaps the recent group, lacks history, or exceeds this tolerance is invalid.
The latest slot always uses the current guarded image. History cannot select a
future frame relative to that image. All slots are ordered oldest to newest.

Missing slots are zero-filled, with explicit age and validity features. There is
**no startup waiting period** and no pretend one-second-old copy of a new image.
The whole history clears at reset and optimizer pause/resume. The policy still
gets the inherited action/timing features. Capture freshness limits, guards,
reward settings and PPO hyperparameters remain unchanged.

`run_probe.py --context-ms ...` permits different increasing older ages for a later
chosen configuration. The checkpoint contract records their exact meaning; an old
four-frame checkpoint cannot silently migrate to these inputs.

## Run in another PowerShell terminal

Use the same local Practice server already used by the previous evaluations.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -u .\development\staged\hybrid_history_20261102\suite.py --execute --minutes 4
if ($LASTEXITCODE -ne 0) { throw "Hybrid checks contain a failure. Preserve the printed report folder." }
```

Estimated **about two hours; allow 2–2.5 hours** depending on startup/reset speed.
Scheduled training is 18x1-minute smoke runs + 18x4-minute performance probes =
90 minutes, plus 54 smoke evaluation episodes, browser startup, final updates,
checkpoint I/O and preview export.
The three-episode evaluations have a three-minute cap per case. This estimate is
not a hard overall timeout.

Order:
1. Pure buffer/runner tests and existing road tests.
2. CPU-only real PPO update, all candidate encoder shapes, checkpoint reload and
   incompatible-contract rejection. No browser in this step.
3. All eighteen one-minute fresh-model smoke training runs, then matching three-episode
   evaluations. Any smoke failure stops the suite before performance testing.
4. Eighteen four-minute fresh-model PPO probes. This includes optimizer updates,
   pause/resume, death/reset handling, saving and settled-learning audits.
   An independent probe can fail and later probes continue only after confirmed
   cleanup; interrupt or uncertain cleanup stops everything.

All probes start fresh to avoid migrating incompatible channels. These are short
**systems tests**, not a contest of learning quality. Each model's different random
behavior can affect timing. No existing checkpoint is overwritten; no defaults
are adopted automatically. There is no dashboard.

## Results and interruption

The runner prints stage progress and the root `reports/hybrid_short_...` path.
`performance_summary.json` collects audit status, throughput, reaction metrics,
achieved capture rate, stale-frame retry counts, history ages/availability, and
resource sample paths. Resource samples include system CPU/RAM availability and
GPU telemetry; they are not precise per-worker private-memory measurements.
`performance_summary.md` presents median/p95 policy-request-to-command latency,
throughput and achieved capture rates in a table.

## Inspect exactly what the actor received

Open `policy_input_gallery.html` in the printed suite folder after completion (also
generated if the live suite stops with an error). It groups labeled contact sheets
by configuration. The images are **stored actor rollout observations**, not an
extra screenshot reconstructed later. Up to six observations per smoke training
stage are sampled across its early rollouts, including early and best-available
history where present. They are not guaranteed to contain a particular obstacle
or mature history; use their age/validity metadata.

Inside `smoke/roadSIZE_LAYOUT/train/policy_inputs/`:
- `*_sheet.png`: all frame slots at their native policy resolution, labels outside
  the pixels identifying actual ages and missing slots.
- `*_slotNN.png`: individual lossless RGB inputs without labels or resizing.
- `*.npz`: exact image and full auxiliary arrays, including action/timing history.
- `*.json`: sampled action, rollout index, episode-start flag, ages and validity.

The CNN subsequently applies its usual normalization. The evidence is copied from
populated actor rows at rollout boundaries after the existing pause handling; no
new captures or disk writes occur in the decision loop. All image export occurs
after the game closes. Preview collection is enabled **only for smoke training**;
the dedicated performance probes have no preview capture/copy/export overhead.

## Rewards remain fixed for this systems comparison

The useful part of the quoted advice is to examine temporal credit assignment,
not to assume any single gamma or airborne penalty is a solution. With the current
elapsed-time discounting and gamma=0.995 per 50 ms, one-second-ahead reward has
weight 0.995^20 = approximately 0.905; gamma=0.999 would give approximately 0.980.
Gamma tuning may merit a later learning experiment, alongside GAE/critic quality,
but cannot supply a missing observation of an occluded coin. Short capacity probes
cannot establish which reward setup learns best.

The current settled rewards already give verified active survival progress,
milestones and a death penalty. Adding an airborne penalty could discourage the
required jumps. Shaping rewards can change the objective unless designed carefully;
see [Ng, Harada and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
There is no evidence here that changing rewards is required before testing history.

History reports distinguish startup-inclusive availability from mature history
availability. Missing old slots under contention must not be mistaken for a
successful eight-frame test. Reaction metrics report command issue time, not the
time Unity visibly responds. Diagnostics use bounded samples to limit overhead.

Resume with identical code, retaining completed audited stages:

```powershell
& $py -B -s -u .\development\staged\hybrid_history_20261102\suite.py --execute `
    --resume "reports\hybrid_short_REPLACE_WITH_PRINTED_FOLDER"
```

This creates a new continuation folder and verifies retained artifact hashes.
Incomplete stages restart from scratch; their partial outputs remain in the old
folder. For changed code, retain earlier results as separate evidence rather than
silently mixing implementations.

After reviewing p95 latency, capture retries, history availability and throughput
against each resolution's recent4 baseline, choose a viable input configuration.
Then test worker counts with that configuration before longer diagnostic training.
Capacity alone does not prove improved learning or long-duration reliability.

The completed one-hour vision evaluations are reviewed in
`reports/recovered_evaluations_20261102_160456_846434/REVIEW.md`.
