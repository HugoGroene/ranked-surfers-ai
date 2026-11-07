# Persistent map classifier recordings

Run record_both.ps1 for two independent manual sessions per map (training and
held-out validation). Each launches a fresh local Electron game on port 8010;
no separate server is required. Stop other games/benchmarks first. No inputs
are automated. Both maps remain intact and the active classifier is unchanged.

During each session, leave menus visible for several seconds, start Practice,
play, die and restart several times. Include the startup camera transition,
jumping, trains, tunnels and pause/resume. Escape pauses; do not use browser
fullscreen. The canvas fills the window automatically for classifier recordings.

Canonical images are saved without predicted class names under
ai_data/multimap_screen_states/sessions/<map>_<split>_<timestamp>.
They are the full 1920x1080 guard image with 1920x945 canvas at (0,87), NOT the
PPO road crop. Raw canvas evidence and prediction logs remain under reports.
Session manifests retain map root, split, geometry and evidence location.

Use the printed old-tool label commands: M menu, L loading, G actionable
gameplay, O other (including visible pause overlays/ambiguous transitions),
K skip, B back, Q save/quit. Predictions are never accepted as ground truth.
An overlay-free frozen gameplay image cannot reliably be labelled paused from
pixels alone: skip ambiguous images; runtime pause verification remains separate.
The labeller retains source images and writes labels.csv plus labelled copies
in dataset/<class>/<session>. Quit with Q to save your work.

Do not invoke the old trainer's default random/single-session validation for
multimap promotion: reserve the marked validation sessions from BOTH maps.
Next step after labelling is candidate training with both maps and independent
per-map confusion matrices, then live startup/death/pause/resume checks.
Never overwrite the active checkpoint as part of recording or labelling.

For future maps, extend maps.json with a locally imported map directory and release ID
and record only that new map's train and validation sessions. Reuse the retained
old-map dataset, keep session-level separation, retrain a candidate on all maps
and revalidate each map before adoption. New framing or guard preprocessing
requires a new dataset version, not silent mixing of incompatible images.

The registry selects the active classifier for current multimap entry points and
records map roots and a historical continuation checkpoint. The resume/evaluation
wrappers select completed runs separately; this checkpoint field is not a latest-model pointer. Candidate checkpoints
belong in new timestamped directories under the dataset's candidates directory,
with training configuration, session IDs, image hashes and per-map validation
reports. Promote only after live checks; retain the previous active checkpoint.

Legacy recovery: archive_legacy.py copies the authorised rebuild_01 dataset from
the original project into archives with per-file SHA-256 verification. It never
changes its source. The seven sessions include rebuild01_g, previously held out.
Do not relabel this session as training if comparing to the old validation result.
A reviewed MENU example contains a loading overlay, so legacy labels are not
automatically trusted. Legacy desktop images include chrome/taskbar; fresh exact
guard recordings are recommended for both maps. Preserve the archive for review.

Training startup exclusion (6 November): multimap training/resume now plays one
complete first episode per worker outside PPO collection and Monitor. Its actions,
rewards and value targets never enter the rollout buffer or training statistics.
The normal reset then clears episode history before collecting episode two.
Guards remain active throughout; a guard failure still stops the run. Each worker
saves `startup_episode.json` with the excluded episode's duration, actions and
reward. Low-level environment diagnostics retain that episode. Startup wall time
still counts toward total elapsed time/throughput. This happens once per worker
process, including separately launched smoke workers, not after optimizer pauses.
`stability.py --no-exclude-first-episode` reproduces the old collection behavior.
Evaluation episode counts and behavior are unchanged.

Bounded continuation recovery: `resume.ps1 -Minutes 30 -Checkpoint <path>` runs
the two-minute smoke first (any failure stops), then a shared 30-minute wall-time
window for training and recovery. It uses four headless workers, both maps, no
dashboard, and shows per-map graphs at the end. The window includes worker
startup, excluded first episodes and cleanup; it stops at an optimizer boundary,
so finalization can finish slightly beyond the deadline. Smoke/setup is additional.

Known capture/resync uncertainty stops can restart fresh workers from the latest
verified final atomic checkpoint. CRC, input contract, source hash, completed
optimizer step count and finalization status must agree. No uncertain rollout
is carried across a restart. `-MaxRestarts 3` is the default; use 0 to disable.
Two consecutive failures before 60 active seconds (or without an update), unknown
errors, integrity errors, or Ctrl+C stop recovery. It does not restart Windows
after a power failure. Servers are shared across attempts and closed at the end.

Reports retain `stability/`, any `recovery_01/` etc., and a top-level
`recovery.json` containing attempt errors, paths, remaining budget and checkpoint.
Final per-map graphs combine all training attempts, excluding smoke. Completed
recovery runs are recognized by the usual automatic resume/evaluation selection.
# Active classifier (6 November 2026)

`maps.json` now selects `multimap_20261106_221921_183521/candidate.pt` for the
current multimap training, evaluation, comparison, validation and recording tools,
including the BC recorder. The training/evaluation/probe commands accept
`--state-model` to test a candidate without changing this selection.
Historical experiment defaults and the baseline training manifest remain unchanged.

Promotion evidence: `reports/latency_probe_20261106_231119_982942/classifier_promotion.json`.
The four-worker test completed in 86 seconds with two shared PPO updates, deaths
on both maps, pause/history restoration on every worker, passing learning audits
and no forced cleanup or leaked workers. Ordinary command latency was 34.0 ms
median / 38.3 ms p95. This is a short integration check, not an overnight reliability
claim. The original PPO checkpoint and prior classifier were preserved.

## Current continuation (7 November 2026)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\development\experiments\multimap_classifier\resume.ps1 -Checkpoint "reports\multimap_stability_20261107_035054_502059\stability\train\latest.zip" -Hours 10 -MaxRestarts 3
```

This selects the completed 15-minute post-BC continuation explicitly. Diagnostic
runs (`--diagnostic-only`) and visible-worker probes are excluded from automatic
latest-checkpoint selection. The wrapper uses four headless workers, dashboard off,
its own map servers and graphs at the end. Three restarts are additional attempts
within the shared ten-hour budget; smoke/setup/finalization add wall time. Safety
recovery does not bypass freshness checks or retry arbitrary errors.

The current measured runtime and remaining rare capture stalls are documented in
[the latency review](../latency_tail_20261107/README.md). No classifier was changed
as part of these latency optimizations.
