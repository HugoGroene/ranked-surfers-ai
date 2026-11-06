# Multimap demonstrations â†’ separate BC phase â†’ ordinary PPO

This is isolated development code. It does not change the running trainer, classifier,
guard, checkpoints, or older recorder. The recorder has completed live sessions on both maps. The current dataset export
and its integrity audit are complete; BC gameplay benefit remains unproven.

The video creator trains a supervised image/action CNN, rather than mixing BC into
PPO: [source](https://github.com/nikp06/subwAI/blob/main/ai.py). Our older demo patch
froze the feature extractor. This BC phase instead updates the shared visual/auxiliary
extractor and actor on a checkpoint copy. Critic-head weights stay fixed, but their
predictions change as shared features change. Evaluate before continuing PPO.

## Record after training finishes

Run from the repository root. The recorder starts its own local server on port 8012.
It refuses to run alongside a detected trainer/evaluator/other recorder. No public
website is used. The existing recordings remain untouched.

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$bc = ".\development\staged\multimap_bc_20261106"
& $py -B -s -u "$bc\record.py" --map halloween --split train --seconds 30
```

Click **Practice** to request the recording layout; **F8** is the fallback. The
recording timer starts only after the canvas layout is ready, with a visible
`[BC RECORDING]` terminal message. Startup waits at most 120 seconds, separately
from the requested recording duration. Keep the game foreground and
unobscured. Use **arrow keys or WASD**, with brief taps rather than held keys. Restart using
**R** yourself. Escape ends the current candidate segment; resume/restart manually.
No gameplay keys are injected. The first automatically started round may not be
recorded until a menu has been observed; after that, two gameplay observations start
each candidate. Ctrl+C preserves completed candidates and finalizes the partial one
when capture/writing permits.

Check this first recording before collecting more. Then make separate training and
validation recordings on **each** map, for example:

```powershell
& $py -B -s -u "$bc\record.py" --map halloween --split train --seconds 900
& $py -B -s -u "$bc\record.py" --map tokyo --split train --seconds 900
& $py -B -s -u "$bc\record.py" --map halloween --split validation --seconds 180
& $py -B -s -u "$bc\record.py" --map tokyo --split validation --seconds 180
```

These are initial collection amounts, not a guaranteed sufficient dataset. Include
all five action classes and useful lane changes/jump/roll sequences. You need not
demonstrate perfect four-coin combinations. PPO can subsequently explore beyond the
demonstrations. Long runs are candidates for quality, not automatic proof of it.

The default per-run limit is now 900 seconds (formerly 120); session expiry still
ends the recording. `--max-run-seconds` can adjust this independently. Keep a few
short difficult openings and lane-switch examples alongside the best long runs.

## Passive classifier diagnostics during recording

Every captured frame has a timestamped state/probability record in
`guard_observations.jsonl`, flushed approximately once per second. It includes
candidate run age, capture interval, capture/classifier costs and saved-image
references. `guard_summary.json` totals states globally and per candidate.
Non-GAMEPLAY inside a candidate and classifier exceptions produce rate-limited
terminal warnings. Existing native frames provide evidence; when a terminal frame
falls outside a candidate, up to 120 additional evidence images per session are
saved with the same bounded writer, no faster than one per second. No additional
capture is requested. Missing image references indicate no image was saved.

These are raw classifier warnings, **not a replay of the full stateful PPO guard**.
Normal deaths/pauses can correctly trigger them. No automated keyboard inputs,
pause/resume tests or classifier updates occur. Unknown frames remain recorded
inside candidates and excluded from export. Classifier exceptions are logged as
UNCERTAIN so human footage can continue. Two MENU predictions still segment a
candidate; if those were false positives, inspect both fragments before approval.
Warnings never automatically become corrected labels or accepted BC samples.

The review gallery counts non-GAMEPLAY frames and offers **Next state warning**.
Use this to inspect later-map sections. Any confirmed classifier failures should
be reviewed/labelled separately before a long automated training stability test.
Keep original sessions; no old data or classifier is overwritten.

Native 1920Ã—945 canvas PNGs and timestamped physical key events are saved under
`ai_data/multimap_bc_v1/sessions/`. Native masters allow re-export at another policy
resolution without recording again. Disk usage can be substantial. Start at the
default 30 capture requests/s; 60 is optional, not guaranteed achieved. The bounded
writer drops frames under overload rather than accumulating RAM. Invalid long gaps
are rejected during export, and the dropped-frame count is saved.
The writer now uses two encoding threads with lossless PNG RLE and still permits
only four pending images. A first live check exposed 198 drops with the former
single-thread writer and no actions because WASD was not yet recorded. That old
session cannot supply BC labels; keep it only for image/classifier inspection.
Repeat the short live check to validate the revised writer and WASD capture before
collecting the 15-minute sessions. Native masters can use tens of GB in long sessions.

## Review, select and trim

Latest preparation (6 November evening): the user changed end trimming to **three
seconds** for both old and new recordings. The minimum remains 12 seconds. New
append-only review revisions supersede the earlier five-second selections;
recordings and old review revisions remain intact. The selection report is
`reports/bc_selection_20261106_215913_630820.json`: 206 training runs (~81.7 minutes)
and 15 held-out validation runs (~6.6 minutes). Broken/empty early smoke sessions
and the pre-WASD session without action labels are not selected.

The classifier import now supports `--all-warnings`: every saved uncertain/other/
loading candidate frame, menu transitions, saved diagnostic evidence and nearby
context. This supplements previous sparse imports without duplicating their source
frames or changing existing labels. Missing/dropped native frames cannot be recovered.
Imported images initially remain unlabelled; an uncertainty warning is not an approved state.
The 6 November review queue has since been human-labelled and used for the new classifier candidate.

Review the complete BC-derived queue with:

```powershell
& $py -B -s .\development\staged\multimap_bc_20261106\label_guard_queue.py
```

M=MENU, L=LOADING, G=GAMEPLAY, O=OTHER, K=skip, B=back, Q=save and stop the queue.
Run the same command to resume at the first unlabelled image. Completed sessions
are skipped, with labels and import-manifest hashes checked. The old standalone
label command still starts at the beginning unless `--resume` is explicitly set.
No classifier or PPO training is launched by this command.

Classifier training remains a dataset expansion: `multimap_classifier/train.py`
now verifies that all 1,731 examples from the current baseline manifest remain
present, byte-identical, identically labelled and in their original splits. The
expanded dataset includes them plus newly reviewed examples. It trains a new
candidate; it does not silently replace the active classifier. Training parameters
can still affect old-map behavior, so validation and live checks remain necessary.

For the first pilot, the user explicitly authorized selection by run duration and
trimming the last five seconds. `select_duration.py --sessions SESSION ...
--minimum-seconds 12 --trim-end-seconds 5 --apply-user-selection` appends review
revisions recording this authorization as `user_directed_duration_trim`, with
`reviewed: false`. The exporter accepts this explicit rule separately from visual
approval and carries its provenance forward. It does not claim every retained
action was visually checked. The two 15-minute training sessions selected on
6 November contain 29 retained runs per map, approximately 17.9 minutes after
trimming. Separate validation recordings are still required for BC training.

To reuse classifier warnings, `import_guard_warnings.py --sessions SESSION ...`
creates unlabelled sessions in `ai_data/multimap_screen_states/sessions/` and prints
the existing label-tool commands. Native images are padded exactly to the live
1920x1080 guard framing (canvas y=87), without changing the BC recordings. Warning
bursts are sampled sparsely with neighboring context; source hashes, probabilities
and run ages are retained. Derived sessions inherit their source split; importing
the same source again or across splits is refused. Existing classifier examples
remain in place. No candidate is trained or promoted by this import. Use separately
recorded sessions to validate the new views rather than splitting neighboring
frames from one recording.

```powershell
& $py -B -s "$bc\review.py" list --min-seconds 5
```

Open the printed HTML file. It sorts longer candidates first and provides frame
scrubbing/playback plus nearby key events. Lower `--min-seconds` to inspect short
but useful difficult openings. The filter only affects the gallery; it does not
delete or approve anything.

For each selected run, copy its path from the gallery. Times are seconds relative
to its first saved image. Trim **before the mistaken action**, not just before the
death animation. Explicitly approve reviewed intervals, for example:

```powershell
& $py -B -s "$bc\review.py" approve "FULL_PATH_TO_run_0001" --segments "0.2:7.5" --reviewed
```

Multiple disjoint intervals use `--segments "0.2:3.0,4.0:7.5"`. Every interval
starts fresh history; disconnected pieces never become a continuous sequence.
`review.py reject "RUN_PATH"` adds a rejection revision. Reviews are append-only;
native recordings are never edited. Only the latest approved revision is exported.

## Export the exact policy format

CPU export can run alongside the separate screen-state classifier training:

```powershell
& $py -B -s -u "$bc\dataset.py" --checkpoint $checkpoint --reviewed-states-root ai_data/multimap_screen_states --allow-classifier-training --workers 2
```

This exception permits only the classifier trainer, not simultaneous PPO, BC training,
evaluation or recording. It uses two bounded CPU decoders and no GPU inference.
The reviewed-state option applies human labels to matching native frames after checking
source checksums and split identity. Pending classifier predictions are never used.
Only referenced frames are stored; compact indices preserve the exact observation.
Each completed run has a checksummed `run_record.json`. An interrupted export can be
continued with the same arguments plus `--output PRINTED_EXPORT_FOLDER --resume`;
changed selections, labels or checkpoint require a new export. Partial run files are
preserved, with a separate retry folder. `dataset.json` marks full completion.
Run `audit_export.py PRINTED_EXPORT_FOLDER` to verify arrays and create
`preparation_report.json` and `policy_input_gallery.html` before BC training.

Set `$checkpoint` to the checkpoint you choose **after** the running session ends.
This must be the current detector-off four-frame, 789-aux model, not an older
557/565-aux checkpoint.

```powershell
$checkpoint = "FULL_PATH_TO_CHOSEN_latest.zip"
& $py -B -s -u "$bc\dataset.py" --checkpoint $checkpoint --decision-hz 30
```

The export uses the checkpoint's road crop/resolution and live temporal selection:
four RGB frames at target ages `[0, 0.05, 0.25, 0.75]`, oldest first. It reconstructs
the existing 557 features, frame ages/validity, and 32 timestamped command slots.
Only prior commands enter an observation; its target action never enters its input.
Unavailable history uses the live encoder's padding/validity behavior. No opening
wait is added. Menu/uncertain frames cannot become valid history images.

Physical key-state polling is approximately 2 ms; this is **not** proof that Unity
accepted an input. Labels are actions between successive decision observations,
using only preceding pixels. A key during capture, multiple actions in one interval,
held keys over 130 ms, and gaps over 130 ms are rejected. NONE is derived only from
completed empty decision intervals. Host capture timestamps are not GPU presentation
timestamps. These conservative filters may reject fast human combinations rather
than silently inventing their order.

Each exported run includes `input_*.png`: several exact four-frame input strips,
oldest to newest, plus `samples.json`, `indices.npy`, `aux.npy`, and action counts.
Review these before BC. Whole recording sessions separate train and validation;
adjacent images or mirrored versions cannot enter both splits. Checksum validation
detects changed recordings/reviews/exports. `dataset.json` is written only after
successful completion; incomplete exports cannot be trained.

## BC, evaluation, then PPO

Set `$dataset` to the **folder containing** the printed `dataset.json`:

```powershell
$dataset = "FULL_PATH_TO_EXPORT_FOLDER"
& $py -B -s -u "$bc\train.py" --dataset $dataset --checkpoint $checkpoint --epochs 20 --minutes 60
```

Defaults: batch 16, Adam learning rate 0.0001, equal sampling of both maps, moderate
class weighting, and horizontal mirroring that swaps LEFT/RIGHT in targets and all
history features. `--no-mirror` disables augmentation. Validation reports per-map,
per-action recall as well as natural accuracy/loss. Early stopping uses held-out
map/class-balanced loss, not mostly-NONE accuracy. Five non-improving epochs stop
training. The minute limit is checked between updates; validation/saving adds time.

`bc_best.zip` is a new PPO-compatible checkpoint in a unique report folder. The
source is never overwritten. The source itself is the initial best candidate, so
if validation never improves, no degraded BC epoch is selected. The best saved
candidate is retained if a later epoch fails. PPO optimizer moments are reset
after the separate BC optimizer; model shape and observation contract are preserved.
Training and validation label accuracy do not establish gameplay improvement.

```powershell
$bcCheckpoint = "FULL_PATH_TO_BC_REPORT\bc_best.zip"
& $py -B -s -u .\development\experiments\multimap_classifier\evaluate_both.py --checkpoint $bcCheckpoint --episodes 30
```

Compare with the source checkpoint on both maps, using the same evaluation settings.
Then resume standard multimap PPO, dashboard off and graph at the end:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\development\experiments\multimap_classifier\resume.ps1 -Checkpoint $bcCheckpoint -Minutes 120 -MaxRestarts 3
```

There is **no BC loss or demonstration replay during that PPO training**. BC supplies
a starting policy; it is not a permanent imitation constraint. Keep the source and
BC candidate so either can be continued independently.

## Offline validation

```powershell
& $py -B -s -m unittest discover -s development/staged/multimap_bc_20261106 -p "test_*.py"
```

Twenty-seven BC Python tests cover the resumable label queue, explicit duration selection, guard-image import,
launcher detection, WASD equivalence, lossless PNG output,
passive late-run classifier diagnostics as well as causal labels, temporal selection/padding, segment resets,
invalid/held input rejection, no menu context, mirroring, explicit review/session
separation, bounded asynchronous writes, native export/checksums, and a small CPU
BC update/save/reload with unchanged critic-head weights. This does not substitute
for the first live recording check or real gameplay evaluation.
