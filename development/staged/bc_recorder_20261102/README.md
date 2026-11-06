# Staged native BC recorder

Prepared while the road-vision batch runs. **Do not launch recording alongside
that batch.** No active trainer, experimental runner, capture backend, reward,
architecture, checkpoint or recording was edited. This directory is outside the
batch's source-hash collection. `active_sources_before.json` records that baseline.

## Implemented

- An isolated snapshot of the existing manual Practice recorder. It uses the
  existing physical-input alignment, focus/geometry checks, screen-state guard,
  130 ms label validity bound, autosaving of completed candidates and preview videos.
- Detector forced off and clean `canvas_v1` framing. The launcher exposes capture
  FPS (5–40, default 20) and maximum run length (default 45 seconds).
- **Lossless native canvas PNGs** saved per frame, with timestamps and checksums.
  The old 512×288 replay remains alongside them for compatibility; it is NOT the
  source of the new high-resolution exports. The existing internal field name
  `canvas_jpeg` now carries PNG bytes in this isolated snapshot only.
- Offline exports at adjustable square sizes, including 128, 192, 256 and 384,
  using the exact road experiment crop/letterbox transform. Crops: full,
  road_wide (15% removed from each side), road_center (25% removed from each side).
- An HTML catalogue sorted by duration, with full-canvas video links and suggested
  tail trimming. Configurable policy contact sheets can be generated before approval.
- Explicit visual-review acknowledgement and manually chosen intervals. Reviews
  are append-only revisions; originals remain unchanged. G/T in the recording
  console cannot approve these runs. Long survival never automatically approves one.
- Export keeps physical events, chronological frame timestamps, labels, per-label
  validity, detector-off auxiliary context, interval IDs and training mask.
  Invalid labels are not made valid by trimming. Frames are retained chronologically
  so later history construction need not join unrelated safe intervals.
- Checksums detect changed source frames or labels. Existing exports are never
  overwritten, and `export.json` is written last as a completion marker.

## What is deliberately not selected yet

The final resolution, encoder, history spacing and training rate await the batch.
Native frames allow choosing those later without re-recording at every resolution.
Exports contain single frames, not a preselected stack. A downstream BC loader must
reconstruct the chosen temporal contract **within a single approved interval**,
preserve missing-history flags/ages, and split validation by whole recording/session.
`native-bc-export-v1` is intentionally distinct from the current legacy BC loader;
do not point that loader at these exports. Wiring the winning policy's BC dataset
adapter and checking train/evaluation observation parity remains before BC training.

No weights are trained, loaded or changed by these tools. No live recording test has
been run for this version. Lossless encoding and disk storage cost more than JPEG;
we must measure achieved frame intervals, label coverage, RAM and disk use in a short
human recording before collecting hours. FPS is a target, not a promise. Completed
runs autosave; this is not a guarantee against power loss during an active run.

## After the batch finishes: short human recording first

Run from the repository root with the existing local server available on port 8000.
Start with one or two short episodes; inspect the saved preview and label diagnostics
before collecting a larger dataset. This command opens a visible game for YOU to play.

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$bc = 'development\staged\bc_recorder_20261102'
& $py -B -s -u "$bc\record.py" --fps 20 --max-run-seconds 45
```

Play local Practice manually; restart with R when needed. Completed runs save under
`ai_data/demonstrations_native_v1`. Use Q + Enter in the recorder terminal to finish
cleanly. The existing P + Enter preview remains available for completed runs.

## Review, trim and export (offline)

Use a new catalogue filename each time; the tools refuse to overwrite artifacts.

```powershell
& $py -B -s "$bc\dataset.py" catalogue ai_data/demonstrations_native_v1 `
    --output ai_data/demonstrations_native_v1/review_01.html --trim-end 0.7

# Set this to the actual saved run folder from the recorder log.
$demo = 'ai_data/demonstrations_native_v1/REPLACE_WITH_RUN_FOLDER'
& $py -B -s "$bc\dataset.py" preview $demo `
    --crop road_wide --size 256 --output "$demo\policy256_preview.jpg"

# EXAMPLE intervals only: replace after watching the recording, including its ending.
& $py -B -s "$bc\dataset.py" review $demo --status approved `
    --reviewed --segments '0.4:8.2' --note 'Reviewed full video; startup and mistake trimmed'

& $py -B -s "$bc\dataset.py" export $demo --crop road_wide --size 256 `
    --output "$demo\exports\road256_v1"
```

Multiple safe intervals use `--segments '0.4:8.2,9.0:12.1'`. Trimming suggestions do
not detect the first mistake or the startup camera pan. Inspect both manually. To
withdraw approval, use `review $demo --status rejected`; the previous revision stays
on disk and new exports are blocked. Already-created exports retain their provenance
and must be excluded from future training if that approval is withdrawn.

## Offline verification

```powershell
& $py -B -s -m unittest discover -s $bc -p test_staged_recorder.py -v
```

Tests cover lossless pixels, crop parity at four resolutions, explicit approval,
trim bounds, unchanged masters, invalid-label masks, source tampering, rejection,
candidate previews, export overwrite refusal, snapshot provenance and compilation.

## Next steps once results arrive

1. Select a small shortlist from both seeds' learning results and latency, rather
   than call the best observed mean a universal optimum.
2. Confirm the strongest close candidates with longer runs/additional seeds if
   their uncertainty overlaps; then benchmark useful worker counts on the winner.
3. Live-test this recorder's capture geometry, physical labels, full/policy preview
   parity and recording overhead. Correct any failures before recording hours.
4. Record varied good no-coin gameplay, including useful lane changes and recovery,
   then inspect and trim. Keep whole sessions held out for BC validation.
5. Add the matching native-export BC loader and compare actor-only BC continuation
   on a copied checkpoint, keeping PPO-only results as the baseline.

The video author describes a couple of hours of play and horizontal mirroring,
not an exact published example count in the transcript/repository inspected:
https://github.com/nikp06/subwAI . More frames from the same situation are not the
same as more independent demonstrations.
