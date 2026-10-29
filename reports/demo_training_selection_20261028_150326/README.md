# Trimmed demonstration selection ? approved for first BC experiment

24 new recordings were reviewed. The initial length cutoff was 20 seconds; 11 runs met it. Two were excluded because a start animation reappears after initial gameplay, suggesting mixed attempts. The remaining nine have conservative leading/end trims applied in their candidate manifests. The user subsequently explicitly approved all nine for the first actor-only BC experiment. That experiment has completed; source checkpoint and raw replay files remain intact.

Raw frames, auxiliary arrays, labels, events and videos are unchanged; all raw-file SHA-256 hashes were verified after curation. Original manifest bytes are backed up in `original_manifests/`. `selection.json` records exact half-open intervals and hashes.

## Selected windows

End points below are approximate display values; actual metadata uses an exact captured-frame timestamp to prevent a retained label interval crossing the cut. Start time is 2.15 seconds for each run, excluding the opening transition while leaving sufficient fresh frame history.

| Run suffix | Original duration | Selected interval | LEFT | RIGHT | Ending review |
| --- | ---: | --- | ---: | ---: | --- |
| 145438_471297 | 29.88 s | 2.15?27.800 s | 6 | 7 | [frames](demo_20261028_145438_471297_ending.jpg) |
| 145501_178756 | 22.04 s | 2.15?19.943 s | 3 | 4 | [frames](demo_20261028_145501_178756_ending.jpg) |
| 145548_721806 | 32.51 s | 2.15?25.798 s | 3 | 3 | [frames](demo_20261028_145548_721806_ending.jpg) |
| 145706_224680 | 27.10 s | 2.15?25.852 s | 3 | 2 | [frames](demo_20261028_145706_224680_ending.jpg) |
| 145805_464929 | 22.50 s | 2.15?19.896 s | 3 | 2 | [frames](demo_20261028_145805_464929_ending.jpg) |
| 145830_927998 | 24.95 s | 2.15?22.842 s | 6 | 6 | [frames](demo_20261028_145830_927998_ending.jpg) |
| 145918_817660 | 29.35 s | 2.15?27.243 s | 5 | 4 | [frames](demo_20261028_145918_817660_ending.jpg) |
| 150100_323415 | 37.13 s | 2.15?35.661 s | 6 | 5 | [frames](demo_20261028_150100_323415_ending.jpg) |
| 150144_472981 | 37.25 s | 2.15?29.946 s | 6 | 5 | [frames](demo_20261028_150144_472981_ending.jpg) |

## Basis and limits of the cuts

- 145438: stop before the final late lane-change/rapid-input sequence.
- 145501: stop before the final approach and dense JUMP/ROLL input sequence.
- 145548: cut farther back, before late chase/collision evidence rather than merely removing the last 0.7 seconds.
- 145706: stop before the final jump/right move toward the late obstacle sequence.
- 145805: stop before the final obstacle/rapid-input sequence.
- 145830: stop before the final approach that ends at a train/start transition.
- 145918: stop before the final right/right sequence and visible close chase/collision evidence.
- 150100: stop before the last approach and final rapid JUMP/ROLL inputs.
- 150144: stop before the 736 ms capture gap at 31.496 seconds; this also removes the entire final failing sequence.

These are conservative selections based on contact sheets, dense final-seconds sequences and recorded action timing. They remove visible or suspected failure regions, but sampled visual review cannot prove every retained decision is optimal. The 20-second length rule is a selection preference, not a quality guarantee. No damage/death detector or ground-truth Unity state was used.

## Excluded long recordings

- `demo_20261028_145631_029216`: gameplay is followed by the opening/start animation around 2.65 seconds. See [opening frames](demo_20261028_145631_029216_opening.jpg).
- `demo_20261028_150006_469386`: gameplay is followed by the opening/start animation around 2.84 seconds. See [opening frames](demo_20261028_150006_469386_opening.jpg).

Keeping a later portion of a possible merged run could also retain misleading auxiliary episode/action history, so both remain excluded from this first experiment. Other new runs below 20 seconds and all six old diagnostic runs remain untouched candidates.

## Dataset and BC split preview

Selected duration: 215.631 seconds; 4311 valid observations. Actions: NONE=4168, LEFT=41, RIGHT=38, JUMP=46, ROLL=18.

With the existing seed 77, holdout fraction 0.20 and NONE fraction 0.35, the nine selected runs split into seven training and two holdout runs. After NONE subsampling there are 183 training examples: NONE=64, LEFT=35, RIGHT=33, JUMP=37, ROLL=14. Holdout has LEFT=6 and RIGHT=5, so rare-action accuracy estimates will remain imprecise. This is a pilot dataset, not strong evidence of generalization.

`split_preview.json` records the exact split simulation. It directly reads candidate metadata for diagnostics; normal BC loading still excludes these candidates.

## Approval boundary

`CLAUDE.md` says: "Never auto-approve demonstrations." The user explicitly approved all nine selected trimmed runs for actor-only BC. Exactly those nine manifest statuses are now approved. Exact frame-aligned bounds were restored after detecting that the curation helper rounds time values to milliseconds. Raw replay files remain preserved; the pre-approval manifests are backed up in `pre_bc_approval_manifests/`.

No training should use the excluded runs unless separately reviewed and explicitly approved.


## First BC result

Output: `ai_data/demonstrations_v13/bc_runs/bc_20261028_152607/model.zip`.
Three epochs, batch size 12, learning rate 0.00002, seed 77. Seven training runs,
two whole-run holdouts; 183 subsampled training observations. The exact aligned
bounds yield 946 holdout observations (922 NONE, 6 LEFT, 5 RIGHT, 9 JUMP, 4 ROLL).
The earlier `split_preview.json` predates restoration of exact boundaries; the
actual BC report and per-action diagnostics are authoritative.

Held-out cross-entropy: 1.11905 -> 1.09850; accuracy: 48.84% -> 50.00%.
Deterministic LEFT/RIGHT recall remains zero on both training and held-out demo
observations. Correct-action probability increased slightly for lateral labels,
but no lateral action became the argmax. This is modest imitation-loss progress,
not evidence of better lane behavior or survival. A subsequent 100-episode gameplay evaluation scored 4.8276 s mean versus the source
4.2970 s, with zero lateral actions and zero guard aborts. See
[the live comparison](../bc_pilot_evaluation_20261028_153047/README.md).

Actor-only verification compares all saved policy tensors: four actor tensors
changed; all 40 others remain bitwise identical. Source checkpoint SHA-256 is
unchanged. The output directory contains the BC report, per-action diagnostics,
actor-only verification and experiment provenance. Neither the PPO model nor
its rewards/architecture were changed by source edits for this experiment.
