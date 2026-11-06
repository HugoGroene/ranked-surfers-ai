# Offscreen FPS comparison

Run `batch.py --execute` with the local server available and no other training
running. Two one-worker smoke cases precede four four-worker performance cases
(60/30/30/60 FPS, three active minutes each). All start from the same checkpoint;
updates and outputs are isolated. Production host/defaults are unchanged.

The wrapper must remain the multiprocessing main file. Do not replace its normal
import with runpy: Windows spawn then re-enters run_training.py and loses the
host override. Offline regression: set RS_OFFSCREEN_TEST_FPS to 30 or 60 and run
`run_case.py --check-spawn`. This imports the actual worker adapter in a spawned
process without starting a game and verifies the selected host and inherited cap.

The first attempt (offscreen_fps_20261104_164429_044255) completed its 60-FPS smoke
training but failed host-cap verification. Retain it as a training smoke result,
not a validated renderer-cap comparison. The corrected suite deliberately runs
the smoke again with verified host settings.

New-map classifier review, 4 November: the sampled recording
new_map_validation_20261104_163002_899631 contains clear menu images 00012,
00072, and 00096 labelled GAMEPLAY. See classifier_review_contact.jpg in that
report. This is evidence of menu failures, not a complete accuracy audit.
Update/validate on both maps before enabling new-map automated training; also
validate menu templates and separate pause/resume handling. No classifier weights
were changed by this experiment.
