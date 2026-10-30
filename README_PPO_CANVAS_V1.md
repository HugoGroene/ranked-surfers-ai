# Whole-canvas framing profile

`--framing-profile canvas_v1` explicitly selects whole-canvas preprocessing.
It is supported by `train_rs_v612_ppo.py`, `tools.train_ppo_parallel`,
`rs_v612_demo.py record`, and `rs_v612_demo_bc.py`. The default remains `legacy`
for historical reproduction; select canvas_v1 explicitly for the new branch.

The profile extracts only the calibrated game canvas, converts BGRA to RGB and
letterboxes to 512x288 using linear interpolation. No additional fractional crop.
The canonical canvas is 1920x945; aspect ratio must agree within 0.5%, with minimum
640x320 source size. Invalid geometry stops the path. Existing CLI custom crop,
resolution or interpolation overrides are rejected for canvas_v1 in the trainer.

Headless canvas_v1 uses viewport 1920x945. Its synthetic raw guard image remains
1920x1080 with canvas at (0,87); independent screen-state/menu/pause guards do not
receive the policy image. Visible-browser mode matches the browser canvas image
to the desktop once (similarity >=0.85), then uses that calibrated region. Layout
observers invalidate changes. It does not assume browser bars are a fixed size.
Persistent sessions reject changes of framing profile within a session.

The unchanged detector receives the same native full-canvas RGB as the policy's
resize source. It still provides class labels, boxes and segmentation polygons;
normalized coordinates feed auxiliary features. PPO can adjust its use of these
features, but never trains or automatically removes the detector. No detector
quality improvement or crop invariance is claimed.

## Example short shared training check

With the local server already running:

```powershell
& $py -B -s -u -m tools.train_ppo_parallel `
  --headless --displays 1 2 --framing-profile canvas_v1 `
  --checkpoint ai_data\demonstrations_v13\bc_runs\bc_sampled_batch_20261028_1740_v2\latest.zip `
  --canvas-policy-input --steps 512 --max-minutes 2 --worker-timeout 160 `
  --telemetry --preview-port 8001 --report reports\NEW_UNIQUE_CANVAS_CHECK
```

For visible-browser local evaluation, add `--framing-profile canvas_v1` to the
existing `train_rs_v612_ppo.py --mode eval` command and use the matching checkpoint.
The browser must remain visible, focused and unobstructed for desktop capture.
This profile does not authorize a different site or change the local-only guards.
Live visible-browser calibration is still outstanding as of this implementation.

New BC recording: add `--framing-profile canvas_v1` to `rs_v612_demo.py record`.
The manifest's capture metadata stores the profile. BC training with the same flag
selects only approved canvas_v1 recordings. Untagged historical recordings are
legacy; they are never relabeled or automatically mixed with the new profile.
Old BC tensors/videos remain untouched. See the framing audit for why preview
videos with action labels are unsuitable as direct clean training inputs.

New PPO/BC checkpoints store `framing_profile`; run configs/manifests also record
it. Standalone whole-canvas runs save exact native/policy PNGs and framing metadata
on their first observation. Resuming legacy weights into canvas_v1 is an explicit
experimental adaptation, not proof those weights are visually equivalent.
Use the matching explicit profile when evaluating; the flag is not auto-selected.

## Verification and limits

The broad offline suite passed 130 tests before two additional framing/reset tests
were added; the focused follow-up passed 30 tests including those additions.
Tests cover desktop/headless pixel parity, RGB conversion, aspect rejection,
raw guard preservation, framing evidence, BC profile separation and headless
time-limit reload routing. Existing demonstration recording tests also passed.

`reports/headless_canvas_v1_20261030_01` completed two shared updates, all 512
transitions optimized, both worker audits PASS_SETTLED_LEARNING, no discarded
transitions or safety stops. One worker needed a bounded startup reload; its
diagnostic showed about:blank iframe URLs and no page errors. Active time was
47.997s (10.67 optimized transitions/s), excluding loading. Source hash unchanged.
Checkpoint ZIP integrity passed and saved framing_profile equals canvas_v1.

No long training, detector retraining, old-demo rewrite or model promotion.
The new 300-second headless reload routing is unit-tested, not live time-limit
validated. Visible-browser pixel extraction is unit-tested but not live validated.
The smoke run does not measure survival improvement or eventual detector independence.
