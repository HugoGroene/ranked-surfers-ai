# Policy, detector and BC framing audit

Read-only model/data investigation, 30 October 2026. The audit generated new
comparison files only; no capture defaults, checkpoints or demonstrations changed.

## Confirmed paths

| Path | Native image before letterbox | Content height at 512x288 | Top padding |
|---|---|---:|---:|
| Historical desktop PPO | 1920x983, desktop rows 59:1042 | 262 | 13 |
| Recorded BC | 1920x860, canvas rows 52:912 | 229 | 29 |
| Current headless PPO | 1904x852, canvas rows 52:904 | 229 | 29 |
| Candidate whole headless canvas | 1904x937 | 252 | 18 |

Every path letterboxes rather than stretching. Fractional DEFAULT_CROP is
(0,.055,1,.965), but coordinate origins differ. The headless canvas image is
inserted into a black 1920x1080 guard image at (8,87); the background policy path
extracts that canvas again before applying fractions. Padding does not restore
the historical policy framing. Exact reconstruction of a saved headless policy
PNG is checked in audit.json (actual_saved_headless_equal).

The same-frame comparison uses a saved headless gameplay JPEG for all three
variants. Its desktop variant is a simulation with black padding, not historical
Chrome/taskbar pixels. historical_policy_reconstructed.png independently applies
the current resize function to a genuine saved legacy detector-source JPEG;
it shows browser chrome. This reconstruction is not an archived original tensor.

The full canvas retains more top/bottom gameplay than the current headless/BC
fractional crop. In these width-limited examples object scale is largely retained;
vertical position, visible content, padding and normalized detector coordinates
change. Different browser canvas aspect ratios may also change the game's camera.
It would be wrong to describe every difference as a large zoom change.

## Demonstrations and recovery

Scanned 321 manifests: 151 approved, 170 candidate. All describe calibrated
region (0,87,1920,945). The recorder applies DEFAULT_CROP within that region;
BC trains on stored RGB frames.npz arrays, never preview.mp4. The sampled pilot
stored frame 100 confirms the old 512x288 input with canvas-based cropping.

The preview writer emits one video frame per sample in source order, using a
constant playback rate derived from median timing. The inspected pilot has 597
samples/video frames, but video is 960x472 rather than source 1920x945. The writer
also burns time and ACTION NAME into a bottom-left rectangle. These pixels must
not enter BC training: that would expose the target label and lose original pixels.

Existing tensors cannot recover cropped-away content. They remain usable as a
versioned legacy dataset; they should not be relabeled as whole-canvas data.
Some overlap-region remapping is possible, but cannot exactly reconstruct the
missing source. A separate derived dataset could use videos only after explicit
overlay exclusion, frame alignment and quality checks; it is lossy and not an
exact conversion. Preserve actions/timestamps/curation, and recompute relevant
detector auxiliary features for any genuinely changed geometry. No conversion
was performed in this audit.

## Detector and model implications

Runtime PPO and detector share the same native crop. Detector boxes/polygons
are normalized to that image's width/height, and downstream distance bands/features
therefore change with crop even when the detector finds the same object.

The detector checkpoint metadata points to an old export_round04_fixed/segment/
data.yaml under the original ranked-surfers-ai project, with no training dataset
images/labels in the current v5_hazards directory. Its actual training framing
was not verified here. Legacy runtime inference framing is confirmed; that does
not prove detector training images used it. Existing PPO training never updates
YOLO: any detector adaptation would require separate labeled validation/training.

PPO can continue from the existing weights because tensor shapes remain compatible,
but successful adaptation, its required training budget, and quality are unproven.
The previous headless throughput checks establish pipeline operation, not visual
equivalence, survival improvement or a promoted checkpoint.

## Recommended consistent profile

Use the entire calibrated game canvas, preserve aspect ratio and letterbox to
512x288 with one shared preprocessing function for PPO, detector and new BC.
Use a fixed canvas aspect ratio/viewport (historical visible canvas was 1920x945;
headless prototype is 1904x937, very close but not identical). For visible-browser
evaluation, calibrate the actual canvas; do not apply old desktop fractions first.
Reject layout/aspect changes instead of guessing. Record a framing version in
run/demo/checkpoint metadata, not merely tensor dimensions.

Do not globally apply this profile to screen-state classification, independent
death templates, pause references or raw guard frames: each has its own trained
or calibrated source contract and needs a separate compatibility check.

Before new long training: implement the shared versioned profile; compare
headless/visible inputs with identical canvas geometry; validate frozen detector
outputs against labeled/synchronized examples; run a short guarded frozen-policy
check; then continue a new experimental PPO branch from the preserved source.
No automatic retraining, threshold changes, BC dataset rewrite or promotion.
