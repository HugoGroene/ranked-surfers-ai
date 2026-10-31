# Startup camera recording review

The interrupted tool session left a usable recording:
`startup_camera_20261031_192004/frames.json`, 114 frames, error=null.
No repeat live run was needed. Capture completion timestamps are relative to
guarded reset returning, not Unity render timestamps.

Visually reviewed: text remains at frame 68 (-331.5 ms); frame 70 (-146.3 ms)
has no Find Ranked Match text but still shows the opening camera angle.
Frame 80 (+419.1 ms) remains in the opening animation; frame 90 (+885.6 ms)
is turning toward the tracks. Frame 93 (+1038.9 ms) is approximately aligned
with the track and shows 0 m. Frame 95 (+1135.2 ms) shows 2 m; frame 97
(+1234.3 ms) shows the forward-running view. Frame 100 (+1369.5 ms) shows 5 m.

Thus the camera transition extends roughly 1.0–1.25 seconds beyond guarded
reset completion, and roughly 1.2–1.6 seconds beyond text disappearance in
this single observed restart. These are visual estimates, not an input-readiness
measurement or a guarantee across workloads. A 1.5-second diagnostic lead-in
is conservative but not clearly excessive. Do not impose this delay on training
based on one sample. The next action probe can use 1.3 seconds after reset and
must verify its saved pre-action frames. Controls may become effective earlier
than camera settling; this recording cannot establish that.
