# Six-worker canvas_v1 live verification

Requested approximately three minutes with live grid. Completed five shared
updates, 3840 collected/optimized transitions, 640 per worker, no discarded partial
transitions and no guard hard stops. All six worker ledgers PASS_SETTLED_LEARNING,
including four post-update passive resync transitions per worker and guarded deaths.
Wall-time limit is checked after a completed shared update; active runtime was
208.666s rather than exactly 180s. Throughput: 18.403 optimized transitions/s.

Whole canvas verified from saved metadata and policy image: 1920x945 native,
512x288 letterboxed policy, no Chrome/taskbar pixels. Guard source remains separate.
Checkpoint framing_profile=canvas_v1; ZIP CRC passed. Source checkpoint hash
unchanged. latest.zip is experimental, not promoted on survival performance.

Sampled resources (includes loading and updates): CPU mean 43.8%, peak 99.25%;
GPU mean 38.4%, peak 85%; VRAM peak 7343 MiB (7.17 GiB); system RAM peak 92.4%.
The RAM peak leaves limited headroom, despite the shorter six-worker test's 85.6%.
No claim of long-run memory stability or improved model quality.

Grid HTTP returned 200 with six tiles; runtime/phase/count/frame-age displayed.
Three preview HTTP request threads logged ConnectionAbortedError WinError 10053
on client disconnect. These did not stop PPO; overall status COMPLETE.
All workers and preview server closed at completion. No physical visible-browser
calibration/evaluation was performed. Virtual display driver remains disabled.
