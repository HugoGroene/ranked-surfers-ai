# Shared memory and preparation work - 31 October 2026

The optional Electron path now supports Windows shared-memory CPU frames through
Koffi 3.3.6 prebuilt native FFI. Playwright remains the default; Electron socket
transport is retained. No Chromium rebuild, GPU texture integration or model/
reward/observation/threshold change was made.

## Four-worker live checks

All used the preserved interrupted overnight checkpoint (SHA256
804d919442f748cdd9b75a509a0968524ae8e1a67dc54fef1f41737017f20d93),
canvas_v1, rollout 128 per worker, LR .0003, five epochs, no previews, telemetry,
profiling and 1024 transitions/two shared updates. All completed, passed all
four learning audits and reported source_unchanged=true. Diagnostic policies
are separate unpromoted outputs. These are short stochastic comparisons,
not learning-quality or multi-hour stability experiments.

| Path | Pooled command median / p95 ms | Mean ms | Optimized transitions/s |
|---|---:|---:|---:|
| Earlier matched Playwright | 80.15 / 161.83 | 88.95 | 20.62 |
| Tuned Electron socket, repeat | 54.42 / 115.98 | 70.30 | 21.11 |
| Shared memory | 37.65 / 81.24 | 46.58 | 27.39 |
| Shared memory + preparation trial (reverted) | 39.09 / 81.88 | 48.90 | 26.23 |
| Shared memory + exact-frame state cache | 38.51 / 83.27 | 48.27 | 26.67 |

Report directories are pipeline_profile_electron_shared_four_20261031_01,
pipeline_profile_electron_shared_fast_four_20261031_01 and
pipeline_profile_electron_shared_cache_four_20261031_01. Corresponding
pipeline_analysis_* directories contain complete summary.json and decisions.

Shared-memory runs covered guarded restarts, verified freezes, pause/resume,
fresh-stack refresh and optimizer coverage. CPU frame publication/reading averaged
about .60/.94ms in the initial shared-memory profile; total source delivery
averaged 14.60ms, versus 30.18ms in the previous tuned socket repeat. This is
not zero-copy: producer copies into the map, consumer copies into owned pixels.
Actual GPU presentation, pure GPU readback and Unity input consumption remain
unmeasured. The command-latency endpoint is command issue.

The cache run recorded 43 hits across four workers (11/10/11/11), with each new
capture still classified. These avoid duplicated model calls within the same
capture, particularly during resynchronization. Ordinary decision latency did
not clearly improve over shared memory alone; the cache remains opt-in.

Frame-age acceptance stayed at 50ms, prepared-reading validity at 20ms and
resync freshness at 100ms. Guard source changes still occurred on roughly
33-37% of movement commands in the shared-memory checks. Tail variability and
the preparation/handoff costs have not been eliminated. Peak RAM was 86.1%
without the cache and 84.8% with it; this is not proof of a memory saving or
evidence that six workers are now feasible.

## Ownership and validation

One per-renderer, randomly named Local-session mapping holds one frame. The
mapping identifier is separate from socket authentication. Producer invalidates
the header, copies pixels, then publishes a validated header before sending
socket metadata. Python opens the existing mapping read-only, validates magic,
version, dimensions, serial and byte count, copies pixels, then checks that the
header stayed unchanged. Only the next capture request permits overwrite.
Previous returned frames remain valid. Mapping close is idempotent; normal
process shutdown releases the handles. No bulk pixels travel over the socket.

The initial and final hardened capture-only checks are preserved under
shared_frame_capture_20261031_01 and _02. Both passed exact synthetic pixel and
HTML-overlay parity, BGRA orientation, static repaint, trusted synthetic input,
actual NVIDIA WebGL loading and clean renderer exit. Capture medians were
16.65/16.74ms with p95 19.34/18.59ms. The final check exercised the invalidation
marker and distinct random mapping identifier. No model ran in those probes.

Tests cover owned-frame retention, rejected changed headers/serials/geometry,
cache invalidation on image/capture/model/confidence changes, and owned
probability dictionaries. Existing protocol, learning and guard regressions
remain part of the focused suite.

## Reverted experiment and remaining target

A trial resized BGRA before converting channels for the screen classifier.
Random inputs/crops produced identical classifier input pixels, but the live
profile showed no preparation or latency improvement. Its source/flag was
removed; rs_v4_screen_states.py retains its original implementation. The
experiment directory is preserved and is not counted as an adopted optimization.

GPU texture sharing is deferred: most of the measured bulk-transfer cost was
removed without it. Further ordinary-decision work should target classifier/
preparation and collector handoff variability while preserving frame lineage,
sampling probabilities and guard freshness. Short tests do not justify widening
freshness limits or starting another unattended run without monitored validation.

Setup/flags: [shared-memory guide](../README_PPO_SHARED_MEMORY.md).

Final validation: 186 focused tests pass, including rejected changing shared
headers, owned pixel retention, exact-frame cache invalidation, pipeline profiling
invariance and existing runtime/learning regressions. A mocked confirmation-clock
fixture was corrected to provide positive elapsed time on coarse Windows clock
ticks; production clock validation was not changed. JavaScript syntax and
whitespace checks pass. Source checkpoint hashes remain unchanged.
