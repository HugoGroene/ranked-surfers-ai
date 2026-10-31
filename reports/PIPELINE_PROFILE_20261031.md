# Four-worker pipeline profiles - 31 October 2026

The retained Playwright implementation and optional Electron renderer were
profiled sequentially from the same preserved overnight checkpoint, SHA256
804d919442f748cdd9b75a509a0968524ae8e1a67dc54fef1f41737017f20d93.
All runs used four workers, canvas_v1, 128 steps per worker, LR .0003, five epochs,
no previews, telemetry, two shared updates and 1024 total optimized transitions.
Each completed with all four settled-learning audits passing, no discarded
rollout transitions and an unchanged source. Diagnostic models were not promoted.

These are short, stochastic engineering comparisons. Episode trajectories,
sampled actions, reset frequency and contention differ; no learning-quality
conclusion follows. Profiling changes timing slightly, although synthetic real
SB3 tests confirm identical actions, rewards and final CPU weights with it on/off.

## Actual observation-request to command timing

Metrics are pooled across actual movement commands, excluding NONE and passive
transitions. The endpoint is command issue, not Unity receiving/applying input.

| Run suffix under reports | Commands | Mean / median / p95 ms | Optimized transitions/s | Peak RAM |
|---|---:|---:|---:|---:|
| pipeline_profile_playwright_four_20261031_01 | 871 | 88.95 / 80.15 / 161.83 | 20.62 | 80.3% |
| pipeline_profile_electron_four_20261031_01 | 864 | 91.23 / 102.32 / 129.62 | 19.57 | 85.1% |
| pipeline_profile_electron_buffer_four_20261031_01 | 865 | 75.10 / 56.76 / 120.05 | 21.28 | 85.0% |
| pipeline_profile_electron_buffer_four_20261031_02 | 876 | 70.30 / 54.42 / 115.98 | 21.11 | 85.1% |

Analysis directories are named pipeline_analysis_* with matching suffixes.
Each contains summary.json and per-command decisions.jsonl. Worker profile
metadata reported no dropped events. Absolute maxima remained 177-183ms in
the two receiver-tuned checks; the slow tail has not been eliminated.

## Where the wall time went

The following means partition the timeline without overlaps. Column components
sum to the mean total; they are not individual percentile values added together.

| Additive stage, mean ms | Playwright | Electron before tuning | Tuned Electron repeat |
|---|---:|---:|---:|
| Capture request to complete image reply | 55.48 | 34.35 | 30.18 |
| Reply to inherited capture completion | 8.41 | 3.60 | 3.20 |
| Capture completion to policy start | 12.62 | 16.65 | 15.19 |
| Policy input, forward and action-ready host time | 3.53 | 5.43 | 4.68 |
| Action ready to command: guard and handoff | 8.91 | 31.19 | 17.05 |
| Total | 88.95 | 91.23 | 70.30 |

Before tuning Electron delivered its image sooner but much more often needed
another guard capture: 543/864 commands (62.8%), versus 104/871 (11.9%) for
Playwright. The tuned checks reduced this to 379/865 (43.8%) and 325/876 (37.1%).
The prepared-reading limit remains 20ms. Several milliseconds of preparation and
inference variability can push a decision past that boundary, adding a whole
capture/classification cycle. This produces two latency populations and makes
the median especially sensitive to which population exceeds 50% of commands.
The large median change should not be interpreted as an equivalent reduction
in every frame's transfer time.

Child timings explain the costs but overlap the table above:

- Original Electron bitmap extraction averaged 1.04ms, not tens of milliseconds.
- Node request-to-paint averaged 11.59ms. This combines frame wait, rendering,
  readback and scheduling; pure GPU readback is **not isolated**.
- Python payload reading averaged 19.86ms before tuning; the broader endpoint
  transport/scheduling residual was 21.51ms. These include copies/scheduling,
  not pure network latency.
- The first tuned payload mean was 17.12ms, repeated at 16.64ms. The original
  total capture-call mean of 34.38ms fell to 31.04ms and 30.22ms.
- Original Electron packing averaged 12.30ms, including 2.40ms native-frame
  preparation and 8.18ms screen-state classification. Playwright packing
  averaged 9.67ms, with 1.60ms and 6.68ms respectively.
- Playwright's bounding-box request averaged 4.95ms, screenshot API 50.51ms,
  JPEG decoding 4.83ms and guard composition 2.24ms. The earlier direct-CDP
  experiment did not materially improve screenshot capture; this profile does
  not justify repeating that approach blindly.
- Host policy execution averaged about 3.5-5.4ms. CUDA stream intervals were
  similar in scale; they include stream contention/idle time, not exclusive
  kernel work. Model size is not the dominant measured cost.

In the slowest 5% of commands, guard/handoff remained large: mean 75.92ms for
Playwright, 56.22ms for original Electron and 51.63ms for the tuned repeat.
Tuned slow-tail mean total was 126.65ms, including 28.36ms between capture
completion and policy start. Future work should address both transfer and
preparation/handoff variability rather than only screenshot throughput.

## Receiver change

The CPython 3.10 StreamReader implementation pauses transport reading when its
buffer exceeds twice its limit, and resumes while readexactly waits for more
data. The default limit is 64KiB; a full frame here is 7257600 bytes. Electron
now uses a limit of one complete frame plus 4096 header bytes (7261696 bytes).
It does not request or retain a FIFO of multiple frames, change the decoded
pixels, or raise the validated payload size. A synthetic flow-control test
confirms identical bytes with fewer pause/resume operations.

Both tuned runs retained the 50ms paint-age acceptance limit, bounded stale
discards, 20ms prepared-reading limit and 100ms resync freshness limit. No
rewards, observations, actions, model architecture or PPO hyperparameters changed.
Playwright remains the default and fallback; Electron remains optional.

## RAM interpretation

The matched four-worker runs started at the same first-sample system RAM usage,
45.5%. Peaks were 80.3% for Playwright and 85.1% for Electron, about 1.5GiB apart
on this machine. Components were not attributed with a private-memory profiler.
Electron adds a Node host and raw offscreen bitmap/transport allocations, but
that does not establish which component owns all the extra memory.

Earlier six-worker Playwright first-sample RAM was 35.0%, versus 56.5% in the
later six-worker Electron attempt. These are startup samples, not clean idle
measurements. The six-worker RAM failure cannot be attributed entirely to the
renderer: system baseline also differed materially. The interrupted overnight
six-worker Playwright run itself peaked at 95.6% RAM.

An allocation check using the installed SB3 buffer API confirmed image storage
is already uint8: 216MiB per 128-step worker image buffer. Six worker image
buffers plus the learner's aggregate image buffer total about 2.53GiB before
packet copies, models, libraries, browsers or training tensors. This is shared
pipeline overhead, not an Electron-only cost; there is no float32 image-storage
mistake to fix on the evidence inspected.

## Next engineering target

Retain the small receiver change and investigate shared-memory transport of the
same validated CPU pixels before undertaking a GPU-texture integration. A small
Windows native addon could replace bulk socket copies without rebuilding
Chromium. Buffer ownership, publication/synchronization and stale-frame checks
must remain explicit. It would not automatically remove GPU readback.

Also locate avoidable copies/bookkeeping in preparation and handoff so a valid
prepared reading reaches dispatch within its existing 20ms limit more often.
Do not widen the limit or silently resample an action from another observation.
GPU shared textures remain a possible later experiment; these measurements do
not prove that pure GPU readback dominates or that a Chromium fork is necessary.

Validation: 181 focused tests passed, including profiling invariance, overlapping
interval reconciliation, raw transport/freshness, receiver flow control, menu
recovery and existing corrected/parallel learning paths. JavaScript syntax and
git whitespace checks also passed. All four profiling runs independently report
source_unchanged=true and four passing settled-learning audits.
