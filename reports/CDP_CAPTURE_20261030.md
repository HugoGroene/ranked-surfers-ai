# CDP capture investigation - 30 October 2026

Status: optional prototype, not a six-worker performance fix. Existing Playwright
capture remains the default. No long experiment was started or existing model
overwritten. All short policy tests used a separate output directory and the
same preserved canvas checkpoint:

`reports/v612_canvas_diag_30m_20261030_024112/latest.zip`

SHA256 `779d295ec4fe04e569511987c3867ac8d68cf8a22eb12c547f028c591763907b`.
Both completed six-worker reports independently confirm source_unchanged=true.

## Capture measurements

The paired local Practice test in `cdp_capture_paired_20261030_01/report.json`
measured approximately 50.7ms median for existing screenshots and 49.8ms for
direct CDP. Removing Playwright's bounding-box work did not materially help.

The initial demand-only stream measured about 33ms median. Allowing one frame
ahead while inference/inputs execute, but rejecting frames older than the next
request, reduced the isolated median to 16.88ms and p95 to 18.83ms over 295 reads
in five seconds (`cdp_prefetch_stream_20261030_01/report.json`). The saved probe
completed successfully; a pasted UI transcript incorrectly labeled it status 1.
These are capture-call timings, not policy throughput or Unity response times.

## Policy timing

Times below are per-worker screenshot-request to actual command issue, excluding
NONE actions. They do not measure when Unity consumes the key or changes state.

| Path / report directory | Median ms | p95 ms | Optimized transitions/s |
|---|---:|---:|---:|
| Original visible MSS: `visible_original_reaction_20261030_01` | 27.6 | 53.7 | Frozen policy, no updates |
| One screenshot worker: `reaction_one_metrics_only_20261030_01` | 82.8 | 175.0 | Short diagnostic |
| One stream worker: `stream_policy_one_20261030_01` | 49.9 | 95.8 | 8.38 |
| Six warmed screenshot workers: `reaction_six_warmup_20261030_01` | 89-97 | 188-192 | About 19.6 |
| Six stream workers: `stream_policy_six_20261030_04` | 128-133 | 157-165 | 18.48 |
| Six stream workers, OpenCV one thread: `stream_policy_six_cv1_20261030_01` | 129-133 | 154-166 | 17.75 |

The visible baseline uses legacy framing and a frozen-policy measurement harness;
headless tests use canvas_v1 and the settled collector. Workloads, episode
trajectories and sampled actions differ. The one-worker screenshot test also had
a preview toggled off late; it is not an exact controlled backend comparison.
Do not treat the observed differences as causal proof or learning improvement.
The visible baseline's MSS grab alone was 13.6ms median / 16.0ms p95.

Each completed six-worker stream test optimized 1536 transitions in two shared
updates, all six settled-learning audits passed, no partial transitions were
discarded and no worker hard-stopped. No diagnostic checkpoint was promoted.
Stream 04 used 3722 stream captures and zero independent-surface fallbacks.

Worker 0 in stream 04 measured 84.3ms median from completed policy capture to
command, versus 9.3ms from the completed guard capture. Request-to-command was
131.6ms for policy and 57.1ms for guard. A new safety image can therefore accompany
an action sampled from an older policy image. Faster image delivery alone has
not solved policy/guard alignment. The 20ms prepared-reading bound and 100ms
resync freshness limit were not raised.

The OpenCV thread experiment tested a plausible contention hypothesis: this
machine defaults to 16 OpenCV threads per process. Restricting stream workers
to one thread did not improve reaction latency. That change was reverted.
CPU/RAM headroom by itself does not establish the cause of the remaining delay.

## Failed attempts and bounded menu recovery

Preserved `stream_policy_six_20261030_01` and `_02` stopped before any update on
real MENU transitions that exhausted the original six-reading guard window.
A longer sleep between ambiguous readings did not fix this and was reverted.
`_03` optimized one shared update before another MENU window exhausted its budget.
None of these failures is counted as successful throughput or silently discarded.

The final stream-only extension checks up to six more passive frames only for
a nearly corroborated MENU tail. It uses the existing terminal confirmation
rules on the newest six readings and can return TERMINAL or ABORT, never PLAY.
Conflicting gameplay or non-MENU evidence aborts. This recovered an actual MENU
transition in `_04`, whose two updates then completed. It changes the observation
budget, not the terminal thresholds, rewards or action mapping.

## Implementation and validation

- `tools/ppo_cdp_capture.py`: verified full-viewport geometry and direct rendered
  surface screenshots with JPEG quality 90.
- `tools/ppo_cdp_stream.py`: one latest-frame slot, acknowledged delivery,
  validated geometry/clocks, new presentation after each request, maximum 50ms
  stream-frame age on capture return, independent screenshot after a 150ms wait.
- `tools/ppo_headless_env.py`: explicit backend selection, existing decoded shape
  and geometry guards, separate raw guard image, stream lifecycle on reload.
- `tools/ppo_reaction_metrics.py`: separate browser-frame-to-command metric when
  validated stream metadata is available. Browser compositor time is not Unity
  physics time or an input-response measurement.
- `tools/ppo_stream_menu_tail.py`: bounded MENU-only transition confirmation.
- `tools/check_ppo_stream_pixels.py`: synthetic full-canvas/HTML Pause-overlay
  comparison, saved in `stream_pixels_20261030_01/result.json`. Image shape
  945x1920x3; both whole-image and overlay mean absolute differences were 0.
- Guard BGRA composition is pixel-identical for legacy and canvas_v1 profiles.
- 149 focused regression tests passed across capture, clock/geometry rejection,
  stale-frame rejection, independent static fallback, menu-only confirmation,
  framing, runtime, settled/parallel learning, rewards, pauses, browser lifecycle,
  time limits, freshness retries and telemetry helpers.

Remaining limits: static-surface fallback and 300-second stream reload are not
live-validated. Short runs do not establish long-run memory stability, improved
survival, complete timing equivalence or faster learning. No driver/display
settings, model architecture, PPO hyperparameters or reward definitions changed.

## Next targeted investigation

Instrument policy packing, inference, environment handoff and guard recapture
separately under the same six-worker workload. Use that evidence to remove the
old-policy/new-guard gap without relaxing safety or silently resampling an action
whose PPO log probability belongs to another observation. Keep streaming opt-in
until a matched comparison shows an actual reaction improvement. More long
training is not justified by the isolated 17ms capture result.
