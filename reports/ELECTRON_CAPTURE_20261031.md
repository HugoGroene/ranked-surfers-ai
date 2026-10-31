# Electron raw-frame prototype - 31 October 2026

The existing Playwright screenshot implementation is preserved and remains the
default. Electron is opt-in with `--headless --headless-capture electron`.
The prototype uses GPU-capable offscreen rendering with raw CPU BGRA bitmaps,
not capturePage, JPEG/Base64 transport or shared GPU textures. No virtual
display or physical monitor setting was changed.

Runtime: Electron 45.2.1 / Chromium 156.0.8078.75, isolated under ignored
development/local/electron_worker. The official runtime ZIP matched SHA256
`7c342aee45d5b4ccafe5dfe9037028796b082735a576e21e2b4e2d8422ebb727`
from the pinned npm package checksums before extraction. The installer stalled;
the direct official download was resumed after its bounded timeout.

## Correctness and one-worker measurements

`electron_capture_20261031_04/result.json` passed lossless synthetic page parity
(maximum pixel difference 0), HTML Pause overlay inclusion, channel order and
orientation, static repaint, trusted synthetic ArrowLeft delivery, actual Unity
loading and capture, and clean renderer exit. WebGL identified the NVIDIA
4060 Ti through ANGLE D3D11. Ready-file GPU feature status is an early snapshot
and should not be interpreted as a definitive compositor configuration.
Raw capture took 29.82ms median / 34.45ms p95 over 193 reads in five seconds.

Live policy checks used the preserved interrupted overnight model:
`v612_canvas_6workers_5h_20261030_042039/latest.zip`, SHA256
`804d919442f748cdd9b75a509a0968524ae8e1a67dc54fef1f41737017f20d93`.
Both backends used canvas_v1, rollout 128, LR .0003, five epochs, no previews,
warmup and the same corrected collector. Each completed two updates/256
transitions, passed its settled-learning audit, dropped no transitions and
verified source_unchanged=true. Tests created separate, unpromoted checkpoints.

| One worker | Command samples | Request-to-command median / p95 / max ms | Optimized transitions/s |
|---|---:|---:|---:|
| Electron: electron_policy_one_20261031_01 | 196 | 44.42 / 62.94 / 173.64 | 10.17 |
| Playwright: electron_comparison_playwright_20261031_01 | 216 | 77.45 / 150.81 / 226.34 | 6.74 |

Both tests exercised guarded resets and verified pause/resume/four-frame refresh.
The Electron log also rejected a stale rendered Pause overlay before rebuilding
its gameplay stack. Policy capture-complete-to-command p95 was 19.36ms for
Electron versus 88.54ms for Playwright. Eight of 196 Electron command samples and
15 of 216 Playwright samples used a different guard capture source (request-time
delta >0.1ms). Some policy/guard gaps remain; the maximum is still significant.

These are short stochastic diagnostics, not controlled learning-quality tests.
There was one completed episode in Electron versus five in Playwright; reset
frequency contributes to throughput differences. Browser versions/rendering paths
also differ. Neither a faster learning rate nor improved survival is established.

## Six-worker attempts

`electron_policy_six_20261031_01`: stopped before any shared update on a frame
older than the new source's 50ms paint-callback-age bound. Peak RAM 99.6%, minimum
available RAM 0.119 GiB, peak sampled VRAM 4793 MiB.

Added up to two passive discard-and-recapture retries, preserving the 50ms
acceptance bound. Every retry demands a future paint; invalid clocks or geometry
still stop immediately. This does not resample PPO actions or change rewards.

`electron_policy_six_20261031_02`: stopped before any update after persistent
staleness; final rejected frame age 51.4ms. Peak RAM 99.0%, minimum available
0.347 GiB, peak sampled VRAM 5303 MiB. Only 16-30 command samples per worker were
collected, with medians 158-169ms and p95 218-328ms. Do not count this aborted
attempt as successful training throughput or a stable latency benchmark.

The user reported normal main-monitor browsing responsiveness. That does not
establish worker responsiveness; measured policy command latency was poor.
Memory pressure is a plausible contributor; page-fault causation was not traced.
Six workers are not suitable for unattended Electron training on this evidence.

Audio was heard in the first tests. Worker output is now muted at Electron's
webContents level; all six ready files in attempt 02 record audio_muted=true.
System volume and Unity audio processing were not changed.

## Four-worker checks

`electron_policy_four_20261031_01` completed one update/512 optimized transitions,
then stopped on a MENU animation exhausting the six-capture pre-action window.
The saved screenshot is a real menu; full-precision evidence includes a final
score .676606, below .68 despite the rounded log showing .68. RAM peaked at
84.2%, and no stale-frame retries occurred. This is distinct from the six-worker
memory/freshness failure.

Enabled the already-tested stream MENU-only extension for Electron too. It
requires a nearly corroborated MENU tail before bounded additional captures,
uses the original terminal decision rules, and can never authorize movement.
Actual-runtime regressions replay the full-precision scores and verify both
recovery without a key and safe refusal on gameplay disagreement.

`electron_policy_four_20261031_02` completed 1024 transitions/two shared updates
in 54.57 active seconds (18.76 optimized transitions/s). All four settled-learning
audits passed, with no discarded rollout transitions, no hard stops and an
unchanged source checkpoint. A live MENU tail recovered after one extra capture.

| Worker | Command samples | Policy request-to-command median / p95 / max ms | Stale-frame retries |
|---|---:|---:|---:|
| 0 | 206 | 99.17 / 130.96 / 195.32 | 11 |
| 1 | 225 | 97.38 / 125.19 / 185.53 | 0 |
| 2 | 214 | 96.58 / 128.39 / 194.67 | 0 |
| 3 | 209 | 100.94 / 127.38 / 198.81 | 0 |

RAM peaked at 87.2%, minimum available RAM was 3.95 GiB, and peak sampled VRAM
was 5542 MiB. All four workers were invisible and muted. Timing variability is
still material; stale captures are discarded, not accepted. Four workers have
not been tested for hours. Comparing this result with earlier six-worker
Playwright tests changes worker count, source policy and aggregate rollout size;
it is not a controlled learning-quality or backend-only comparison.

## Safety, failures and limits

The binary source has one demanded frame at a time, token-protected loopback
requests, bounded header/payload sizes, fixed geometry/channels, increasing
serials, callback-age checks, capture timeouts and explicit static-page repaints.
It preserves composited HTML and the existing raw safety-image geometry.
The callback timestamp is not a Unity tick, GPU swap or input-consumption time.

Initial capture attempts 01-03 are preserved. Attempt 01 hit an Electron/Node
argument-name conflict; attempts 02-03 hit Node's reserved socket.pending
property. Both were corrected. Uncaught app errors now log and exit instead of
opening a modal dialog. Attempt 04 and live policy checks then completed.

The earlier weak-template post-resume MENU recovery fix is also in the working
tree and regression-tested; it is independent of capture speed. No source model
or historical result was overwritten. Long-run memory stability, six-worker
operation, 300-second Electron reload and visible-browser policy transfer remain
unverified. Use the existing Playwright path for fallback.

Primary API references: [Electron offscreen rendering](https://www.electronjs.org/docs/latest/tutorial/offscreen-rendering),
[paint and invalidate](https://www.electronjs.org/docs/latest/api/web-contents#event-paint).

Final focused regression suite: 174 tests passed across raw transport, stale-frame
retry bounds, menu recovery, current guards/runtime, pause exclusion, settled and
parallel learning, capture framing, freshness, timing, browser lifecycle and
existing fallback helpers. No long training was launched by the assistant.
