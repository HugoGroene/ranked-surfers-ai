# Experimental Electron capture

This is an optional alternative renderer for the local Practice copy. Existing
Playwright, direct-CDP and stream paths remain available; Playwright stays the
default. One and four workers have passed short live checks. Explicit scaling
tests support up to six workers, but six have not passed freshness validation.
It does not install or enable virtual monitors or change the physical display.

Electron 45.2.1 uses GPU-accelerated offscreen rendering with a CPU bitmap output.
The entire composited page, including HTML overlays, is delivered as raw BGRA
over an authenticated loopback binary socket. There is no JPEG/Base64 stage and
no pixel data in JSON. A request waits for a future paint; no image FIFO or cached
frame is returned. Static pages are explicitly invalidated for an independent
repaint after 80ms; absence of paint after 500ms stops capture. Byte counts,
geometry, channels, increasing serials and a maximum 50ms paint-callback age are
checked before pixels enter the existing raw-image guards. A stale frame can be
discarded and independently recaptured twice; persistent staleness stops safely,
and clock/geometry errors do not retry. A paint callback
timestamp does not establish GPU presentation or a fresh Unity physics tick.

This first prototype still copies pixels GPU-to-CPU and across a process boundary.
It is not a shared-GPU-texture or zero-copy PyTorch integration. Its offscreen
paint ceiling is 60Hz; this does not change the physical monitor's 144Hz setting
or intentionally accelerate game simulation. Keep framing and game timing under
review when comparing renderers.
Worker output audio is muted by default through Electron, without changing the
system volume or suspending the game's audio processing. Startup metadata records
audio_muted. Earlier smoke tests predate this addition.

## Local runtime

Dependencies are isolated under ignored `development/local/electron_worker`:

```powershell
npm.cmd install --prefix development/local/electron_worker --ignore-scripts `
  --no-audit --no-fund --save-exact electron@45.2.1
node development/local/electron_worker/node_modules/electron/install.js
```

The runtime must exist at
`development/local/electron_worker/node_modules/electron/dist/electron.exe`.
The official installer downloads and verifies the release. On this machine its
download stalled; a bounded direct download from the official GitHub release
was used instead, with SHA256 checked against the pinned package's checksums.json
before extraction. Do not use an unverified third-party binary.

The prototype's browser profile, logs and startup metadata go under the new report
directory's `electron` folder. The renderer window stays invisible. Node integration
is disabled for the game, sandbox/context isolation remain enabled, new windows are
denied and navigation is limited to the local server or the blank test page.

## Short capture validation

Start the existing local `server.py` on port 8000 if it is not already running.
With `$py` set to the existing Python venv:

```powershell
& $py -B -s -m tools.benchmark_ppo_electron_capture --seconds 5 `
  --report reports/electron_capture_UNIQUE_NAME
```

This checks channel order, corners, HTML overlay parity against a lossless browser
reference, static repaint, real Unity loading, canvas dimensions, GPU renderer
and capture-call latency. It sends no policy actions or performs training.

For a separate one-worker shared-PPO smoke test, use the existing parallel CLI
with `--headless --headless-capture electron --displays 1`, the current
`--framing-profile canvas_v1 --canvas-policy-input`, a preserved checkpoint,
`--steps 256 --max-minutes 2` and a unique report directory. These are optional
prototype settings, not a recommendation for a long training run. Existing PPO
and safety settings remain active; outputs never replace the source checkpoint.

Fallback: use `--headless-capture playwright` with the same existing CLI. No
source rollback, driver change or Python package installation is needed.

## Validation status

Live raw capture checks passed: exact synthetic pixel/HTML overlay parity against
a lossless screenshot, BGRA orientation, static repaint, trusted synthetic input,
NVIDIA WebGL renderer and clean application exit. Capture-only median was 29.82ms,
p95 34.45ms. Electron paint callbacks are not GPU presentation timestamps.

One worker completed 256 transitions/two shared updates, its settled-learning audit
passed and the source checkpoint was unchanged. Median/p95 policy observation
request-to-command timing was 44.42/62.94ms, compared with 77.45/150.81ms for a
fresh Playwright check from the same source/settings. These are short stochastic
runs with different episode trajectories, not proof of faster learning.

Six-worker checks stopped before completing any update, including after bounded
stale-frame retries. Physical RAM peaked at 99.6% and 99.0%, with as little as
0.12 GiB available. The second attempt's brief reaction samples had per-worker
medians 158-169ms; these aborted samples are not successful throughput results.
Do not use six Electron workers for long training on the evidence available.

The startup GPU feature-status object can precede GPU initialization and showed
inconsistent values. Actual in-game WebGL reported the NVIDIA D3D11 renderer;
do not interpret the startup object as a definitive compositor configuration.

See [measurements and limits](reports/ELECTRON_CAPTURE_20261031.md). Keep the
existing Playwright backend as the fallback while scaling is investigated.

Four workers subsequently completed 1024 transitions/two updates, all four
settled-learning audits passed, with no discarded rollout transitions and the
source unchanged. Per-worker request-to-command medians were 96.58-100.94ms,
p95 125.19-130.96ms. Peak RAM was 87.2% (3.95 GiB minimum available), sampled
VRAM 5542 MiB. Worker 0 discarded/retried 11 stale frames; the other three had
none. A nearly corroborated MENU transition recovered using the existing
bounded MENU-only extension now enabled for Electron as well as stream.
No terminal acceptance threshold changed and the extension never returns PLAY.
The first four-worker attempt stopped before this routing fix; it is preserved.
This is a candidate for further monitored validation, not proof that an
unattended multi-hour run will remain stable.

31 October follow-up: a later four-worker continuation stopped at episode 112
after five confident MENU readings contained two template animation dips.
The MENU-only extension can now also collect fresh evidence for a candidate
with two strong template hits, three corroborated readings and corroboration
on both final frames. Original terminal rules and the six-extra-capture bound
are unchanged. Offline failure replay passes; live recovery is still unverified.
See [the focused recovery report](reports/MENU_CONFIRMATION_20261031.md).
