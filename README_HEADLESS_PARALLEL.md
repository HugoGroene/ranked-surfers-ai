# Headless parallel PPO: current development state

30 October 2026. Six independent LOCAL Practice browser instances can feed one
shared PPO learner without physical/virtual game windows. No VDD is required;
the installed VDD device is disabled. GPU rendering was verified as NVIDIA ANGLE
D3D11. Checkpoints, raw demonstrations and Python environments remain local.

## What is implemented

- Independently guarded collectors, synchronized policy versions, per-worker GAE,
  passive-transition actor masks and one shared optimizer. Workers pause at rollout
  barriers. Death/menu handoffs and post-resume time are represented before returns.
- Explicit `--framing-profile canvas_v1`: complete calibrated 1920x945 canvas,
  linear letterbox to 512x288; separate raw safety images. Detector weights unchanged.
  New BC recording/training supports this version; historical datasets stay legacy.
- Six previews in three columns/two rows; one preview/s from existing images.
  The preview cap is adjustable to 1/5/10/15/30 FPS; encoding is background,
  latest-only work and never requests extra game captures. Actual unique FPS is
  limited by collected observations. See [scaling options](README_PPO_SCALING.md).
  The graph appends immutable points at 10-second boundaries and retains all points
  for the full requested runtime. Each averages the last 100 completed episodes,
  pooled across workers. Time-limited episodes are included and counted explicitly.
- PPO statistics, sampled action counts, optimized throughput, timings and sampled
  CPU/RAM/GPU/VRAM. Telemetry/graph refresh every five seconds in the browser.
- Atomic latest checkpoint plus separate numbered snapshots at save barriers.
  Source checkpoint remains unchanged. Saved dashboards can be reopened without PPO.
- Terminal survival summaries work without a dashboard. After successful training,
  `--end-plot auto` (default) opens the same saved survival curve in Matplotlib
  when no dashboard was enabled, and saves a PNG. Use show/save/off to override;
  frozen benchmarks skip automatic plots. The window opens only after workers
  stop and results are saved. Close it to return to the terminal. See
  [post-training plot options](README_PPO_SCALING.md#plot-the-mean-curve-after-training).

## Evidence and limitations

[Scaling report](reports/HEADLESS_SCALING_20261030.md): short checks measured
6.7 / 11.3 / 15.0 / 17.0 / 18.6 optimized transitions/s for 1/2/3/4/6 workers.
Budgets, episodes and aggregate rollouts differ: these are throughput checks,
not controlled learning-quality comparisons.

[Six-worker whole-canvas live check](reports/headless_canvas_v1_six_3min_20261030_01/README.md):
3840 transitions, five updates, all six audits passed, no discarded transitions
or guard stops, 18.4 optimized transitions/s. Peak RAM 92.4%, VRAM 7343 MiB.
No long-run memory-stability guarantee. Browser startup can intermittently expose
blank iframes: bounded reload/stagger is implemented; failures are preserved.
Visible-browser canvas calibration is implemented/unit-tested but still needs live
verification. Headless screenshot timestamp semantics and 300-second time-limit
reload require further validation. No experiment earned model promotion here.

The selected next diagnostic source is
`ai_data/rl_v2_runs/v612_largebc_defaults_3h_20261029_204226/latest.zip`,
SHA256 `66dc546f04253d9d2cf4aeb5af0764d9500c70ce78439159e518d5a96553bc9e`.
It is the latest substantial continuation, not the short smoke-test outputs.
It was saved after a UI obstruction safety stop, reviewed in the recovered context.
It contains 1713054 timesteps, rollout 128, LR .0003, batch 32, five epochs,
gamma .995, lambda .95, no target KL. Six workers aggregate 768 transitions/update.
This legacy-framing source will explicitly adapt to canvas_v1 on a new branch.

## Run the planned 30-minute diagnostic

Use the existing local server (`python -B -s server.py`) in its own terminal.
In another terminal, from the repository root with `$py` set to the existing venv:

```powershell
$checkpoint = 'ai_data\rl_v2_runs\v612_largebc_defaults_3h_20261029_204226\latest.zip'
if (-not (Test-Path -LiteralPath $checkpoint)) { throw 'Source checkpoint missing' }
$run = 'v612_canvas_diag_30m_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$output = "reports\$run"
$log = "ai_data\ppo_$run.log"
& $py -B -s -u -m tools.train_ppo_parallel `
  --headless --displays 1 2 3 4 5 6 --framing-profile canvas_v1 `
  --canvas-policy-input --checkpoint $checkpoint --steps 768000 `
  --max-minutes 30 --save-every-minutes 5 --worker-timeout 160 `
  --telemetry --preview-port 8001 --report $output `
  2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Inspect $output and $log" }
Get-Content "$output\parallel_training.json"
```

Open http://127.0.0.1:8001/ after startup. No game click needed. The high step
ceiling leaves the time budget in control. Loading is excluded and the current
shared rollout completes before stopping, so allow some overrun beyond 30 minutes.
The assistant has not launched the planned 30-minute diagnostic.

Preserve all run artifacts. Review optimized coverage/audits, KL/action coverage,
survival trend, per-update timings and memory growth, then compare saved policies.
A rising stochastic-training curve alone does not establish policy improvement.

To reopen the saved new-run graph later without starting training:

```powershell
& $py -B -s -m tools.ppo_worker_preview --report $output --port 8001
```

Older reports without episode metrics are not automatically backfilled. Historical
installer scripts/docs are retained for provenance; do not reapply them to the
current trainer without their explicit source/version checks.

## Experimental capture backends

`--headless-capture` applies with `--headless`:

| Value | Behavior | Status |
|---|---|---|
| `playwright` | Existing canvas-clipped rendered-page screenshot | Default |
| `cdp` | Direct CDP surface screenshot with cached, verified canvas clip | Similar capture median; no demonstrated speed improvement |
| `stream` | Demand-acknowledged latest rendered-page frame | Experimental; improves one-worker timing, regresses six-worker median |
| `electron` | Raw offscreen rendered-page bitmap | Short checks passed with one/four workers; six failed freshness/memory checks; see [Electron guide](README_ELECTRON_CAPTURE.md) |

Electron transport can optionally use `--electron-transport shared_memory` with
the isolated Koffi dependency. `--reuse-frame-state` caches the frozen classifier
only for an identical capture, without creating new evidence or changing frame
freshness. See [setup and results](README_PPO_SHARED_MEMORY.md). Socket transfer
and the default Playwright backend remain available.

Streaming keeps one latest image slot, checks viewport geometry and browser frame
timestamps, and requires a new frame presented after each capture request. It
does not return a cached image when a static Pause panel produces no new frame:
after 150ms it requests an independent surface screenshot. A stream frame must
be no more than 50ms old at capture return; existing downstream freshness guards
still apply. Neither bound guarantees that the selected policy action remains
that recent at dispatch. Both sources include rendered HTML overlays.

Faster sampling exposed nearly corroborated MENU transitions exhausting the
original six-capture window. Stream and Electron workers may add up to six passive
MENU-only checks when the last five readings consistently predict MENU, at least
four have template corroboration, and at least one has strong menu evidence.
An additional animation-dip candidate may request the same bounded checks when
those five readings contain two strong template hits, at least three corroborated
readings, and corroboration on BOTH final frames. This only requests new evidence;
it cannot confirm a terminal from the original six captures.
Repeated moderate evidence can also request these checks when at least four of
the five MENU readings are corroborated and the last three are corroborated.
It still needs a new capture to complete the original four-consecutive rule.
The existing terminal acceptance rules still decide confirmation; this extension
can never authorize a gameplay action. Conflicting gameplay evidence aborts it.

The four-worker two-hour attempt on 31 October stopped after 53 shared updates
and 27136 optimized transitions because the old candidate gate excluded this
two-dip pattern. Exact evidence replay and actual runtime tests cover the routing
fix; no classifier retraining or terminal threshold change was made. The existing
six extra captures at 45ms intervals remain the bound. See the
[failure, checkpoint verification and validation limits](reports/MENU_CONFIRMATION_20261031.md).

Two six-worker updates completed with all six settled-learning audits passing,
but median request-to-command latency was 128-133ms versus 89-97ms in the earlier
warmed screenshot test. Keep the default for now. The independent surface
fallback is unit-tested; successful live stream runs did not exercise it.
The 300-second stream teardown/reload path also needs live verification.

Full measurements, failed attempts and remaining bottlenecks:
[CDP capture investigation](reports/CDP_CAPTURE_20261030.md). Diagnostic checkpoints
are separate outputs and have not earned promotion. No long training was started.

## Approved five-hour continuation (30 October)

The user selected a five-hour overnight diagnostic instead of the proposed
30-minute run. Start from the substantial large-BC continuation named above,
not a short stream benchmark checkpoint. Use six headless workers, explicit
`--headless-capture playwright`, `--framing-profile canvas_v1`,
`--canvas-policy-input`, `--max-minutes 300`, `--steps 768000`,
`--save-every-minutes 5`, `--worker-timeout 160`, `--telemetry` and
`--preview-port 8001`, with a new timestamped report directory. The step ceiling
is deliberately high enough to let wall time control the run at measured rates.
Source LR .0003, 128 steps per worker, batch 32 and five epochs are retained;
six workers produce an aggregate 768-transition update.

Run `server.py` in a separate terminal if the local server is not already running.
Use the metrics-only dashboard button to stop preview image encoding/fetching.
No visible game window or virtual monitor is required. Keep Windows awake during
the run. The time budget excludes startup and finishes a complete shared update,
so actual elapsed time can exceed five hours. Safety stops still end the run.

The source checkpoint uses legacy framing; this continuation explicitly uses the
current whole-canvas profile. Tomorrow compare the source and new checkpoint
under identical whole-canvas evaluation conditions before claiming improvement.
Inspect parallel_training.json, all six worker audits, optimized coverage,
survival.json, progress.csv, latency metrics and system_usage.jsonl. Evaluate
visible-browser transfer separately. The assistant supplies commands; this
document does not establish that the overnight run was started or completed.

### Actual overnight outcome and recovery fix (31 October)

`reports/v612_canvas_6workers_5h_20261030_042039` stopped after 2727.28 active
seconds, 74 shared updates and 56832 optimized transitions. The saved latest.zip
passes ZIP integrity checks and contains 1769886 total timesteps; the source is
unchanged. All committed worker ledger prefixes pass the settled-learning audit.
Workers 2, 4 and 5 each have one unoptimized trailing transition, which makes
their untrimmed ledgers fail full coverage as expected after interruption.

Worker 0 reached a rendered MENU during post-resume refresh (ML .999, template
.484). The former .544 candidate-entry requirement prevented independent
confirmation from even starting. The current fix allows a valid confident MENU
prediction to request the existing bounded confirmation path. It still requires
four consecutive fresh ML+template corroborations before terminal handling and
guarded restart. Eight inconclusive confirmation captures still stop safely.
Invalid evidence, focus loss, Pause exclusion and freshness checks remain active.

162 focused tests pass, including replay of the failed values, persistent weak
templates, gameplay disagreement and integrated passive terminal/death credit.
An older uncertainty fixture was updated to include the later verified Pause
reference through the shared runtime fixture. No live test or new long run was
started for this fix; no checkpoint or historical result was modified. The fix
does not promise recovery from every possible UI failure or improve capture speed.
