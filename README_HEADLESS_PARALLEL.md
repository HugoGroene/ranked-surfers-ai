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
  The graph appends immutable points at 10-second boundaries and retains all points
  for the full requested runtime. Each averages the last 100 completed episodes,
  pooled across workers. Time-limited episodes are included and counted explicitly.
- PPO statistics, sampled action counts, optimized throughput, timings and sampled
  CPU/RAM/GPU/VRAM. Telemetry/graph refresh every five seconds in the browser.
- Atomic latest checkpoint plus separate numbered snapshots at save barriers.
  Source checkpoint remains unchanged. Saved dashboards can be reopened without PPO.

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
