# Speed-first configuration checks — 31 October 2026

## Decision

User priority: reaction speed takes precedence over memory savings. Keep separate
Electron hosts and inference replicas. Do not select the slower shared inference
service/shared host simply because it uses less RAM/VRAM.

Provisional training choice: **three workers, detector off, dense observations**,
shared-memory screenshot and rollout transport, cached detector geometry,
OpenCV threads 1 and exact-frame screen-state reuse. Leave previews off for these
measurements. Existing CLI defaults have not changed. A single worker had the
lowest measured latency; four detector-off workers had the highest collection
throughput. Three workers provided a useful compromise in these short checks.

These are local Practice engineering measurements, not policy-quality evaluations
or proof of long-run stability. Detector-off removes ongoing object inputs; it
does not remove the independent screen-state classifier or safety guard.

## Common setup and source

All eight runs used the same preserved checkpoint:
`reports/v612_sharedmem_4workers_2h_20261031_112933/latest.zip`.

SHA-256: `93cc8f240342dff3b1b418e547d56e1415e5b5704a98b6a5c168e5adf8b324dd`.
The source hash was independently rechecked after the tests and is unchanged.

Electron offscreen rendering, shared-memory frame/rollout transfer, per-worker
hosts/inference, canvas_v1 whole-canvas policy input, `--reuse-frame-state`,
`--cache-detector-features`, `--opencv-threads 1`, telemetry and no previews were
common. Source rollout 128 per worker, LR .0003 and five epochs were retained.
No new runtime changes were made during this speed investigation. New analysis
code, tests and documentation were added; all experiment outputs use new folders.

## Frozen-policy collection screening

Each run collected 256 transitions per worker across two cohorts, without
optimization or new checkpoints. Latency is screenshot-request to command issue,
not browser presentation age or Unity input response. It covers actual movement
commands; NONE and passive transitions still count toward collection throughput.

The steady column excludes the first 20 movement commands **per worker**, solely
to describe steady operation. Full samples are preserved in the comparison JSON.

| Report under reports/ | Workers | Detector | Compact | Steady median / p95 | Processed/s, no optimizer |
| --- | ---: | --- | --- | ---: | ---: |
| speed_screen_1_off_20261031_01 | 1 | Off | Yes | 28.0 / 34.9 ms | 16.47 |
| speed_screen_2_off_20261031_01 | 2 | Off | Yes | 30.3 / 37.9 ms | 28.02 |
| speed_screen_3_off_20261031_01 | 3 | Off | Yes | 32.5 / 43.8 ms | 41.94 |
| speed_screen_3_off_dense_20261031_01 | 3 | Off | No | 31.7 / 39.8 ms | 39.42 |
| speed_screen_4_off_20261031_01 | 4 | Off | Yes | 34.8 / 72.0 ms | 53.60 |
| speed_screen_4_on_20261031_01 | 4 | On | Yes | 36.4 / 80.3 ms | 38.03 |

All six completed, with no guard hard stop or discarded partial transitions.
All report unchanged source and frozen policy weights. The four-worker ON/OFF
pair measured 41% higher collection throughput with the detector off, but this is
not optimized training throughput. Input changes can alter actions, episode
lengths and reset costs, so it does not isolate detector compute alone.

Dense storage gave lower p95 but slightly lower aggregate collection throughput
than compact storage at three workers. This single short pair does not prove a
repeatable storage effect. Dense storage is the provisional choice because the
user prioritizes latency over memory use, and it passed actual-update validation.

## Actual optimization checks

Both started from the same preserved source, used three workers and dense storage,
and completed **768 optimized transitions / two shared updates**. Source settings
and all other selected options were fixed. Runs were stochastic, sequential OFF
then ON, with no repeated trials or matched procedural seeds.

| Report under reports/ | Detector | Active seconds | Optimized/s | Steady median / p95 / p99 | Worst worker p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| speed_validate_3_off_dense_20261031_01 | Off | 25.71 | 29.88 | 32.3 / 38.4 / 61.0 ms | 38.5 ms |
| speed_validate_3_on_dense_20261031_01 | On | 27.93 | 27.50 | 33.2 / 61.5 / 73.5 ms | 62.0 ms |

Detector-off measured 8.6% higher optimized throughput and 37.6% lower pooled p95.
All six worker audits returned `PASS_SETTLED_LEARNING`; both runs completed with
zero guard hard stops or discarded transitions. Separate diagnostic checkpoints
were saved in each run directory. Neither is promoted as the new reference.

The detector-off full sample includes 638 commands, median/p95 32.1/38.7 ms and
maximum 130.7 ms. Three commands exceeded 100 ms, one at startup per worker.
After excluding the first 20 commands per worker, 578 remained: max 77.1 ms,
none above 100 ms. Guard-capture-completion to command p95 was 16.9 ms, max 19.7 ms.
Startup spikes therefore still exist; these results do not establish a strict
latency guarantee. Do not hide them by reporting only the steady sample.

## Reproduce and inspect

Start the local server in a separate terminal if it is not already running:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s server.py
```

The tested three-worker detector-off short training configuration is:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$output = "reports\speed_check_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& $py -B -s -u -m tools.train_ppo_parallel `
  --headless --headless-capture electron --electron-transport shared_memory `
  --rollout-transport shared_memory --cache-detector-features --opencv-threads 1 `
  --reuse-frame-state --displays 1 2 3 `
  --framing-profile canvas_v1 --canvas-policy-input `
  --checkpoint reports\v612_sharedmem_4workers_2h_20261031_112933\latest.zip `
  --detector-mode off --steps 768 --max-minutes 1 --worker-timeout 160 `
  --telemetry --report $output
```

Omit `--compact-observations`, `--shared-inference` and `--electron-host shared`.
Headless display numbers are worker labels; virtual displays remain disabled.
The omitted preview port disables preview work. If monitoring is needed, add
`--preview-port 8001 --preview-fps 1`, then use metrics-only mode; its performance
was not measured in this matrix. Increasing steps/wall time is a separate run.

Analyzer usage (existing inputs are read-only; output directory must be new):

```powershell
$analysis = "reports\speed_analysis_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& $py -B -s -m tools.analyze_ppo_speed `
  reports\speed_validate_3_off_dense_20261031_01 `
  reports\speed_validate_3_on_dense_20261031_01 `
  --warmup-commands 20 --report $analysis
```

`--warmup-commands` controls only descriptive exclusion per worker, default 20;
use 0 to include startup in the ranking. Full samples are always reported.
`--report` is required and refuses an existing path. Ranking keeps worst-worker
p95 and aggregate throughput separate, with a Pareto frontier; RAM savings are
not a ranking objective. Failed/incomplete runs are excluded, and mixed source
hashes or frozen/training throughput cannot be compared in one ranking.

Saved analyses:

- [Six frozen checks](speed_selection_analysis_20261031_01/speed_comparison.json).
- [Actual-update ON/OFF pair](speed_training_pair_analysis_20261031_01/speed_comparison.json).

Four analyzer regression tests passed. Earlier implementation validation and
the slower shared-service experiments remain in
[SCALING_PIPELINE_20261031.md](SCALING_PIPELINE_20261031.md).

## Remaining limits

No claim that detector-off learns a better policy is supported yet. Its input
ablation can initially hurt a detector-dependent policy. Keep both paths and
preserved checkpoints, and evaluate continuations with matching detector mode.
No long training/evaluation was started in this investigation. The short matrix
does not establish a global optimum, repeatability of close differences, or
latency under unrelated desktop load. One worker remains the lowest-latency
tested choice if reaction time alone matters; three is the recommended parallel
training compromise rather than a claim of equal single-worker latency.
