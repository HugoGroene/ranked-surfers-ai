# Isolated latency candidates

Nothing in this directory is imported by ordinary training or evaluation.
Do not enable these candidates until the current training has finished and the
offline/live checks pass. Checkpoints and observation contracts are unchanged.

## Implemented candidates

- `pump`: replace only the temporal pump's period sleep with an absolute-deadline,
  high-resolution Windows wait on one bounded executor thread. It never blocks
  the browser event loop while waiting. The deadline remains `capture start +
  period`; executor scheduling does not start an additional full period. Existing
  future-frame, serial, age, generation/reset, on-demand-pause and guard rules
  remain intact. This targets timer overshoot, NOT the inherent wait for a future
  paint. It deliberately does not reuse an earlier in-flight/cached frame.
- `classifier`: FP32 channels-last layout for the existing CPU screen classifier.
  Same weights, crop, normalization, softmax and thresholds. Includes the cost
  of input layout conversion. No ONNX dependency or classifier retraining.
- `both`: combine only after separate measurements. `baseline` uses today's
  production optimizations, including precise decision pacing and pinned PPO
  transfers, without either new candidate.
- `policy_layout`: channels-last for the worker policy's RGB encoder only, in
  FP32. The coordinator, auxiliary MLP, action distribution, sampling and guard
  are unchanged. Worker weight installs preserve the layout. This is separate
  from the screen-classifier layout experiment.
- `fused_adam`: CUDA fused Adam in the central learner only, preserving parameter
  groups, learning rates, epsilon, betas, moments and step counts. PPO losses,
  masks, epochs, clipping and minibatch order are unchanged. A state export hook
  writes the source optimizer's execution flags and CPU step counters into
  saved checkpoints, so a regular resume does not silently opt into fused Adam.
  Floating-point updates need numerical verification; they are not claimed to
  be bit-identical. Non-CUDA, non-FP32 and unsupported optimizers are refused.
- `policy_fused`: the previous two candidates combined, not the pump/classifier
  pair. Benchmark the isolated candidates first.

These are experiments, not demonstrated speedups. An executor hop or a memory
layout conversion can make latency worse. CUDA graphs, ONNX export, shared policy
inference and native asynchronous GPU readback are NOT implemented here.

## Check after the running job finishes

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$stage = ".\development\staged\latency_candidates_20261107"
& $py -B -s -u "$stage\check.py"
if ($LASTEXITCODE -ne 0) { throw "Candidate correctness check failed." }
```

The checker refuses an active trainer/evaluator before importing Torch. It tests
real pump freshness and cleanup with a fake source, then compares FP32 classifier
probabilities, top labels and threshold crossings on labeled Halloween/Tokyo
images, sampled across sessions and classes. It alternates inference order and
records timings, but these CPU timings are not a four-worker speedup claim.
It records source and classifier hashes; later probes reject stale results.
This supplements rather than replaces live guard-state/transition verification.

The policy/optimizer candidates have a separate checkpoint-specific CUDA gate:

```powershell
$checkpoint = "reports\bc_actor_repair_20261107_014610_719776\bc_best.zip"
& $py -B -s -u "$stage\check_models.py" --checkpoint $checkpoint --cuda
if ($LASTEXITCODE -ne 0) { throw "Policy/optimizer correctness check failed." }
```

This checks the actual policy's probabilities, values, deterministic and seeded
sampled actions, normalization buffers through weight transport, unchanged input
storage and layout after a worker weight update. It also compares ordinary and
fused Adam across two masked PPO updates, with existing momentum and all-active,
mixed and all-passive minibatches. All-passive actor weights/moments/step counters
must remain unchanged. Exported state must resume with ordinary Adam. The source
checkpoint is never saved over. CPU-only checking is available by omitting
`--cuda`, but cannot authorize a CUDA live probe. No measured speedup is inferred
from these correctness tests.

## Bounded training probes (only while idle)

Use the same source checkpoint for every arm. The launcher starts both map
servers, uses four workers, no dashboard, and completes whole PPO updates within
a configurable wall budget including cleanup (120 seconds by default, up to
300 seconds). A deadline is INCOMPLETE, never PASS.
Probe checkpoints are diagnostic outputs, not recommended continuation models.

```powershell
$checkpoint = "reports\bc_actor_repair_20261107_014610_719776\bc_best.zip"
foreach ($arm in @("baseline", "pump", "classifier", "both", "baseline")) {
    & $py -B -s -u "$stage\probe.py" --arm $arm --checkpoint $checkpoint
    if ($LASTEXITCODE -ne 0) { throw "Probe failed or incomplete: $arm. Keep reports." }
}
```

After `check_models.py --cuda` passes, a separate list can compare `baseline`,
`policy_layout`, `fused_adam`, `policy_fused`, `baseline` using the same command.
These arms require a matching checkpoint hash; the screen-classifier candidate
requires a matching classifier hash. Each gate also checks staged source hashes.

Each probe defaults to 0.6 active minutes; total wall time varies with startup,
resets and update completion. `--active-minutes 3 --wall-seconds 300` permits
longer steady-state checks. The wall bound applies to EACH probe, not the entire
list. Never run the list alongside another trainer or evaluator.

Use `development/experiments/latency_tail_20261107/analyze.py` on each printed
folder to compare ordinary/reset/resume latency, warm optimizer time, guard
fallbacks and throughput. Do not add overlapping stage timings. Also inspect
learning audits, memory, frame age/serials and clean worker shutdown before
promotion. The same runtime patches can be tested in evaluation later; no normal
evaluator is changed by this directory.

## Verification at staging time

The seven stdlib-only tests in `test_runtime.py` and three in
`test_model_staging.py` passed while training ran, without
Torch, CUDA, browsers or workload benchmarks. `protected_sources.json` records
the active diagnostic's source manifest (221 files, all matched). CPU classifier
parity, real-pump integration, CUDA policy parity and optimizer tests are
intentionally deferred until idle. Adding these candidates invalidates earlier
staged correctness reports; re-run the relevant gate before a live probe.

References: [PyTorch channels-last](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
[PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html),
[Windows high-resolution timers](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw).

## Completed comparisons, 7 November

Four separate live candidates passed worker learning audits, source checks and
cleanup. All began from the same repaired BC checkpoint, with four workers and
the production pack/pinned/decision-pacing settings. These short runs are timing
diagnostics, not learning-quality comparisons.

| Arm | Ordinary median / p95 ms | Warm update median seconds | Decision |
|---|---:|---:|---|
| Baseline | 33.5 / 36.4 | 2.85 | Reference |
| Capture pump deadline | 30.4 / 37.7 | 2.92 | Leave experimental; tail worsens |
| Classifier channels-last | 33.6 / 37.3 | 2.89 | Leave disabled |
| Policy encoder channels-last | 33.7 / 38.0 | 2.88 | Leave disabled |
| Fused Adam | 33.5 / 36.3 | 3.10 | Leave disabled |

Reports are `reports/latency_candidate_20261107_031250_763387`,
`031513_325923`, `031649_741780`, `031847_693499`, and `032023_244500`
(the latter suffixes share the same directory prefix).

A separate resume slowdown was found: SB3 loads ordinary Adam's scalar step
counters onto CUDA, unlike Adam's normal CPU initialization. Scalar reads then
synchronize repeatedly during each minibatch. The isolated `adam_cpu_steps` arm
restores only these counters to CPU; moments, values, hyperparameters and PPO
formulas are unchanged. `pump_cpu_steps` combines it with capture pacing.

Three-active-minute comparisons started from the SAME completed 30-minute PPO
checkpoint `reports/multimap_stability_20261107_023453_536275/stability/train/latest.zip`:

| Arm | Ordinary median / p95 ms | Warm update median seconds | Optimized/s |
|---|---:|---:|---:|
| Before counter fix | 33.3 / 37.1 | 4.11 | 36.4 |
| CPU counters only | 33.1 / 36.8 | 2.87 | 39.4 |
| CPU counters + pump | 29.9 / 42.0 | 2.88 | 38.8 |

Reports: `032852_531237`, `033425_973729`, `033820_950567` with the prefix
above. All completed 13 or 14 full updates, all four worker audits, saved their
diagnostic checkpoints and cleaned up without forced termination. Throughput
also depends on deaths/resets; this is not a guaranteed percentage gain.

Only the CPU-counter correction was promoted to the normal measured runtime,
in `development/experiments/latency_tail_20261107/adam_steps.py`. Therefore a
`baseline` probe AFTER this promotion already includes that correction. The
staged counter arms are retained for provenance and are idempotent on the new
baseline; they no longer constitute an independent before/after comparison.
Fused, capturable and differentiable optimizers are excluded from the fix.
CUDA checks compare weights, moments, counter values and update accounting
exactly across active/mixed/all-passive batches. No checkpoint is reset.

One earlier one-minute pump attempt (`032425_126149`) exceeded its 120-second
overall deadline during finalization. Its partial performance is excluded from
passing evidence; the subsequent longer run completed normally. The initial
sandbox Electron startup failure (`031202_226889`) is also excluded.

Phase-separated JSON analyses and the 30-minute run review are in
`reports/latency_review_20261107/`. These are request-to-command-dispatch timings,
not Unity response times, and do not include NONE steps. Startup/reset/resume
must be examined separately. Ordinary latency still has occasional outliers;
the full 30-minute trace included one 211 ms decision delayed by a guard capture.

## Resume verification follow-up

`resync_spacing` tested 33 ms minimum start-to-start capture spacing in the
four-frame post-optimizer verifier, replacing the additional sleep after each
completed check. It retained all checks and minimum spacing; actual-verifier
tests cover menu/pause/focus rejection as well as the four accepted frames.

The baseline (`041534_159287`) and candidate (`042110_236519`) each ran three
active minutes from the completed 15-minute checkpoint, with four workers.
Median resume-to-fourth-frame delay fell from 293 to 195 ms; ordinary latency
was unchanged. The candidate had 211 ms p95, one 360 ms outlier, and no timestamp
fallbacks or sub-33 ms capture intervals. Both passed audits and cleanup.

This correction was promoted to the normal measured runtime. Consequently
`baseline` now includes it and the staged `resync_spacing` arm is idempotent.
Historic before/after results must not be treated as current different defaults.
See `development/experiments/latency_tail_20261107/README.md` for scope and traces.
