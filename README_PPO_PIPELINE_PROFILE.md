# Opt-in four-worker pipeline profiling

`tools.train_ppo_parallel --profile-reaction` adds buffered host timings around
capture, native-frame preparation, screen classification, observation packing,
policy tensor preparation/forward/action retrieval and browser-step guards.
The same flag works with the retained Playwright backend and Electron. It does
not add captures, actions, policy resampling or per-decision GPU synchronization.
CUDA events are warmed before collection; ordinary actions.cpu() already provides
the synchronization used to read the event interval. Passive steps may report no
GPU interval if work has not completed. Profiling adds some overhead and can
change real-time trajectories; these are short engineering diagnostics.

The synthetic real-SB3 invariance regression verifies identical actions, rewards,
actor masks and final CPU policy weights with profiling enabled or disabled.
It does not claim that real-time GPU/game trajectories remain identical.

Electron additionally reports Node request-to-paint and bitmap-extraction time,
Python header/payload receipt, receive-resumption time and an endpoint transport
residual. This residual includes socket delivery, copies and scheduling, not pure
network latency. Paint wait includes frame scheduling/rendering/readback; **GPU
readback alone is not isolated**. CUDA event intervals include contention/idle
time between events, not exclusive kernel execution.

## Short comparable runs

Start the existing local server with `& $py -B -s server.py` in its own terminal,
if port 8000 is not already serving Practice. From the repository root with the
existing Python venv assigned to `$py`:

```powershell
$backend = 'playwright' # Or 'electron'; run sequentially, never concurrently.
$checkpoint = 'reports\v612_canvas_6workers_5h_20261030_042039\latest.zip'
$name = 'pipeline_' + $backend + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$output = "reports\$name"
& $py -B -s -u -m tools.train_ppo_parallel `
  --headless --headless-capture $backend --profile-reaction `
  --displays 1 2 3 4 --framing-profile canvas_v1 --canvas-policy-input `
  --checkpoint $checkpoint --steps 1024 --max-minutes 2 `
  --worker-timeout 160 --telemetry --report $output
if ($LASTEXITCODE -ne 0) { throw "Preserve and inspect $output" }
& $py -B -s -m tools.analyze_ppo_pipeline_profile `
  --report $output --output "reports\analysis_$name"
```

The two-update budget includes verified optimizer pause/resume and guarded
resets. It is not a learning-quality experiment or a recommended long run.
The source model stays unchanged and diagnostics save separate checkpoints.

## Outputs and interpretation

Each worker writes pipeline_profile.jsonl and pipeline_profile_metadata.json at
shutdown. Timings are buffered in RAM (maximum 100000 events, overflow counted).
The analyzer refuses overflowed traces. Existing learning_transitions.jsonl
provides source lineage, actual command timestamps, actions and reward records.
Large detailed traces stay local; selected small summaries can be published.

The analyzer writes a new output directory with summary.json and decisions.jsonl.
It checks source lineage and causality and partitions request-to-command time:

1. Source delivery: request start through complete image reply.
2. Reply to capture completion: decoding, guard padding and inherited capture work.
3. Capture completion to policy start: packing/classification and collector handoff.
4. Policy input/forward/action-ready host interval.
5. Action ready to command issue: guard work and environment/browser handoff.

These five durations reconcile to each decision's total. Child timings overlap
those stages and must not be added again. Nested calls are unioned rather than
double-counted. Summaries compare all commands, the central 50% and slowest 5%.
NONE actions/passive transitions have no command-issue latency and are excluded
from those summaries. Command issue is not Unity input consumption.

Receiver tuning is separate from instrumentation: Electron now uses a frame-sized
StreamReader limit (one bounded frame plus header), rather than the 64KiB default.
This avoids repeated pause/resume while readexactly waits for a roughly 7.3MB
frame. There is still exactly one requested frame and the same validated payload
size, clocks, serials and freshness bounds. It does not preallocate multiple frames.

See the [dated profiling report](reports/PIPELINE_PROFILE_20261031.md) for outcomes
and limits. The tuned Electron checks completed with pooled command latency
medians 54-57ms and p95 116-120ms, versus Playwright's 80/162ms in this short
comparison. Neither long-run reliability nor better learning is established.
