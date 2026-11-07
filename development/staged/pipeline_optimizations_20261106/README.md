# Staged pipeline optimizations

Originally prepared 6 November 2026 while offline BC was running. On 7 November,
correctness and bounded live comparisons established useful improvements from
`pack` and `pinned`, now enabled in normal command-history training, together with
a precise decision-cap timer. Evaluation uses the same packing/timer changes.
`metrics` and `gpu_cache` remain experimental. See the
[measured results and rollback instructions](../../experiments/latency_tail_20261107/README.md).
The BC/PPO observation contract and checkpoint format remain unchanged.

Implemented candidates:

| Feature | Change | What stays identical |
|---|---|---|
| `pack` | Skip the temporary legacy concatenation that TemporalEnv immediately replaces | Native history updates, final spaced RGB stack, aux, clocks, guard calls |
| `metrics` | Accumulate PPO reporting tensors on GPU; reduce/transfer together at update end; use CPU rollout masks for actor-presence branch | Losses, normalization, masks, Adam steps, clipping, sample order, epochs |
| `pinned` | One bounded reusable pinned observation staging buffer per update; event protects DMA source reuse | Exact minibatch data; fresh device output per minibatch |
| `gpu_cache` | Cache immutable rollout observations on CUDA once per update | Uint8 images and original policy preprocessing; same shuffled minibatches |

`pinned` and `gpu_cache` are alternative experiments, not combined. CUDA caching
has a 1024 MiB allocation cap and requires 2048 MiB remaining free headroom before
allocation. Pinned staging is limited to 96 MiB. Cache refusal falls back to the
original buffer iterator and records the reason. These limits are conservative
checks, not guarantees against later allocations or other programs consuming RAM.
All transfer resources are released at the update boundary. No growing caches.

The metric optimization falls back to the original training method for a model
with a target-KL stopping rule. It never defers a decision that controls training.
Guard confidence/freshness, rewards, frame selection, model architecture and PPO
hyperparameters are not changed. Checkpoint compatibility does not prove speed
or learning improvement.

## Not implemented or enabled

- Native asynchronous GPU readback, GPU crop/resize and direct CUDA texture import.
  These need live graphics synchronization, full guard-image parity and frame-age
  checks; the current readback bridge is untouched.
- Shared reusable final observation storage. Returned observations must remain
  immutable for retained histories and rollout buffers.
- AMP, CUDA graphs or different PPO batch sizes. Channels-last and five-worker
  scaling were subsequently tested separately and were not selected as defaults.

## Current verification

10 dependency-free tests passed, covering source preservation, compilation,
configuration rejection, packing transformation and memory-budget boundaries.
After BC finished, 7 CPU checks passed (the CUDA-only check was skipped), and all
8 checks passed with CUDA enabled. This covers parameter-update parity, passive
Adam behavior, frame/aux parity, repeated transfer epochs and buffer lifetimes.
An import-name collision in the staged check launcher was fixed before these
passes. Reports: `reports/pipeline_optimization_checks/{cpu,cuda}.json`.
Live latency/throughput, worker spawn/cleanup and subsequent 30- and 15-minute
continuations have now completed; see the measured-results link above. The expanded
CUDA correctness suite passed 12 checks. Plain-Adam counter restoration and
post-resume capture pacing were also promoted through the normal latency runtime;
their tests and measurements are documented there.
Checks were not run alongside BC: the machine then had only about 0.26 GiB available RAM.
The BC process had approximately 25 GiB resident versus 3.2 GiB private committed;
memory-mapped dataset residency is consistent with much of that difference.
This is not proof of a leak or a full memory attribution.

All heavy check/probe entry points refuse concurrent training/evaluation and
require at least 2 GiB available system memory. Probes also require a passing
correctness report matching every staged Python source file. Transfer probes
require the CUDA report. No promotion to production is automated.

## After BC finishes

Use the existing virtual environment. These checks use tiny synthetic tensors,
not the full dataset, and do not launch games:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$opt = ".\development\staged\pipeline_optimizations_20261106"
& $py -B -s "$opt\check.py"
& $py -B -s "$opt\check.py" --cuda
```

Only proceed if the checks pass. The CUDA test verifies repeated shuffled epochs,
short final minibatches, exact data/dtypes and retained batch storage. CPU tests
compare actual optimizer parameter updates, metrics and sample accounting against
the original loop, including passive-only batches with existing Adam momentum.
They also compare final temporal frames/ages after the packing optimization.

The short four-worker probe starts its own two map servers and uses no dashboard.
Each invocation has a hard 120-second wall budget INCLUDING startup/cleanup;
missing the deadline is INCOMPLETE, not a pass. Use the same source checkpoint
for every case; diagnostic outputs are not continuation recommendations.

```powershell
$checkpoint = "reports\multimap_stability_20261106_171018_244012\stability\train\latest.zip"
& $py -B -s -u "$opt\probe.py" --checkpoint $checkpoint --features none --active-minutes 0.6 --wall-seconds 120 --reuse-pump-crop --threaded-preprocess --defer-startup-previews --bulk-readback --direct-readback
& $py -B -s -u "$opt\probe.py" --checkpoint $checkpoint --features pack --active-minutes 0.6 --wall-seconds 120 --reuse-pump-crop --threaded-preprocess --defer-startup-previews --bulk-readback --direct-readback
& $py -B -s -u "$opt\probe.py" --checkpoint $checkpoint --features metrics --active-minutes 0.6 --wall-seconds 120 --reuse-pump-crop --threaded-preprocess --defer-startup-previews --bulk-readback --direct-readback
```

Test `pinned` and `gpu_cache` separately only after CUDA checks pass and memory
has recovered. Repeat baseline/winner in reverse order before attributing small
timing differences to a change. Compare ordinary/reset/resume request-to-command
distributions, decision cadence, collection versus optimizer time, actual cache
activation, cleanup, guards, masks and sample accounting. Different episode
lengths/reset counts confound short throughput measurements.

The staged evaluator accepts the normal two-map options and enables only `pack`:

```powershell
& $py -B -s -u "$opt\evaluate.py" --checkpoint $checkpoint --episodes 2 --reuse-pump-crop --threaded-preprocess --defer-startup-previews
```

Add `--headless` for the training capture backend. It has normal per-episode
evaluation bounds, not the probe's two-minute wall deadline; do not run it as an
automatically authorized short probe. The bounded `eval_probe.py` in the latency
investigation folder supplies that wall deadline. The normal evaluator now uses
the promoted packing and pacing defaults.

## Why measurements are required

The legacy concatenation is a real redundant allocation, but previous profiling
put the classifier inside `_pack`; its full 8.8 ms cannot be attributed to image
stack construction. The proposed pack change may therefore save only a fraction
of that time. Metric synchronization and transfer changes target optimizer pauses,
not ordinary command latency. No claim that the game runs at a higher effective
decision rate until measured on both maps.

References: [PyTorch tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[pinned transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[Electron offscreen rendering](https://www.electronjs.org/docs/latest/tutorial/offscreen-rendering).
