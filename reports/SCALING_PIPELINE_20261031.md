# Opt-in scaling implementation and checks: 31 October 2026

All live checks below used LOCAL Practice and the same preserved source:
`reports/v612_sharedmem_4workers_2h_20261031_112933/latest.zip`, SHA256
`93cc8f240342dff3b1b418e547d56e1415e5b5704a98b6a5c168e5adf8b324dd`.
The source remained unchanged. Short diagnostic models are separate artifacts,
not promoted references. No expensive policy evaluation or long training ran.

## Implemented paths

- `ppo_shared_arrays.py` and `ppo_rollout_transport.py`: coordinator-owned CPU
  slots, typed layouts, metadata-only pipe messages, version/hash/publication
  checks, read-only consumer arrays, and matching commit before overwrite.
- `ppo_compact_buffer.py`: exact RGB-frame interning, overlapping stack references,
  reset/resync support, borrowed published worker sources in the learner, and
  minibatch-only reconstruction. No lossy image conversion or observation change.
  A final preparation pass removed redundant full-stack/intermediate reconstruction
  copies; offline optimizer parity was rechecked after that change.
- `ppo_detector_features.py`: cache immutable object geometry, check age/episode
  on every observation and preserve action/timing context. Optional per-worker
  OpenCV thread limit avoids uncontrolled nested CPU threads.
- `ppo_shared_inference.py`: a coordinator-versioned policy/YOLO service with
  separate per-game observation/detector slots and request channels. Policy
  round trips default to a strict 50ms deadline. YOLO uses one bounded background
  executor so object extraction cannot synchronously block the policy event loop.
  All possible policy batch sizes are warmed before gameplay. Critic-only
  requests do not sample actions. Action/log-prob/value lineage is retained.
- `electron_worker/multi.cjs`: optional shared host with independent windows,
  session partitions, capture slots and serials. Each window is muted. Startup
  is serialized to reduce simultaneous Unity initialization/readback stalls.
  Original per-worker hosts remain the default.
- `--detector-mode off`: skip YOLO construction, warmup and scheduling; retain
  image/schema, unavailable detector encoding, action/timing features and the
  separate screen-state safety classifier. This is an input ablation.
- `benchmark_ppo_detector.py`: frozen-policy ON/OFF/OFF/ON stages from one source;
  no optimization or checkpoint writes, processed/optimized counts distinguished.
- Background thumbnail publisher with one replaceable pending RGB image,
  configurable CLI/dashboard FPS cap and metrics-only mode. No extra game captures.

Flags, commands and ownership details: [scaling guide](../README_PPO_SCALING.md).

## Successful short checks

| Report suffix under reports/ | Workers | Optimized transitions | Optimized/s | Movement commands | Request-to-command median / p95 | Peak system RAM | Peak sampled VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scaling_v2_train_two_20261031_02 | 2 | 512 | 17.73 | 390 | 33.52 / 69.77ms | 54.6% | See raw telemetry |
| scaling_v2_train_four_20261031_01 | 4 | 1024 | 19.87 | 810 | 59.09 / 117.54ms | 67.7% | 3447 MiB |
| scaling_v2_storage_four_20261031_01 | 4 | 1024 | 27.32 | 820 | 39.41 / 81.23ms | 73.8% | 5222 MiB |

The first two combine shared inference, a shared Electron host, shared/compact
rollouts, geometry caching, OpenCV threads=1 and exact-frame state reuse. The
two-worker test also used a 30 FPS preview cap; the four-worker tests had no
preview. The storage-only four-worker bundle keeps separate inference replicas
and Electron hosts. All successful training checks completed two updates,
passed every worker learning audit, discarded zero transitions, recorded zero
hard guard stops and produced CRC-valid separate checkpoints. Each includes
optimizer pause/resume, fresh-stack recovery and guarded episode resets.

The storage-only four-worker run held 57.95-59.28 MB of unique RGB data and stack
references per final worker rollout, versus 226.49 MB of dense images. This is
about a 74% reduction in that image storage, not total process RAM. Shared slots
still reserve capacity for worst-case frame diversity; model/browser/runtime
allocations also remain. The full shared bundle used less sampled RAM/VRAM but
was slower in these short checks. Neither is a controlled independent ablation
of every flag, and system baseline usage/episodes differ.

`scaling_v2_frozen_two_20261031_02` completed 512 processed transitions in 20.61s
(24.84 processed/s), with zero optimized transitions, unchanged source/policy
hashes, no checkpoint writes, no discarded transitions and no guard stops.
Detector was off; all shared paths were enabled. Its pooled movement latency
was 31.08ms median / 44.05ms p95. This is NOT an isolated detector speedup estimate
because it has no optimizer work and different observations/trajectories.

## Failed attempts and corrections

- `scaling_v2_frozen_two_20261031_01`: shared-host startup failed because the
  initial implementation used Node Socket's read-only `pending` property. The
  application field was renamed `frameRequest`; the successful frozen check
  followed. No weights were changed.
- `scaling_v2_train_two_20261031_01`: no update completed. A policy reply expired
  while synchronous shared YOLO work occupied the service. Moving detection to
  a separate bounded executor retained the 50ms deadline; the next training check
  passed. This does not promise that future deadline misses cannot occur.
- `scaling_v2_train_six_20261031_01`: no update; worker 5 timed out receiving its
  initial Electron frame. Peak RAM 86.2%, minimum available 4.27 GiB.
- `scaling_v2_train_six_20261031_02`: serialized initialization reached gameplay,
  then worker 4 exhausted the normal guard on a real title/menu screen. Scores
  [.574, .628, .526, .619, .574, .574] ended with three consecutive corroborated
  MENU frames. Exact JSON is preserved in a second regression fixture. Such a
  candidate can now request the same bounded extra frames when four of the last
  five readings have template corroboration and the last three are corroborated.
  Confirmation still requires the unchanged original terminal rules, with no
  movement allowed. No update completed. Its 53 brief movement samples measured
  265.55ms median / 528.30ms p95, not a valid steady-state benchmark. Peak RAM was
  82.9%; this attempt is not evidence that six workers are responsive enough.
- `scaling_v2_train_six_20261031_03`: no update; an intermittent initial capture
  timeout remained despite serialized startup. Peak RAM 86.3%, minimum available
  4.24 GiB. Six shared-host workers are therefore NOT validated or recommended.

No terminal acceptance threshold, menu/classifier confidence, prepared-reading
20ms limit, Electron 50ms paint-age check, 100ms resync limit or 2s frame reply
timeout was widened. Existing safe stops remain. The final MENU candidate change
has offline replay coverage; the affected six-worker path has no successful live
recovery evidence yet.

## Verification and next use

Offline coverage includes exact dense/compact minibatches and final real CPU PPO
weights across multiple epochs; shared publication ownership; immutable frame
retention; source geometry/freshness/reset parity; detector-off construction;
bounded inference and version/source mismatch refusals; real spawned policy
service values/log probabilities/updates; diagnostic forward RNG invariance;
latest-only preview ownership; actual HTTP FPS controls and invalid/foreign-origin
refusal; and the prior runtime/learning/capture/guard regressions.

Final totals: 211 focused pipeline tests and 100 existing feature/guard/demo/
recording tests passed, with no skips. Electron host/bridge and dashboard script
syntax checks passed, as did Git whitespace checks. Source checkpoint SHA256,
screen-classifier source hash and terminal decision source hash remained unchanged.
The short test coordinators and Electron worker hosts were no longer running
at the final read-only process check.

The full four-stage detector ablation has NOT been run. Its command is ready for
the user to run in a separate terminal after reviewing these results. Keep the
lighter four-worker storage/preparation configuration for that first benchmark.
Shared inference and the multi-window host remain opt-in experiments. Do not
adopt six workers, infer learning improvement, or promote diagnostic checkpoints
from these checks. Policy evaluation remains separate from pipeline throughput.
