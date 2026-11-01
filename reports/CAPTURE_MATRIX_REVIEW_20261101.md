# Capture matrix review and temporal startup repair — 1 November

Source: `capture_matrix_20261101_154254_066769/comparison.json` and its raw worker
reports. All five compositor calibrations and all twelve frozen-policy stages
completed. They collected 25,728 transitions, optimized zero transitions, and
reported unchanged policy/source hashes. The temporal batch failed before its
first training rollout: no temporal learning result was produced by that batch.
All original artifacts are retained.

## Results

| Capture arm | One-worker collected/s | Three-worker collected/s | Three-worker request-to-command median / p95 (ms, worker range) |
|---|---:|---:|---|
| Bitmap full, first control | 18.92 | 47.77 | 32.4–32.7 / 38.6–39.4 |
| Bitmap, Unity 75% | 19.52 | 46.69 | 32.0–32.5 / 38.7–39.7 |
| Bitmap, Unity 50% | 20.28 | 42.59 | 32.4–32.6 / 38.8–39.6 |
| GPU shared texture, full pixels | 21.06 | 48.92 | 31.9–32.1 / 38.5–38.6 |
| GPU half readback | 20.81 | 49.39 | 30.9–31.7 / 36.8–37.2 |
| Bitmap full, final control | 19.06 | 45.96 | 31.8–32.4 / 37.8–38.8 |

These are frozen collection rates, NOT optimized training throughput. Each stage
had one active minute plus complete-rollout overshoot. Different death/reset
trajectories affect throughput, and the three-worker bitmap controls differ by
about 4%. Results do not prove an optimal configuration or improved learning.
Latency summaries are per-worker recorded samples, not pooled full-run percentiles;
occasional maxima were much higher than p95. No timing measures Unity input response.

Full-pixel GPU collection is roughly 11% above the one-worker control average and
4% above the three-worker control average. Half readback is about 5% above the
three-worker control average, with around 1–2 ms lower p95. This is promising but
modest, and half readback changes detail. Do not promote it on these data alone.

With three workers, GPU full readback took about 16.0 ms median capture time versus
15.6–15.7 ms for bitmap; it does not uniformly reduce capture latency under load.
GPU half readback took 13.9 ms, but additionally required approximately 2.23 ms CPU
geometry restoration. GPU bridge median costs were 2.55 ms full and 1.03 ms half.
Whole-system median CPU was 25.3–27.9% for bitmap controls, 20.3% GPU full, and
18.6% GPU half. System memory availability changed materially during the sequence;
do not attribute all memory differences to capture. Lower overall CPU is not proof
that adding workers will improve reaction speed.

Recommendation: retain full visual detail. Run a focused repeated bitmap/GPU-full
comparison at three and four workers to test whether the lower CPU load improves
scaling. Do not repeat the completed lower-resolution sweep or immediately expand
to six workers. Keep capture choice fixed across temporal learning arms.

## Repairs and verification

The experiment assigned floating-point `30.0` to FPS, which the inherited V3 -> V2
CLI serialized and passed to an integer-only parser. An isolated process-local
bridge now parses legacy defaults at integer 30, then restores the validated
experimental runtime rate. This supports fractional and 60 Hz experiments without
rounding or modifying production parser limits. New tests exercise the actual
inherited parser for every preset, fractional rates, 1 Hz and 60 Hz; invalid rates
are rejected, argv is restored and repeated installation is harmless.

A new eight-frame scratch smoke run completed 128 optimized transitions with
PASS_SETTLED_LEARNING and an intact checkpoint:
`temporal_ratefix_smoke_20261101_01`. Its subsequent two-episode evaluation loaded
and ran successfully, but exposed a capture-pump shutdown cancellation race. A
persistent stop flag now prevents a completed receive from restarting the pump
after cancellation. An offline regression deliberately swallows cancellation to
verify termination. These smoke results are functional checks, not learning scores.
The repeated two-episode evaluation `temporal_ratefix_eval_20261101_02` completed
and exited cleanly after the shutdown fix. All 36 offline tests across the two
experiment folders passed, including the targeted follow-up plan test.

## Next batch

Use `run_batch.ps1 -CaptureFollowup`: two synthetic calibration arms and repeated
bitmap/full-GPU comparisons at three/four workers, followed by temporal smoke tests,
six scratch training runs and their six evaluations. This does not repeat the
entire completed matrix, and does not use its checkpoints as training initialization.
The follow-up uses the same preserved latest detector-off checkpoint with no updates.
`-SkipCapture` instead runs only the repaired temporal batch. Existing outputs are
never overwritten; new reports receive unique names. No automatic resume or backend
promotion is implied. Estimated combined runtime remains about 4–6 hours, dominated
by three hours of training plus evaluation; episode length affects the estimate.
