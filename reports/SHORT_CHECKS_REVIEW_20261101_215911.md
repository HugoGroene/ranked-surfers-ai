# Road-vision short-suite review

Source: `short_checks_20261101_215911_369524/short_checks.json` and its individual stage reports. This review adds no runtime changes.

## Outcome

38 of 40 stages succeeded. All nine training smoke stages, nine matching evaluation stages, offline checks, crop evidence, behavior controls and compositor calibrations passed. The final PowerShell error reports two failed six-worker GPU stages; it does not invalidate or delete the successful stages.

Both failures were `No fresh offscreen paint within 500ms`. One failed waiting for worker readiness; the other failed waiting for a rollout. Neither completed a rollout. Both preserved the source policy and reported no finalization errors. This establishes a reliability problem for that tested configuration, not a diagnosis of its underlying cause. Do not weaken the freshness deadline to pass this benchmark.

## Capture and concurrency

| Configuration | Collected transitions/s | Worker request-to-command p95 |
|---|---:|---:|
| Four workers, full-resolution GPU capture, two repeats | 60.12–60.68 | approximately 39–41 ms |
| Four workers, bitmap capture, four runs | 47.03–56.56 | approximately 62–75 ms |
| Six workers, bitmap capture, four runs | 41.22–45.05 | approximately 90–96 ms |
| Six workers, GPU capture, two repeats | No completed rollout | Failed |

These are frozen-policy collection measurements, not optimized training throughput or Unity response latency. Full-resolution GPU capture is a supported candidate for the experimental comparison. Six workers are excluded from the next comparison. Production defaults are unchanged.

## What was implemented and exercised

- Separate full-image, wider road and narrower center crops; 96/128/192-pixel candidates and the larger baseline.
- Existing encoder and new spatial encoder compared at identical road crops and resolutions. The new encoder retains an 8x8 spatial grid before its projection, versus the existing 3x6 grid. It is an adaptation, not an exact reproduction of the video's network.
- Four-frame timestamped input in this suite. The broader temporal experiment remains separate.
- Optional BC training and consistent horizontal mirroring are implemented separately, but no new BC is included here. Human demonstrations were central to the video's results; this suite cannot reproduce that benefit without demonstrations.
- Safety classification continues to use the full composed image rather than the smaller policy crop.

One-minute training achieved approximately 10.47–11.40 optimized transitions/s for the smaller candidates versus 8.50/s for full512. These short runs include startup, resets and policy differences; they are compatibility checks, not a reliable speed ranking. Three-episode smoke evaluations cannot rank learning capability.

The inspected `crop_evidence/crop_evidence/02_road192_spatial_input.png` retains the approaching road and player, with letterboxing. One example cannot establish coverage of every lane, obstacle or camera state.

## Behavior controls

Each independent control completed 50 episodes:

| Controller | Mean duration |
|---|---:|
| NONE | 2.79 s |
| Learned temporal four-frame checkpoint | 4.35 s |
| Constant ROLL | 2.98 s |
| Constant JUMP | 3.91 s |

The learned controller selected JUMP for every recorded decision. Its 0.44-second advantage over constant JUMP does not establish image-dependent decisions; track variation and timing can differ. Retain simple controls when judging the longer learning experiments. Do not pool the separate six-episode crop-evidence run into these measurements.

## Next steps

1. Optional final targeted check: one worker, `road192_spatial`, full-resolution GPU capture, ten minutes of fresh training and ten evaluation episodes. Existing road_batch also runs a one-minute smoke gate first. Estimate 13–16 minutes including initialization; this is a reliability check, not a learning ranking.
2. Overnight comparison: keep capture backend, frame count/history, decision cap, detector-off mode, PPO settings and worker count fixed. Compare full128_tiny, road128_tiny, road128_spatial and road192_spatial with more than one training seed if the budget permits. Preserve completed outputs from this suite as separate evidence.
3. Compare three versus four workers with the selected model configuration afterward, including actual optimization and reaction latency. Worker count changes rollout size and update schedule as well as resource contention. Five workers can be investigated later if justified; another unchanged six-worker retry is not useful now.

No global optimum, long-run reliability or improved learning capability has been established by the short suite.
