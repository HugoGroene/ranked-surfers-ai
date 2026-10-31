# Three-worker detector-off diagnostic — 31 October 2026

Run: `reports/v612_speed3_detectoroff_15m_20261031_133058`.
Log: `ai_data/ppo_v612_speed3_detectoroff_15m_20261031_133058.log`.

## Outcome

Completed 79 shared updates, **30,336 optimized transitions in 903.73 active wall
seconds (33.57/s)**. All three freshly rerun ledger audits returned
`PASS_SETTLED_LEARNING`, with 10,112 transitions used per worker, no guard hard
stop, no discarded partial transitions and no unledgered completed interactions.
This is a successful 15-minute pipeline diagnostic, not evidence of multi-hour
reliability or improved evaluated policy quality.

The configuration matches the recommended speed candidate: three per-worker
Electron hosts and policy replicas, detector off, shared-memory screenshot and
rollout transport, dense observation storage, OpenCV threads 1 and exact-frame
state reuse. Previews were disabled. Source rollout 128 per worker (384 aggregate),
LR .0003 and five epochs were retained. No reward, guard or architecture changes
were made for this run.

## Survival and behavior

| Statistic | Result |
| --- | ---: |
| Completed episodes, pooled across workers | 355 |
| Overall mean active survival | 3.7593 s |
| Median active survival | 3.2902 s |
| Maximum active survival | 10.9253 s |
| First 100 completed episodes, mean | 3.7186 s |
| Last 100 completed episodes, mean | 3.9283 s |
| First / middle / final five-minute section, mean | 3.7813 / 3.5400 / 3.9835 s |
| Time-limit truncations | 0 |

All 355 recorded episode terminations were deaths. The per-worker means were
3.6246, 3.8895 and 3.7740 seconds over 123, 114 and 118 episodes respectively.
The pooled overall mean weights individual episodes, rather than averaging the
three worker means.

Survival dipped in the middle and recovered late. The last-100 mean exceeded
the first-100 mean by 0.21 seconds; this stochastic, changing-policy training
sample does not establish a statistically reliable learning improvement. It
cannot be compared directly with historical deterministic evaluations or used
to isolate the detector-off effect. No matched before/after evaluation was run.

Actor selections were NONE 6,981, LEFT 28, RIGHT 32, JUMP 13,285 and ROLL 9,621.
The 60 lateral choices were only 0.20% of 29,947 actor transitions: behavior is
still overwhelmingly jump/roll/none. Faster collection has not demonstrated
that the model has learned substantially longer survival or lateral strategies.

## Reaction latency

Request-to-command measurements are from each policy source screenshot request
to input command issue. They do not measure Unity action response or the age of
the underlying rendered simulation frame. NONE/passive transitions are not
included in the movement-command latency sample.

| Sample | Commands | Median | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full run | 22,966 | 31.57 ms | 38.28 ms | 42.82 ms | 150.63 ms |
| Excluding first 20 movement commands per worker | 22,906 | 31.57 ms | 38.27 ms | 42.67 ms | 84.77 ms |

The full sample has three commands above 100 ms, one above 150 ms. No command
in the steady sample exceeded 100 ms. Steady p95 by five-minute section was
38.57, 38.13 and 38.07 ms: no progressive reaction slowdown is visible here.
Worst-worker steady p95 was 38.38 ms. Guard-capture completion to command p95
was 16.66 ms; maximum was approximately 20.00 ms (the logged timestamp includes
instrumentation after the freshness check). Existing freshness limits remain
unchanged.

The initial short detector-off update test measured 29.88 optimized/s; this longer
sample measured 33.57/s with similar latency. Different trajectories and startup
amortization limit causal interpretation. This does not measure preview overhead.

## Optimization and resources

Approximate KL median was .0303, maximum .0677; clip fraction median .1745,
maximum .2953. Final explained variance was .8276 and final value loss .2155.
These show updates were executed; neither statistic proves improved survival.
No settings were changed in response to these observations.

During active training, sampled whole-system CPU mean utilization had a median
of 24.0%, busiest-core utilization median 62.5% and maximum 98.8%. Whole-system
RAM utilization median was 65.85%, maximum 68.4%. Sampled NVIDIA total VRAM was
3,611 MiB median, 3,691 MiB maximum. These include other applications and are
not per-training-process allocations. Low aggregate use does not rule out
per-core or frame-delivery bottlenecks.

## Checkpoints preserved

Latest: `reports/v612_speed3_detectoroff_15m_20261031_133058/latest.zip`.

SHA-256: `b6d324b12a458e5e2eef8019babfcf426d680aefb7e349784b06fa05a85a4244`.
Saved total timesteps: 1,877,022; update counter: 68,415; detector mode: off.
Five-minute snapshots at 9,984, 20,352 and 30,336 new transitions are intact.
All snapshot/latest ZIP CRC checks passed. The latest, latest.prev and final
numbered checkpoint contain the same final model bytes.

The preserved detector-enabled source remains unchanged:
`reports/v612_sharedmem_4workers_2h_20261031_112933/latest.zip`, SHA-256
`93cc8f240342dff3b1b418e547d56e1415e5b5704a98b6a5c168e5adf8b324dd`.
The diagnostic continuation is available, but is not promoted as a better policy.
Use its explicit detector-off mode for any matched continuation/evaluation.

## Terminal statistics implementation

After this run completed, the coordinator gained one summary after each shared
update and a final refreshed summary at shutdown, including when no dashboard
server is running. An example using this run's final data is:

```text
[PPO STATS] episodes=355 overall_mean=3.76s recent100_mean=3.93s recent_n=100 max=10.93s time_limits=0 optimized/s=33.6 elapsed=15:03 stats_age=0.0s
```

`overall_mean` pools all completed episodes in this run, across workers.
`recent100_mean` pools the latest up to 100, ordered by completion; `recent_n`
states the actual count. Unfinished episodes and previous runs are excluded.
The duration is the existing pause-aware active reward duration. Time-limited
episodes are included in the averages and counted separately in `time_limits`.
Before the first episode, means/max display `n/a`, not zero survival.

The coordinator reads a small locked in-memory snapshot of the existing
five-second background aggregator. No new per-decision disk reads, captures,
inference, episode scans or synchronous metric refresh are added. `stats_age`
exposes snapshot age; the final shutdown update refreshes completed data.
Frozen diagnostics are labeled `[FROZEN STATS]` with `processed/s`, so collection
throughput is not mislabeled as optimized experience.

Validation: 16 focused survival, terminal formatting, parallel optimizer and
speed-analysis tests passed. Coverage includes pooled means, exact latest-100
versus graph-boundary values, empty runs, incomplete log lines, cached snapshots
without file access and frozen/training labels. No new live game was launched
to validate this display-only change. Existing experiment outputs were not edited.

Machine-readable analysis:
[run and model details](speed3_15m_analysis_20261031_01/run_analysis.json),
[full latency distributions](speed3_15m_analysis_20261031_01/speed_comparison.json).

## Next step

The setup passed this engineering diagnostic and is a candidate for a longer
monitored detector-off continuation. Keep evaluating policy quality separately:
the recorded survival remains short and the late recovery is inconclusive.
Neither further capture rewriting nor new learning settings are justified by
this run alone. Keep the source and periodic snapshots for matched evaluations.
