# One-worker background diagnostic and preview check — 31 October 2026

## Completed one-worker run

`reports/v612_speed1_detectoroff_5m_20261031_135701` completed 42 updates,
5,376 optimized transitions in 307.25 active seconds: **17.50 optimized/s**.
Its learning audit passed, with 50 terminal transitions, no hard guard stop,
no discarded partial transitions and an unchanged source checkpoint.

The run continued the completed three-worker 15-minute detector-off checkpoint.
It used one headless Electron worker, separate inference/host, shared-memory
frame/rollout transfer, dense observations, OpenCV threads 1 and no dashboard.
Terminal overall/recent summaries printed successfully; the saved Matplotlib
curve opened after training finished. This is an engineering check, not a
controlled learning-quality comparison: the starting model, duration and
aggregate rollout (128 instead of 384) differ from the three-worker run.

| Metric | One worker, five minutes | Three workers, fifteen minutes |
| --- | ---: | ---: |
| Optimized transitions/s | 17.50 | 33.57 |
| Steady request-to-command median / p95 | 25.64 / 33.59 ms | 31.57 / 38.27 ms |
| Completed episodes | 50 | 355 |
| Overall mean active survival | 3.9641 s | 3.7593 s |
| Maximum active survival | 7.0351 s | 10.9253 s |
| Median whole-system CPU utilization | 9.12% | 23.95% |
| Median / maximum sampled GPU utilization | 9 / 73% | 29 / 86% |
| Median total NVIDIA memory | 2,445 MiB | 3,611 MiB |
| Median available system RAM | 15.40 GiB | 10.54 GiB |

Steady timing excludes the first 20 movement commands per worker. The one-worker
sample contains 3,390 steady commands, p99 35.44ms, maximum 85.31ms and none above
100ms. The full 3,410-command sample has one above 100ms, maximum 107.52ms.
These endpoints measure screenshot-request to command issue, not Unity response.

Resources were sampled during active training, before the Matplotlib window
opened. CPU/RAM/GPU figures describe the whole PC, including other applications,
not training-only allocations. Low median GPU utilization still includes bursts;
gaming/Resolve coexistence was discussed but has not been measured. One worker
delivered about 52% of measured three-worker aggregate throughput, not two-thirds.
The 0.20-second survival difference does not establish a better model.

Completed checkpoint:
`reports/v612_speed1_detectoroff_5m_20261031_135701/latest.zip`.

SHA-256: `bf49a7fd2a7325c9689d9e1394d2ddc7c19c188cb50a9ba73884796c4cb0d4ca`.
ZIP CRC check passed. Saved total timesteps 1,882,398, update counter 68,625,
detector mode off. All local checkpoints and raw experiment files remain local.

## User-stopped three-worker preview check

`reports/v612_speed3_preview30_5m_20261031_140835` requested five minutes,
three workers, preview port 8001 and preview cap 30. The user stopped it early
after reporting that the viewer looked fine. This is user-observed viewer
functionality, not a completed five-minute performance or learning experiment.

At inspection, `parallel_training.json` was still `RUNNING`, with 11 committed
updates / 4,224 optimized transitions and 131.65 reported active seconds. No
final completion status or final learning-audit set was present. Preserve that
unfinalized state; do not edit the report to imply successful completion.

The two-minute saved checkpoint is CRC-valid, SHA-256
`086c11c2570d3048830368e907b114e2327b34c44233401e20c43c546fba3024`, total
timesteps 1,886,622 and update counter 68,680. It is preserved separately and
is not promoted over the last completed diagnostic.

The saved preview control ended with `enabled=false, fps=30`. Therefore the
configured 30 FPS ceiling and the user's smooth-view report do not prove
continuous 30 unique preview frames/s or establish preview overhead. Previews
reuse collected frames; actual unique rate is limited by collection. Browser
preferences may override the initial CLI cap and must be checked in the UI.

## Current practical configuration

Use three workers for dedicated training and one as the lower-load background
candidate. Both keep per-worker inference and Electron hosts; avoid selecting
shared inference/host or compact storage just for memory savings when reaction
speed is the priority. Sustained multitasking and longer-run learning quality
remain separate validation questions.

The current UI supports a 1–30 FPS preview cap and metrics-only mode. Terminal
statistics remain available without the dashboard. Successful no-dashboard
training opens the saved survival curve after cleanup by default; override with
`--end-plot show|save|off`. See [usage](../README_PPO_SCALING.md).
