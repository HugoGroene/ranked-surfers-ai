# Headless shared PPO checks, 2026-10-30

All runs read the preserved large-BC source with SHA256
9963b7229d2aa5dae4514cad359c1f904d6ee24f721137ab298d55d7618c5936.
Source unchanged in every report. No checkpoint promotion.

| Workers | Report directory | Updates | Optimized transitions/s | CPU average | GPU average/peak | Peak VRAM MiB | Peak system RAM |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | headless_shared_ppo_one_20261030_01 | 2 | 6.70 | unmeasured | unmeasured | unmeasured | unmeasured |
| 2 | headless_two_repeat_20261030_01 | 2 | 11.29 | 14.9% | 14.2% / 72% | 3859 | 61.1% |
| 3 | headless_three_repeat_20261030_02 | 2 | 15.03 | 20.6% | 20.8% / 76% | 4378 | 62.0% |
| 4 | headless_four_repeat_20261030_01 | 2 | 16.97 | 23.7% | 25.2% / 77% | 5123 | 68.7% |

All completed runs passed every worker ledger audit, with no discarded
transitions or guard hard stops. Post-update resync, menu/death/reset handling
and atomic checkpoint saving occurred. The 2/3/4-worker checkpoint ZIP CRCs
passed. Preview HTTP returned 200 and exposed four worker tiles and runtime data.
The preview server closes when the run finishes.

The first three-worker attempt (headless_three_repeat_20261030_01) failed
before training: one page exposed no visible Unity canvas. That artifact is
preserved. Added startup error screenshots/page errors, one bounded pre-action
reload, and a two-second stagger between workers. The retry passed.

These are short smoke/scaling checks, not sustained throughput or learning-quality
benchmarks. Active time excludes worker/model loading, includes first reset and
final cleanup/checkpoint. Resource samples include loading and optimization.
Episodes are stochastic and budgets vary by worker count (256 transitions per
worker in each completed repeated-update check). Source n_steps=128 is retained
per worker: aggregate rollouts grow 128/256/384/512. LR remains 0.0003.
Two updates only establish one post-update continuation; long-run stability and
headless screenshot age semantics require further validation.

The marginal four-worker gain over three is about 13%, with higher RAM/VRAM.
More workers are not yet justified by these samples. The runner currently permits
up to four headless workers. Preview refreshes existing policy images at ~1 Hz;
no extra game capture. It includes phase/count/image age, plus aggregate updates,
active runtime and optimized throughput.

VDD remains disabled (problem code 22, persisted ConfigFlags=1).
No VDD-named startup command or task was found. Restart behavior has not been
verified by reboot. Physical display was not reconfigured for headless runs.

Six-worker follow-up: headless_six_repeat_20261030_01 completed two shared updates, all 1536 transitions optimized in 82.689 active seconds (18.576/s). All six ledger audits passed, zero discarded transitions or guard hard stops, source unchanged, checkpoint ZIP CRC passed. CPU averaged 38.1% (65.4% peak overall, busiest logical core 86.3%); sampled GPU averaged 35.1% and peaked at 81%; VRAM peaked at 6954 MiB (~6.79 GiB); total system RAM peaked at 85.6%. About 9.5% more throughput than four workers, at substantially higher memory use. Runner limit extended to six. Same short-run/update-size caveats apply; no long-run stability or learning-quality claim.
