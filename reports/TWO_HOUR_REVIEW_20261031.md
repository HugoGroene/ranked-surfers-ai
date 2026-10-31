# Two-hour run review and applied changes

Run: `v612_optimized3_detectoroff_2h_20261031_142111`. This review uses saved artifacts only; no live test or frozen-policy evaluation was launched.

## Outcome

COMPLETE: 644 shared updates, 247,296 optimized transitions in 7,206 seconds (34.32/s). All three transition audits were rerun and passed. No discarded partial rollouts or guard hard stops. Final checkpoint ZIP integrity passed; saved total timesteps 2,133,918. SHA256: `fcfdb5091f61d84b508e108263e71f5c344591266e90923c98288c04ac087661`.

2,706 completed episodes: overall mean 3.973s, median 3.757s, p95 7.017s, best 14.601s. Eighteen episodes reached 10 seconds; none reached the 300-second time limit.

Equal episode-count quartile means: 3.774, 3.902, 4.077, 4.139 seconds. This is a modest favorable training trend, but first/last 100 means were 3.681/3.761 seconds. These changing-policy stochastic samples do not establish a better frozen policy. A matched evaluation is still needed; do not compare them directly with historical wall-clock deterministic means.

Actor choices: NONE 81,543; LEFT 52; RIGHT 1,677; JUMP 100,500; ROLL 60,433. Lateral choices remain rare (approximately 0.71%) and highly asymmetric. Their causal usefulness cannot be determined from counts alone. Do not add action-specific rewards based on this observation.

Full-run policy screenshot-request-to-command latency: median 32.09ms, p95 38.62ms, p99 54.50ms, maximum 141.36ms. This supports sustained typical performance, but does not measure Unity input consumption and includes startup/recovery tails. Raw analysis is in `TWO_HOUR_REVIEW_20261031.json`.

## Applied work

The staged 16-file audit patch was applied with source checks and backups under `development/patch_backups/audit_20261031_184159_653272`. Detector-off defaults, input contracts, independent seeds, evaluation clocks/status, terminal-source tags, finalization protection and BC compatibility are now in the active source. Existing experiment outputs and checkpoints were preserved.

Training limits now accept 2,880 minutes (48 hours), including the parallel worker argument. The coordinator still stops at whichever comes first: total step budget or wall time at a completed update. For three workers with 128-step rollouts, a sufficiently large example budget is 12,000,000 steps (divisible by 384). A 48-hour allowance is not a reliability claim. Periodic checkpoints remain necessary.

89 post-application offline tests passed across collector, coordinator, input contracts, framing, duration limits, BC and recording checks. `git diff --check` passed. The earlier staged package had 117 passing offline tests. No live validation was performed and no code was committed/pushed in this task.

## Live validation still required (not performed)

1. Short three-worker smoke test from a COPY/continuation of the preserved latest checkpoint: verify seeds, unavailable detector inputs, pause/resume, reset, terminal-source tags and checkpoint saving, and compare reaction latency against the above baseline.
2. Matched frozen-policy evaluation of the source and final checkpoint with identical canvas/detector-off inputs and the new verified-time metric. Confirm evaluation completion/status and retained wall-clock metrics. This is required before claiming learning improvement.
3. Short action-effect diagnostic: correlate LEFT/RIGHT/JUMP/ROLL commands with visible responses and collisions; review the strong LEFT/RIGHT imbalance and latency tails.
4. Review template-only post-action deaths against fresh ML/visual evidence. Terminal-source logging is added, but thresholds and acceptance behavior were deliberately not changed without evidence.
5. Validate game progression against an independent time/distance signal under load; a passing ledger audit does not independently measure Unity time.
6. Short detector-free recorder check before any future BC collection. Verify clean framing, unavailable auxiliary fields and label alignment; original demonstrations remain untouched.
7. After the smoke/evaluation gates, a longer soak can establish multi-hour stability of the patched version. Neither parser tests nor the earlier two-hour run establish 48-hour reliability.

No architecture, reward coefficient, exploration setting or stale-action redispatch change is justified by the saved aggregates alone. The next priority is matched evaluation and targeted action/terminal diagnosis, not another infrastructure rewrite.
