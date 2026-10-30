# Diagnostic stop and bounded freshness recovery

Failed user run: reports/v612_canvas_diag_30m_20261030_024112.
Three shared updates completed: 2304 optimized transitions, 120.966 active seconds,
19.047 optimized transitions/s. Worker 4's fourth post-update GAMEPLAY frame aged
117.8ms before collector consumption; the unchanged limit was strictly under
100ms. All four frames were ML-confirmed gameplay. This was an observation-age
safety stop, not a demonstrated bad reward or corrupted model.

The checkpoint was saved in finally: ZIP CRC passed, num_timesteps=1715358,
framing_profile=canvas_v1. Original source hash remained unchanged. The partial
checkpoint SHA256 is 779d295ec4fe04e569511987c3867ac8d68cf8a22eb12c547f028c591763907b.
It is a valid resume source retaining all three updates. Other workers abandoned
partial collection when the cohort aborted: three additional ledgered transitions
were discarded; four completed interactions were reported as not ledgered. These
did not reach the central optimizer. Original artifacts were not rewritten.

Resource samples: peak RAM 90%, peak VRAM 7154 MiB, average CPU 40.1% (peak 77.15%),
average GPU 36.5% (peak 80%). Contention may contribute to latency but is not
established as its cause. There is insufficient training/evaluation evidence to
infer learning improvement from this short run.

Fix: verified_refresh retries the complete existing four-frame resync up to two
additional times if the finished stack is stale. It never sends movement or
repeats Resume clicks, restores the paused-panel visual reference between attempts,
and immediately returns MENU/uncertain results to their existing safe handling.
The strict 100ms threshold, classifier thresholds, rewards and PPO settings stay
unchanged. Persistent staleness or an invalid future timestamp still stops safely.
Settled collection represents the total post-resume interval in its existing passive
transition; the actor is not trained as though it acted during the retry gap.

Validation: 142 offline tests passed, including exact 117.8ms fault reproduction,
bounded persistent failure, MENU/uncertainty handling, Pause-reference preservation
and invalid timestamps. Live six-worker copy run headless_freshness_fix_20261030_01
completed two updates/1536 transitions in 87.651s (17.524 optimized/s); all six
learning audits passed, no discarded transitions or guard stops, source unchanged,
ZIP CRC passed. The live run did not encounter a stale-refresh retry, so recovery
of that exact rare path is fault-tested offline, not demonstrated live yet.

Resume a new 30-minute diagnostic from the failed run's preserved latest.zip,
not by writing into its output directory. The short validation copy is separate.
If stale refresh persists despite bounded retries, investigate/reduce contention
rather than relaxing the freshness guard.
