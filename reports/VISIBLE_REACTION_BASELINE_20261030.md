# Original visible MSS path measurement

Requested original-path baseline only; no alternative capture backend implemented.
Frozen policy SHA256 779d295ec4fe04e569511987c3867ac8d68cf8a22eb12c547f028c591763907b,
same weights used in preceding headless reaction checks. Original visible MSS fast
capture and legacy desktop crop, reuse-observation guard enabled, uncapped decisions,
5ms requested key hold, inference warmed before first reset. No training updates.

Result file: visible_original_reaction_20261030_01/result.json.
30.031s measured after loading/first reset, 633 decisions, six completed episodes,
485 actual sent-command samples. No guard hard stop; source checkpoint unchanged.

| Metric | Median | p95 | Maximum |
|---|---:|---:|---:|
| MSS capture call | 13.62ms | 16.03ms | 18.29ms |
| Policy capture request to command issue | 27.58ms | 53.73ms | 75.05ms |
| Policy capture complete to command issue | 12.56ms | 40.24ms | 62.98ms |
| Guard capture complete to command issue | 12.46ms | 18.48ms | 19.94ms |
| Playwright command round trip | 16.36ms | 21.60ms | 24.52ms |

Decision throughput over the explicit measurement window was 21.08/s including
subsequent deaths/resets. Command-request latency excludes later command completion
and game response. It is not exact GPU presentation-to-Unity response time.

Recent six-worker warmed headless checks measured per-worker request-to-command
medians 89.4–96.8ms, p95 188.0–192.3ms, maxima 238.4–263.3ms. Their screenshot calls
historically cost ~50–70ms, versus ~14ms MSS here. The headless path has an actual
per-worker reaction slowdown, not just a slow preview. Capture and extra safety
recaptures are strong candidates; this is not proof capture alone explains every
tail. Different framing (legacy here, canvas_v1 headless), stochastic episodes,
predict-vs-collector call paths and worker counts prevent treating this as a strict
single-variable benchmark. All weights, observations and guards stayed unmodified
for this original-path measurement; timing wrappers only read existing stage times.
