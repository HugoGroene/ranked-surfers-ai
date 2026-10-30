# Reaction latency and metrics-only dashboard

Timing diagnostics added without changing rewards, classifier thresholds, policy
architecture or input dispatch. The source checkpoint for the short checks was
the preserved partial diagnostic (SHA256 779d295ec4fe04e569511987c3867ac8d68cf8a22eb12c547f028c591763907b).

## What the timings mean

The inherited key controller already records press_timestamp before calling
Playwright keyboard.press. Record the policy observation's screenshot request
start, JPEG reply and capture-complete time separately from the final guard image.
Neither a screenshot request nor its reply is an exact GPU presentation timestamp.
Command issue is not Unity input consumption or visible action response. The stats
are therefore explicitly request/reply/capture-complete to command, not proven
end-to-end reaction time. Suppressed actions and NOOPs have no dispatch sample.

Per-transition data is added to the existing learning ledger. Last-1000-command
samples are bounded in RAM; dashboard summaries are recomputed at most every five
seconds. No extra captures or per-action browser RPCs are added for measurement.

## Short live evidence

| Configuration | Policy screenshot request to command median | p95 | Maximum |
|---|---:|---:|---:|
| One worker, reaction_one_metrics_only_20261030_01 | 82.8ms | 175.0ms | 314.3ms |
| Six, no preview server/images, reaction_six_no_images_20261030_01 | 90.6–92.6ms per worker | 184.2–188.7ms | 326.6–361.1ms |
| Six, inference warmed, no images, reaction_six_warmup_20261030_01 | 89.4–96.8ms | 188.0–192.3ms | 238.4–263.3ms |

All three short runs completed and passed their learning audits; source unchanged.
Each six-worker check used one shared update/768 transitions. These are small,
stochastic samples, not a controlled statistical comparison. The one-worker check
started with previews enabled and switched off toward its end, so it is not a
fully matched images-disabled benchmark. Six-worker checks had no preview server
or image encodes for the entire run.

Before warmup, the inspected worker's first action was its worst (361.1ms from
request). Added deterministic inference warmup before worker readiness/reset.
It does not learn, send actions, alter weights or consume action-sampling RNG;
unit tests check the latter properties. This does not resolve the steady-state
latency tail: p95 remains around 190ms. Do not advertise the pipeline as extremely
fast merely because aggregate throughput is high (~19.6 transitions/s here).

The key distinction is observable: guard capture-complete-to-command p95 stayed
under 20ms in the unwarmed six-worker run, while policy capture-complete-to-command
p95 was 110–116ms. Refreshing a safety frame does not resample an already-selected
policy action from that new scene. Optimize capture and policy/guard alignment
before inferring that more long training will solve fast late-game reactions.

## Dashboard toggle

Hide previews stops new image requests and hides the grid; Show previews resumes.
The choice persists across page reloads. The loopback preview-control endpoint
writes a small shared control file; training workers poll it at their existing
one-second preview cadence and stop JPEG encoding when disabled. Status/latency
metadata, runtime statistics, hardware samples and the persistent graph continue.
In-flight requests may finish once. This controls shared-training previews; it
does not pause game rendering, observations, input guards or learning.

Browser checks verified no image requests while disabled, metrics/chart remain
visible, state persists on reload and previews can be re-enabled. The real
one-worker check verified status updates continued with images_enabled=false and
the last image timestamp stayed unchanged. Existing training processes must be
restarted to load the new worker-control code. No long run was started.
