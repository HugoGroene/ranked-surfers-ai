# Visible evaluation capture (4 November)

Visible evaluation now defaults to MSS desktop capture on a dedicated thread.
The previous visible path requested a full 1920x945 Chromium capturePage bitmap
for every capture. Removing those requests is a candidate fix for whole-game
stuttering; offline checks cannot establish the live performance improvement.
The headless training path is unchanged.

The same fixed Electron window, page, canvas dimensions, render scale, policy
crop, RGB layout, frame ages, command history and model are retained. Desktop
capture checks foreground ownership and client geometry before and after each
read. Keep the window unobscured, including by overlays. Losing foreground stops
evaluation. Do not minimize, move or resize it. No desktop or monitor settings
are changed. Reads exceeding 50 ms are discarded, with the existing bounded
retry behavior; the capture pump retains its own delivery bound.

A new read means a newly acquired desktop snapshot, not necessarily a distinct
Unity frame. Its request duration does not prove GPU presentation freshness.
Desktop composition, color management and overlays can differ from offscreen
capture. Inspect the saved capture_preview.jpg as well as gameplay.

Each Electron output directory gains visible_render_timing.json at close:
bounded requestAnimationFrame interval histogram, median/p95/p99, maximum and
counts above 25/50 ms. This covers startup, menus and gameplay; it measures page
scheduling, not actual Unity simulation or monitor FPS. The updated probe measures
both the outer page and the arena iframe, selecting the iframe with canvas#game
for its headline statistics. Earlier reports measured only the outer page and
cannot establish Unity smoothness. There is no per-frame
telemetry IPC. Page navigation resets this diagnostic histogram.

Run the current command-history checkpoint:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    .\development\experiments\command_history\evaluate_visible.ps1 -Episodes 10
```

Add `-CompareCapture` to run desktop capture followed by the old capturePage
backend, using the same checkpoint and episode count. Compare animation
smoothness and interval statistics, not just survival (different game samples).
`-Checkpoint` can select another compatible command-history checkpoint. Each
invocation creates new report folders. Environment selection is restored at exit.

visible_episode_trace.json also records reset duration, first eight requested
actions per episode and per-episode action counts. It distinguishes waiting for
the initial menu from a running policy choosing NONE, without changing controls.
The console report lookup uses the known electron/visible_render_timing.json
path rather than traversing the browser's temporary profile cache.

Visible startup now holds the first HTML Practice activation until the evaluator
has loaded and warmed the policy. The first reset warms capture/classification,
then releases the click and requires the existing two consecutive confident
GAMEPLAY observations, low menu-template scores and startup grace. It adopts
that first Practice run instead of intentionally waiting for its death. If the
build instead opens a confirmed menu, it delegates to the existing guarded R
restart. Ambiguous startup stops; subsequent resets retain the normal behavior.
This change is process-local to visible evaluation, not headless training or
public-site evaluation. The first run still needs live validation.

No live game was launched during implementation. Twenty-two offline tests cover
visible startup, size/focus rejection, byte ownership, BGRA layout, serials,
slow-read rejection and telemetry summary. The Electron host passes Node's
syntax check. MSS 10.2.0 is already installed; no dependencies were installed.
