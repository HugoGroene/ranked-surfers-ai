# Ranked Surfers AI — rules for Claude sessions

Read first: [PROJECT_STATUS_AND_RESEARCH.md](PROJECT_STATUS_AND_RESEARCH.md),
[README.md](README.md), [README_DEMONSTRATIONS_V612.md](README_DEMONSTRATIONS_V612.md).
Their branch/commit metadata is stale; trust code and raw reports over prose.

## Scope
- Local Practice copy only (`http://localhost:8000` via `server.py`). Never automate the public/ranked site.

## Workspace boundary
- Develop only in this repo (`ranked-surfers-ai-new`, branch `claude-development`).
- `C:\Users\Hugo\Documents\ranked-surfers-ai` (original repo) is a read-only reference/backup.
  Ask, naming the specific files and why, before reading anything there. Never write, move,
  delete, copy data out of, train in, or save anything there without explicit permission.

## Do not change without explicit approval
- PPO architecture, observation schema (12x288x512 image + 557 aux), action mapping
  (0 NONE, 1 LEFT, 2 RIGHT, 3 JUMP, 4 ROLL), rewards, gameplay control, guard semantics.
- Checkpoints and recordings anywhere. New outputs go to new, uniquely named paths.
- The 130 ms demo label-validity bound. Never auto-approve demonstrations.
- Keep the recorder backward-compatible with existing V1.3 replays (manifest fields are additive).
- Never lose a recorded run: runs autosave as candidates; never auto-approve, overwrite or delete.
- Do not delete V2–V5 modules: `PPOV61Env` inherits a 10-level class chain through them.

## Way of working
- Propose a plan and wait for approval before editing. One variable per experiment; small steps.
- Do not launch the game, browsers, recorder, probe, training or evaluation; the user runs live tests.
- User workflow update: when explicitly authorized, short live tests may be run
  by the assistant. For tests longer than a few minutes, give the user commands
  to run in a separate terminal, including the local server command. Do not
  spend assistant credits polling long runs; inspect saved artifacts afterward.
- Current background path is headless Chromium with local browser-image capture;
  VDD is disabled and should stay disabled unless the user explicitly requests it.
  Physical NVIDIA monitor stays 1920x1080 at 144 Hz; do not change brightness,
  HDR or colour settings. canvas_v1 uses whole 1920x945 canvas to 512x288 policy.
  Legacy capture remains available explicitly for reproduction.
- Up to six headless worker replicas feed one shared settled PPO optimizer.
  Preserve source rollout 128 per worker and LR .0003 unless explicitly changed;
  aggregate rollout grows with worker count (768 with six). Dashboard at localhost
  8001 reuses existing frames/telemetry; never add desktop captures for previews.
  Persistent graph dots are fixed completed-time samples, not a moving bucket.
  Thirty-minute user-run diagnostic is pending; do not claim learning improvement.
- Label claims: [repo] verified here, [local] verified in original repo, [doc] docs only, [hyp] hypothesis.
- PPO baseline: original repo `ai_data/rl_v2_runs/v612_overnight_next_420min_press5_20261027_094116/latest.zip`
  (SHA-256 6e49da55…e773f0): 4.61 s mean [95% CI 4.26–4.99], 0 LEFT/RIGHT. A 240-min checkpoint
  scored 4.42 s, so differences under ~0.5 s are within noise.
- Comparisons: 100 deterministic episodes, identical flags, bootstrap CIs, action counts.
- Pass explicitly: `--press-delay-ms 5`, `--state-model ...screen_state_candidate_v2.pt`,
  `--root ai_data\demonstrations_v13` (script defaults differ).

## Offline tests
Interpreter (permitted, run against this clone; `-B` writes no bytecode):
`C:\Users\Hugo\Documents\ranked-surfers-ai\.venv\Scripts\python.exe -B -s -m unittest test_rs_v612_demo test_rs_v612_demo_diagnostics test_rs_v612_demo_recording test_rs_v612_guard test_rs_v61 -v`
Never install packages without approval.

## Demo replay analysis (read-only)
`python tools/analyze_demo_run.py --latest ai_data\demonstrations_v13 --count 3`

## Git
- Never `git add .`/`-A`; stage explicit files. `ai_data/demonstrations_*` is not ignored.
- Commit/push only when asked.

- Reaction latency is now a priority: six-worker screenshot-request-to-command p95 ~190ms even without previews. A fresh guard frame can accompany an older policy decision. Do not equate aggregate optimized throughput with per-worker response speed, or promise screenshot timing equals GPU presentation / Unity action response. Policy warmup is implemented; steady-state capture and guard alignment still need optimization. Metrics-only preview mode stops image fetches and shared-worker thumbnail encoding, not game observations.
