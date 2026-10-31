# Audit patch live checks — 31 October

Local Practice only; Electron shared-memory capture, canvas_v1, detector OFF.

## Completed

- `audit_patch_live_3workers_20261031_01`: COMPLETE, four shared updates,
  1,536 optimized transitions, 31.27/s including finalization. All three
  PASS_SETTLED_LEARNING; no discarded partial transitions or guard stops.
  Final checkpoint includes all 1,536 transitions; no finalization errors;
  source checkpoint unchanged. Worker seeds are independently assigned.
- `ai_data/rl_v2_runs/audit_patch_eval_smoke_20261031_01`: three deterministic
  episodes completed. Verified durations 3.26, 3.80, 4.82 seconds; wall durations
  separately recorded as 3.78, 4.24, 5.22. Three death penalties recorded.
  Two terminals used ML/visual preflight and one inherited template evidence.
  No detector requests, no guard stops. 333 actions: NONE 134, JUMP 125,
  ROLL 74, LEFT/RIGHT zero. This is a functional smoke test, not a quality estimate.

The frozen evaluation used the preserved two-hour checkpoint, not the new smoke
training checkpoint. No source model was overwritten.

## Remaining validation

1. Matched 100-episode deterministic source/final evaluations, same detector-off
   canvas path and settings. Run separately in a terminal; do not compare older
   wall-duration means directly with verified-progress durations.
2. Visually verify that LEFT/RIGHT/JUMP/ROLL inputs cause their intended Unity
   movements. Dispatch logs alone do not establish action consumption.
3. Review terminal screenshots for false positives; inherited template acceptance
   remains distinct from ML/visual preflight. Logging does not validate thresholds.
4. Independently check game progress against wall time under load. Fresh images
   and passing ledger audits do not establish unique Unity ticks.
5. Human recorder smoke test and visible-browser evaluation framing parity.
6. Long-duration reliability and interruption recovery remain distinct from this
   short successful cleanup/save check. No 48-hour reliability claim is made.

## Learning interpretation

Removing detector inputs changes the policy's input distribution and may require
adaptation. It is not isolated as the cause: historical detector-on models also
rarely used lateral actions. The two-hour run's quartile means rose modestly, but
training statistics alone cannot establish frozen-policy improvement. Keep the
detector off as requested; diagnose action effectiveness and lane avoidance before
changing rewards or architecture.
