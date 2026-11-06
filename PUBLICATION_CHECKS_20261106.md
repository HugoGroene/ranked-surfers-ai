# Publication checks: 6 November 2026

Scope: current multimap, command-history, capture/latency and offline BC work on
`headless-parallel`. Source and documentation only; current model weights, maps,
recordings, BC exports and generated reports remain local. The BC training process
was already running during publication; its Python/JavaScript sources were not edited.

## Offline checks

Checks ran with CUDA disabled and single-thread CPU library settings. No new live
gameplay or benchmark was started during the running BC job.

| Suite | Tests run | Result |
|---|---:|---|
| core | 625 | 49 historical cases fail (50 assertions); 3 skipped |
| legacy | 60 | Passed |
| command_history | 17 | Passed |
| multimap_classifier | 30 | Passed |
| temporal_history | 52 | Passed |
| road_vision | 34 | Passed |
| hybrid_history_20261102 | 19 | Passed |
| fixed_spacing_workers_20261103 | 8 | Passed |
| history_training_compare_20261103 | 28 | Passed |
| multimap_bc_20261106 | 27 | Passed |

All 11 selected JavaScript syntax/native-copy/visible-port checks passed. Python
source syntax and Git whitespace checks passed. The selected source was scanned
for common credential patterns, and generated datasets/checkpoints were excluded.

The full core discovery is **not green**. Its 49 failing test cases produce 50
assertion failures across four historical modules:
`test_ppo_overnight_v176`, `test_ppo_pause_settle_v177`,
`test_ppo_pause_state_v179`, and `test_ppo_resync_uncertain_v178`.
These fixtures require old trainer hashes or lack newer verified-pause references.
The exact same 49 cases were rerun against an isolated source archive of the
previous commit `5443145`; they produced the same 50 assertion failures. They are
pre-existing, not newly introduced regressions, and have not been hidden or disabled.
Current runtime guard/settled-learning regression tests in core discovery passed.

Detailed logs are local under `reports/github_publish_validation_20261106/`.
Historical installer tests also depend on local backup fixtures; a clean clone is
not guaranteed to reproduce every legacy installer test without those assets.

## Existing live evidence (before BC training)

Candidate classifier `multimap_20261106_221921_183521` was activated after
`latency_probe_20261106_231119_982942`: four workers, both maps, 86 seconds total,
two shared PPO updates and 13 completed training episodes. All four learning audits
passed; each worker restored pause history. No hard guard stop, forced cleanup or
remaining owned process was reported. Source PPO checkpoint unchanged.

Ordinary image-request-to-command latency: 34.0 ms median / 38.3 ms p95 over 987
samples. Reset/resume tails were higher. This is not Unity response latency or
proof of overnight reliability. The classifier's 633/634 validation agreement
includes a visually apparent mislabeled menu image; four static menu hashes overlap
classifier splits. Validation selected the epoch and is not an independent test.

The prepared BC dataset passed all exported array checksums, shape/range checks,
recording-session split isolation and source-checkpoint checksum verification.
It contains 101,755 training and 8,214 validation examples, using human-reviewed
state corrections and the authorized three-second end trim. BC training results
and subsequent gameplay improvement were still pending at this publication snapshot.
