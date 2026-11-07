# Publication checks: 7 November 2026

This update publishes the BC actor repair, measured pipeline defaults, diagnostic
instrumentation and isolated alternatives. Checkpoints, recordings, maps and run
reports remain local. No long training was started for publication.

## Checks repeated before publication

- BC workflow discovery: 42 tests passed, including export/review, balanced retry,
  actor repair, candidate selection and serialization checks.
- Recovery discovery: seven tests passed, including checkpoint integrity, bounded
  retries, shared time budget, cancellation and pooled map plots.
- Resume-spacing tests: seven passed, including the actual verifier's four-frame
  requirement, focus/pause/menu rejection and conservative timestamp fallback.
- Pipeline CUDA checks: 12 passed, including update parity, passive Adam behavior,
  temporal observation parity, transfer-buffer lifetime and instrumentation.
- Source syntax, PowerShell launcher parsing and Git whitespace checks passed.

The recovery tests require unittest discovery from their directory because they
import the sibling `recovery` module. An initial dotted-module invocation failed
that import; the documented discovery invocation passed without source changes.

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -m unittest discover -s development/staged/multimap_bc_20261106 -p 'test_*.py'
& $py -B -s -m unittest discover -s development/experiments/multimap_classifier -p test_recovery.py
& $py -B -s -m unittest development.staged.latency_candidates_20261107.test_resync_spacing
& $py -B -s development/staged/pipeline_optimizations_20261106/check.py --cuda
```

## Existing live evidence and limitations

The 30-minute and 15-minute post-BC runs completed with all four learning audits
passing and clean finalization. Subsequent short probes checked normal Adam
counter placement, resume spacing and five-worker scaling. See the detailed
[latency results](development/experiments/latency_tail_20261107/README.md).
The latest selected continuation is the 15-minute run's checkpoint, supplied
explicitly because diagnostic runs are excluded from automatic selection.

Four workers remain the default. Rare capture stalls remain possible, and a fresh
guard capture does not by itself make an earlier policy decision fresh. Bounded
recovery is not a guarantee against all failures. Timings measure command dispatch,
not Unity's visible response; these tests do not establish learning superiority.

The full historical test discovery was not rerun for this publication. Known
baseline failures are documented in [the previous publication checks](PUBLICATION_CHECKS_20261106.md);
the entire repository is not claimed to be test-clean.
