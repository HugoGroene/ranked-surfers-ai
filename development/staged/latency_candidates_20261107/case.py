"""Spawn-local candidates layered on the measured production baseline."""
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'development/experiments/multimap_classifier')]
os.environ['RS_COMMAND_HISTORY_ACTIVE'] = '1'
import stability_case
from development.staged.pipeline_optimizations_20261106.adapter import install as install_baseline
install_baseline()
from development.staged.latency_candidates_20261107.runtime import install
install(os.environ['RS_LATENCY_CANDIDATE'])

# Preserve the existing per-run manifest checks and include our staged sources.
import command_adapter
original_sources = command_adapter.sources
def sources():
    result = original_sources()
    for file in Path(__file__).parent.glob('*.py'):
        result[file.relative_to(ROOT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result
command_adapter.sources = sources
stability_case.run_training.sources = sources

if __name__ == '__main__': stability_case.run_training.main()
