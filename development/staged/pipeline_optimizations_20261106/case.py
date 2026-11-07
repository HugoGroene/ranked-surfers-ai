"""Spawn-safe experimental worker entry; identical maps/guards/learning settings."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'development/experiments/multimap_classifier')]
os.environ['RS_COMMAND_HISTORY_ACTIVE'] = '1'
import stability_case
from development.staged.pipeline_optimizations_20261106.adapter import install
install()

if __name__ == '__main__':
    stability_case.run_training.main()
