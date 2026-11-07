"""Evaluation uses the same optional pack optimization, never a different model."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'development/experiments/command_history')]
from development.staged.pipeline_optimizations_20261106.adapter import install

if __name__ == '__main__':
    install()
    runpy.run_path(str(ROOT/'development/experiments/command_history/evaluate.py'), run_name='__main__')
