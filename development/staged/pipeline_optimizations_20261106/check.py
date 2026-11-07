"""Small CPU correctness checks. Optional CUDA transfer tests; never a game.

Refuses to run while a trainer/recorder/evaluator is active or RAM is scarce.
The static suite in test_static.py has no Torch/NumPy dependency and can run now.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', action='store_true')
    a = p.parse_args()
    os.environ['RS_PIPELINE_EXPERIMENT']=''  # Test the unmodified optimizer against each alternative.
    from preflight import require_available
    require_available()
    from development.staged.pipeline_optimizations_20261106.adapter import verify_sources
    verify_sources()
    import unittest
    import test_runtime
    test_runtime.CUDA = a.cuda
    began = time.monotonic()
    from development.experiments.latency_tail_20261107 import test_instrumentation
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in (test_runtime,test_instrumentation)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = ROOT/'reports/pipeline_optimization_checks'
    output.mkdir(exist_ok=True)
    from preflight import source_fingerprints
    report = dict(passed=result.wasSuccessful(), tests=result.testsRun, skipped=len(result.skipped),
        cuda=a.cuda, seconds=time.monotonic()-began, finished_wall=time.time(),
        sources=source_fingerprints())
    dest = output/('cuda.json' if a.cuda else 'cpu.json')
    dest.write_text(json.dumps(report, indent=2))
    print('CHECK REPORT:', dest)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
