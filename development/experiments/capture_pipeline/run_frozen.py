"""Spawn-safe opt-in wrapper around the unchanged production collector."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
if __name__=='__main__':
    if len(sys.argv)<3 or sys.argv[1]!='--arm':raise SystemExit('First argument must be --arm NAME')
    os.environ['RS_CAPTURE_EXPERIMENT_ARM']=sys.argv[2]
    del sys.argv[1:3]
if os.environ.get('RS_CAPTURE_EXPERIMENT_ARM'):
    from adapter import install_capture
    install_capture(os.environ['RS_CAPTURE_EXPERIMENT_ARM'])
if __name__=='__main__':
    from tools.train_ppo_parallel import main
    if '--benchmark-only' not in sys.argv:raise SystemExit('This wrapper only permits frozen-policy benchmarks')
    main()
