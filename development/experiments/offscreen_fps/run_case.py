"""Spawn-safe opt-in renderer-cap experiment; production defaults unchanged."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
fps = int(os.environ['RS_OFFSCREEN_TEST_FPS'])
if fps not in (30, 60):
    raise ValueError('Renderer FPS must be 30 or 60')
sys.path[:0] = [str(ROOT), str(HERE.parent/'capture_pipeline'),
               str(HERE.parent/'command_history')]
import adapter
adapter.HERE = HERE
# Keep this file as multiprocessing's main module. runpy would replace __file__
# with run_training.py, causing Windows children to bypass the host override.
os.environ['RS_COMMAND_HISTORY_ACTIVE'] = '1'
import run_training

def verify_child(queue):
    from tools.ppo_electron_capture import ElectronCapture
    host = ElectronCapture.launch.__func__.__globals__['HERE'] / 'host.cjs'
    queue.put((str(host), os.environ['RS_OFFSCREEN_TEST_FPS']))

if __name__ == '__main__':
    if '--check-spawn' in sys.argv:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        child = ctx.Process(target=verify_child, args=(queue,))
        child.start()
        actual = queue.get(timeout=60)
        child.join(timeout=10)
        assert child.exitcode == 0, child.exitcode
        assert actual == (str(HERE/'host.cjs'), str(fps)), actual
        print('PASS: spawned worker selects isolated host at', fps, 'FPS; no game launched')
    else:
        run_training.main()
