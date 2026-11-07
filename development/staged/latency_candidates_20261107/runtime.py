"""Isolated candidates; production code never imports this directory."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import math
from pathlib import Path
import sys
import textwrap
import time


class AsyncDeadlineWait:
    """One sleeping native waiter; never block the browser's asyncio loop."""
    def __init__(self, waiter=None):
        if waiter is None:
            from development.experiments.latency_tail_20261107.pacing import DeadlineWait
            waiter = DeadlineWait(precise=True)
        self.waiter = waiter
        try:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='capture-deadline')
        except BaseException:
            waiter.close()
            raise
        self.closed = False

    def _wait(self, deadline):
        # An executor handoff must not extend the original capture deadline.
        self.waiter.wait(max(0., deadline-time.perf_counter()))

    async def until(self, deadline):
        if not math.isfinite(deadline):
            raise ValueError('Nonfinite capture deadline')
        if self.closed:
            raise RuntimeError('Capture waiter is closed')
        if deadline <= time.perf_counter():
            await asyncio.sleep(0)
            return
        await asyncio.get_running_loop().run_in_executor(self.pool, self._wait, deadline)

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Cancelling the asyncio Future does not cancel a running native wait.
        # Drain it BEFORE closing its native timer handle (normally <=17ms).
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.waiter.close()


def paced_run(original):
    """Replace just the pump's rate sleep; preserve all capture/guard logic."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = ast.parse('asyncio.sleep(max(0,self.period-(time.perf_counter()-began)))', mode='eval').body
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.dump(node) == ast.dump(expected)]
    if len(matches) != 1:
        raise RuntimeError('Capture pump changed; revalidate staged pacing')
    replacement = ast.parse('self._candidate_wait.until(began+self.period)', mode='eval').body
    matches[0].func = replacement.func
    matches[0].args = replacement.args
    matches[0].keywords = []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<isolated capture pacing>', 'exec'), namespace)
    return namespace[original.__name__]


def pump_class(base):
    class PreciseCapturePump(base):
        run = paced_run(base.run)

        def __init__(self, *args, **kwargs):
            self._candidate_wait = AsyncDeadlineWait()
            try:
                super().__init__(*args, **kwargs)
            except BaseException:
                self._candidate_wait.close()
                raise

        async def capture(self):
            image = await super().capture()
            self.last_timing['capture_pacing_candidate'] = self._candidate_wait.waiter.mode
            return image

        async def close(self):
            try:
                await super().close()
            finally:
                self._candidate_wait.close()
    return PreciseCapturePump


def patch_pump_modules(modules):
    # Historical adapters import this file both as `live_adapter` and by its
    # package name. Patch the module globals actually used by each env closure.
    for module in modules:
        if not getattr(module.CapturePump, '_candidate_precise', False):
            module.CapturePump = pump_class(module.CapturePump)
            module.CapturePump._candidate_precise = True


def channels_last(net, torch):
    """Keep original preprocessing/normalization/softmax; change layout only."""
    net.to(memory_format=torch.channels_last)
    def prepare(module, args):
        if len(args) != 1 or args[0].ndim != 4:
            raise ValueError('Unexpected classifier input contract')
        return (args[0].contiguous(memory_format=torch.channels_last),)
    return net.register_forward_pre_hook(prepare)


def install(arm):
    if arm == 'resync_spacing':
        from development.staged.latency_candidates_20261107.resync_spacing import install as install_resync
        install_resync()
        return
    if arm in ('adam_cpu_steps','pump_cpu_steps'):
        from development.staged.latency_candidates_20261107.adam_steps import install as install_steps
        install_steps()
        if arm=='adam_cpu_steps': return
        arm='pump'
    if arm in ('policy_layout', 'fused_adam', 'policy_fused'):
        from development.staged.latency_candidates_20261107.model_runtime import install_model_features
        install_model_features(arm)
        return
    if arm not in ('baseline', 'pump', 'classifier', 'both'):
        raise ValueError('Unknown latency candidate')
    if arm in ('pump', 'both'):
        from development.experiments.temporal_history import live_adapter
        target = Path(live_adapter.__file__).resolve()
        modules = [m for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                   and Path(m.__file__).resolve() == target]
        patch_pump_modules(modules)
        print('[CAPTURE CANDIDATE] Absolute period deadline; future-frame checks unchanged',flush=True)
    if arm in ('classifier', 'both'):
        import rs_v4_screen_states as states
        cls = states.ScreenStateClassifier
        if not getattr(cls, '_candidate_layout', False):
            original_init = cls.__init__
            def initialize(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self._candidate_layout_hook = channels_last(self.net, self.torch)
                print('[CLASSIFIER CANDIDATE] CPU FP32 channels-last; original preprocessing',flush=True)
            cls.__init__ = initialize
            cls._candidate_layout = True
