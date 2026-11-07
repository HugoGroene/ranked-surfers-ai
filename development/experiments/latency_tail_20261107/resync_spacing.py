"""Measured resume cadence; retain four checks and >=33ms capture spacing."""
import ast
import asyncio
import inspect
import json
import math
from pathlib import Path
import textwrap
import time


class CaptureSpacing:
    def __init__(self, clock=time.perf_counter, sleep=asyncio.sleep):
        self.clock, self.sleep = clock, sleep
        self.last = self.invoked = None
        self.starts = []
        self.waits = []
        self.fallbacks = 0

    async def before(self):
        began = self.clock()
        if self.last is not None:
            while True:
                remaining = self.last + .033 - self.clock()
                if remaining <= 0:
                    break
                await self.sleep(remaining)
        self.invoked = self.clock()
        self.waits.append(self.invoked - began)

    def captured(self, timing):
        now = self.clock()
        stamp = (timing or {}).get('request_started_at')
        # Accept only this capture's clock, not stale metadata from a preceding
        # capture. Otherwise use completion, retaining a conservative full gap.
        valid = (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                 and math.isfinite(stamp) and self.invoked <= stamp <= now)
        if valid:
            self.last = float(stamp)
            self.starts.append(float(stamp))
        else:
            self.last = now
            self.fallbacks += 1

    def report(self):
        return dict(request_intervals_ms=[1000*(b-a) for a,b in zip(self.starts,self.starts[1:])],
                    waited_ms=[1000*v for v in self.waits], fallback_captures=self.fallbacks,
                    minimum_spacing_ms=33, required_frames=4)


def replace_spacing(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    old_wait = ast.parse('if attempt:\n    await asyncio.sleep(.033)').body[0]
    old_capture = ast.parse('legacy, menu_score, _ = await self._capture()').body[0]
    old_init = ast.parse('previous_native_present = None').body[0]
    counts = dict(wait=0, capture=0, init=0)
    class Replace(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node) == ast.dump(old_wait):
                counts['wait'] += 1
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(old_init):
                counts['init'] += 1
                return [node, *ast.parse('spacing = _CaptureSpacing()\nself._candidate_resync_spacing = spacing').body]
            if ast.dump(node) == ast.dump(old_capture):
                counts['capture'] += 1
                return [*ast.parse('await spacing.before()').body, node,
                        *ast.parse("spacing.captured(getattr(self, '_headless_capture_timing', None))").body]
            return self.generic_visit(node)
    tree = Replace().visit(tree)
    if counts != dict(wait=1,capture=1,init=1):
        raise RuntimeError('Resume source changed; revalidate resume cadence: '+str(counts))
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _CaptureSpacing=CaptureSpacing)
    exec(compile(tree, '<measured resync spacing>', 'exec'), namespace)
    return namespace[original.__name__]


def install(trainer):
    if getattr(trainer, '_measured_resync_spacing', False):
        return
    original_build = trainer.build_env_class
    def build(*args, **kwargs):
        base = original_build(*args, **kwargs)
        owners = [c for c in base.__mro__ if c.__module__ == trainer.__name__
                  and 'refresh_after_ppo_optimizer' in vars(c)]
        if len(owners) != 1:
            raise RuntimeError('Expected one base resume verifier')
        owner = owners[0]
        owner.refresh_after_ppo_optimizer = replace_spacing(owner.refresh_after_ppo_optimizer)
        class MeasuredResume(base):
            def __init__(self, *a, **kw):
                self._candidate_resync_rows = []
                super().__init__(*a, **kw)
            def refresh_after_ppo_optimizer(self):
                self._candidate_resync_spacing = None
                began = time.perf_counter()
                try:
                    return super().refresh_after_ppo_optimizer()
                finally:
                    spacing = self._candidate_resync_spacing
                    if spacing is not None:
                        self._candidate_resync_rows.append(dict(spacing.report(),
                            verification_wall_ms=(time.perf_counter()-began)*1000))
                        self._candidate_resync_rows = self._candidate_resync_rows[-512:]
            async def _close_browser(self):
                try:
                    return await super()._close_browser()
                finally:
                    (Path(self.output_dir)/'resync_spacing.json').write_text(
                        json.dumps(self._candidate_resync_rows,indent=2))
        return MeasuredResume
    trainer.build_env_class = build
    trainer._measured_resync_spacing = True
