"""Tiny stdlib-only checks. No Torch, GPU, browser or production patches."""
import asyncio
import math
import threading
import time
import unittest
from runtime import AsyncDeadlineWait, paced_run, patch_pump_modules
from types import SimpleNamespace


class FakeWait:
    def __init__(self):
        self.calls = []; self.closed = False
        self.started = threading.Event(); self.release = threading.Event()
    def wait(self, seconds):
        self.calls.append(seconds); self.started.set()
        self.release.wait(.05)
    def close(self):
        self.closed = True


async def source_run(self):
    began = time.perf_counter()
    await asyncio.sleep(max(0,self.period-(time.perf_counter()-began)))
    await asyncio.sleep(0)  # Other sleeps must remain untouched.
    self.finished = True


async def changed_run(self):
    await asyncio.sleep(.01)


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_all_import_aliases_patched_once(self):
        class Base:
            run = source_run
        modules = [SimpleNamespace(CapturePump=Base),SimpleNamespace(CapturePump=Base)]
        patch_pump_modules(modules)
        classes = [m.CapturePump for m in modules]
        self.assertTrue(all(c is not Base and c._candidate_precise for c in classes))
        patch_pump_modules(modules)
        self.assertEqual(classes,[m.CapturePump for m in modules])

    async def test_deadline_elapsed_yields_without_executor(self):
        fake = FakeWait(); waiter = AsyncDeadlineWait(fake)
        try:
            await waiter.until(time.perf_counter()-1)
            self.assertEqual(fake.calls, [])
        finally: waiter.close()
        self.assertTrue(fake.closed)

    async def test_nonfinite_rejected(self):
        waiter = AsyncDeadlineWait(FakeWait())
        try:
            for value in (math.inf, -math.inf, math.nan):
                with self.assertRaises(ValueError): await waiter.until(value)
        finally: waiter.close()

    async def test_async_loop_remains_available_and_cancel_drains(self):
        fake = FakeWait(); waiter = AsyncDeadlineWait(fake)
        task = asyncio.create_task(waiter.until(time.perf_counter()+.2))
        try:
            for _ in range(100):
                if fake.started.is_set(): break
                await asyncio.sleep(.001)
            self.assertTrue(fake.started.is_set())
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            fake.release.set()
        finally: waiter.close()
        self.assertTrue(fake.closed)
        self.assertLessEqual(fake.calls[0], .2)
        self.assertGreaterEqual(fake.calls[0], 0)

    async def test_replacement_uses_absolute_deadline(self):
        class Recorder:
            async def until(self, deadline): self.deadline = deadline
        class Pump: pass
        pump = Pump(); pump.period = .02; pump._candidate_wait = Recorder()
        start = time.perf_counter()
        await paced_run(source_run)(pump)
        self.assertTrue(pump.finished)
        self.assertGreaterEqual(pump._candidate_wait.deadline, start+.02)
        self.assertIs(source_run.__globals__['asyncio'], asyncio)

    async def test_source_drift_refused(self):
        with self.assertRaises(RuntimeError): paced_run(changed_run)

    async def test_close_idempotent_and_use_after_close_refused(self):
        waiter = AsyncDeadlineWait(FakeWait()); waiter.close(); waiter.close()
        with self.assertRaises(RuntimeError): await waiter.until(time.perf_counter())


if __name__ == '__main__': unittest.main()
