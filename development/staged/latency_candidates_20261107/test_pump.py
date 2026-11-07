"""Idle-only integration against the real pump; fake pixels, no browser/GPU."""
import asyncio
import time
import unittest
from types import SimpleNamespace
from development.experiments.temporal_history.live_adapter import CapturePump
from runtime import AsyncDeadlineWait, pump_class


class Source:
    def __init__(self):
        self.started = asyncio.Event(); self.release = asyncio.Event()
        self.serial = 0
    async def capture(self):
        self.serial += 1
        stamp = time.perf_counter()
        if self.serial == 1:
            self.started.set()
            await self.release.wait()
        self.last_timing = dict(request_started_at=stamp,electron_serial=self.serial)
        return self.serial


class PumpTests(unittest.IsolatedAsyncioTestCase):
    async def test_does_not_deliver_inflight_frame_from_before_demand(self):
        source = Source(); prepared = []
        def callback(image,stamp,generation):
            prepared.append((image,stamp,generation)); return image
        pump = pump_class(CapturePump)(source,callback,60,token=lambda:1)
        try:
            await asyncio.wait_for(source.started.wait(),1)
            consumer = asyncio.create_task(pump.capture())
            await asyncio.sleep(0)
            source.release.set()
            image = await asyncio.wait_for(consumer,1)
            self.assertGreater(image,1)
            self.assertGreaterEqual(pump.last_timing['request_started_at'],
                                    pump.last_timing['temporal_consumer_requested_at'])
            self.assertEqual(pump.delivered_prepared[2],image)
            self.assertEqual(pump.delivery_serial,image)
        finally: await pump.close()
        self.assertTrue(pump._candidate_wait.closed)
        self.assertFalse(any(t.is_alive() for t in pump._candidate_wait.pool._threads))

    async def test_inactive_history_remains_on_demand(self):
        source = Source(); source.release.set()
        pump = pump_class(CapturePump)(source,lambda *args:None,60)
        try:
            await asyncio.sleep(.02)
            self.assertEqual(source.serial,0)
            await asyncio.wait_for(pump.capture(),1)
            serial = source.serial
            await asyncio.sleep(.04)
            self.assertEqual(source.serial,serial)
        finally: await pump.close()

    async def test_source_errors_still_propagate(self):
        async def fail(): raise RuntimeError('source failure')
        pump = pump_class(CapturePump)(SimpleNamespace(capture=fail),lambda *args:None,60)
        try:
            with self.assertRaisesRegex(RuntimeError,'source failure'):
                await asyncio.wait_for(pump.capture(),1)
        finally: await pump.close()

    async def test_native_deadline_not_shortened(self):
        waiter = AsyncDeadlineWait(); deadline=time.perf_counter()+.005
        try:
            await waiter.until(deadline)
            self.assertGreaterEqual(time.perf_counter(),deadline)
        finally: waiter.close()


if __name__=='__main__': unittest.main()
