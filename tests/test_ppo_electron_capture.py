import time
import unittest
import asyncio
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np
from tools.ppo_electron_capture import decode_frame,ElectronCapture,StaleElectronFrame
from tools.ppo_headless_env import compose_guard_frame
from rs_v612_gameplay_guard import GuardUncertainError


def metadata(**changes):
    return dict(dict(width=2,height=1,channels=4,format='BGRA',byte_length=8,
        serial=2,paint_wall=time.time()),**changes)


class ElectronCaptureTests(unittest.TestCase):
    def test_raw_channels_orientation_and_guard_padding(self):
        pixels=bytes([0,0,255,255,255,0,0,255])
        frame,timing=decode_frame(metadata(),pixels,2,1,1,time.perf_counter(),time.time())
        self.assertEqual(frame[0,0].tolist(),[0,0,255,255])
        padded=compose_guard_frame(frame,1,1,width=4,height=3)
        np.testing.assert_array_equal(padded[1,1:3],frame[0])
        self.assertEqual(padded[0,0].tolist(),[0,0,0,255])
        self.assertNotIn('browser_frame_at',timing) # Paint callback is not GPU presentation.

    def test_geometry_channels_and_payload_errors_stop(self):
        for changes in [dict(width=1),dict(height=2),dict(channels=3),
                dict(format='RGBA'),dict(byte_length=4)]:
            with self.subTest(changes=changes),self.assertRaises(GuardUncertainError):
                decode_frame(metadata(**changes),bytes(8),2,1,1,time.perf_counter(),time.time())
        with self.assertRaises(GuardUncertainError):
            decode_frame(metadata(),bytes(4),2,1,1,time.perf_counter(),time.time())

    def test_duplicate_regressed_and_invalid_serials_stop(self):
        for serial in [0,1,True,None]:
            with self.subTest(serial=serial),self.assertRaises(GuardUncertainError):
                decode_frame(metadata(serial=serial),bytes(8),2,1,1,time.perf_counter(),time.time())

    def test_old_future_or_missing_clock_stops(self):
        for stamp in [time.time()-.1,time.time()+1,float('nan'),None]:
            with self.subTest(stamp=stamp),self.assertRaises(GuardUncertainError):
                decode_frame(metadata(paint_wall=stamp),bytes(8),2,1,1,time.perf_counter(),time.time())


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def source(self,meta,pixels=b''):
        source=ElectronCapture()
        source.width,source.height=2,1
        source.serial=1;source.reads=0;source.token='test'
        source.stale_frame_retries=0
        source.reader=asyncio.StreamReader()
        header=json.dumps(meta).encode()
        source.reader.feed_data(struct.pack('<I',len(header))+header+pixels)
        source.reader.feed_eof()
        source.writer=SimpleNamespace(write=lambda _:None,drain=AsyncMock())
        return source

    async def test_raw_binary_roundtrip_has_no_image_encoding(self):
        source=self.source(metadata(),bytes(range(8)))
        image=await source.capture()
        self.assertEqual(image.shape,(1,2,4))
        self.assertEqual(image.ravel().tolist(),list(range(8)))
        self.assertEqual(source.reads,1)

    async def test_opt_in_profile_has_host_and_node_timings_without_changing_pixels(self):
        source=self.source(metadata(profile=dict(node_request_to_paint_ms=0.,
            bitmap_extract_ms=0.,node_request_to_send_ms=0.)),bytes(range(8)))
        source.profile_enabled=True
        image=await source.capture()
        self.assertEqual(image.ravel().tolist(),list(range(8)))
        values=source.last_timing['capture_profile']
        self.assertGreaterEqual(values['capture_total_wall_ms'],values['capture_attempt_wall_ms'])
        self.assertGreaterEqual(values['transport_and_scheduling_residual_ms'],0.)

    async def test_truncated_payload_and_server_error_fail_closed(self):
        for meta,pixels in [(metadata(),b'bad'),({'error':'No fresh offscreen paint'},b'')]:
            with self.subTest(meta=meta),self.assertRaises(GuardUncertainError):
                await self.source(meta,pixels).capture()

    async def test_unexpected_payload_size_rejected_before_read(self):
        source=self.source(metadata(byte_length=10**9))
        with self.assertRaises(GuardUncertainError):await source.capture()

    async def test_stale_frame_is_discarded_before_a_new_request(self):
        source=self.source(metadata(paint_wall=time.time()-.1),bytes(8))
        # First two replies are stale; only the last independently requested
        # frame is eligible. Distinct serials are checked by the real decoder.
        source._capture_once=AsyncMock(side_effect=[StaleElectronFrame('old'),
            StaleElectronFrame('old'),np.zeros((1,2,4),np.uint8)])
        source.last_timing={}
        image=await source.capture()
        self.assertEqual(image.shape,(1,2,4))
        self.assertEqual(source.stale_frame_retries,2)
        self.assertEqual(source._capture_once.await_count,3)

    async def test_persistent_staleness_stops_after_three_requests(self):
        source=self.source(metadata())
        source._capture_once=AsyncMock(side_effect=StaleElectronFrame('old'))
        with self.assertRaises(StaleElectronFrame):await source.capture()
        self.assertEqual(source._capture_once.await_count,3)

    async def test_clock_or_geometry_errors_never_retry(self):
        source=self.source(metadata())
        source._capture_once=AsyncMock(side_effect=GuardUncertainError('invalid clock'))
        with self.assertRaises(GuardUncertainError):await source.capture()
        self.assertEqual(source._capture_once.await_count,1)


if __name__=='__main__':unittest.main()
