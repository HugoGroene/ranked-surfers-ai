import base64
from types import SimpleNamespace
import time
import unittest
from unittest.mock import AsyncMock
from tools.ppo_cdp_stream import LatestFrameStream,presentation_time
from rs_v612_gameplay_guard import GuardUncertainError


def frame(identity,data=b'new',age=0):
    return dict(sessionId=identity,data=base64.b64encode(data).decode(),metadata=dict(
        deviceWidth=1920,deviceHeight=945,pageScaleFactor=1,offsetTop=0,
        scrollOffsetX=0,scrollOffsetY=0,timestamp=time.time()-age,
        monotonicTimestamp=time.perf_counter()-age))


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_frame_is_not_returned_and_each_read_requires_new_swap(self):
        session=SimpleNamespace(send=AsyncMock())
        fallback=SimpleNamespace(capture=AsyncMock(return_value=b'fallback'))
        source=LatestFrameStream(session,fallback,1920,945)
        source.on_frame(frame(1,b'old',.05))
        def acknowledge(*args,**kwargs):
            source.on_frame(frame(source.serial+1))
        session.send.side_effect=acknowledge
        self.assertEqual(await source.capture(),b'new')
        first=source.used
        self.assertEqual(await source.capture(),b'new')
        self.assertGreater(source.used,first)
        self.assertEqual(source.stream_reads,2)
        fallback.capture.assert_not_awaited()

    async def test_static_surface_uses_independent_capture_not_cached_pixels(self):
        source=LatestFrameStream(SimpleNamespace(send=AsyncMock()),
            SimpleNamespace(capture=AsyncMock(return_value=b'independent')),1920,945)
        source.on_frame(frame(1,b'stale',.05))
        self.assertEqual(await source.capture(),b'independent')
        self.assertEqual(source.fallback_reads,1)
        self.assertNotIn('browser_frame_at',source.last_timing)

    async def test_wrong_geometry_stops_instead_of_falling_back(self):
        fallback=SimpleNamespace(capture=AsyncMock())
        source=LatestFrameStream(SimpleNamespace(send=AsyncMock()),fallback,1920,945)
        event=frame(1)
        event['metadata']['deviceWidth']=960
        source.on_frame(event)
        with self.assertRaises(GuardUncertainError):
            await source.capture()
        fallback.capture.assert_not_awaited()

    async def test_latest_slot_overwrites_without_image_fifo(self):
        source=LatestFrameStream(SimpleNamespace(send=AsyncMock()),None,1920,945)
        source.on_frame(frame(1,b'older'))
        source.on_frame(frame(2,b'newer'))
        self.assertEqual(source.latest[2],base64.b64encode(b'newer').decode())

    def test_future_browser_clock_is_rejected(self):
        event=frame(1)
        event['metadata']['timestamp']=time.time()+10
        with self.assertRaises(GuardUncertainError):
            presentation_time(event['metadata'],time.perf_counter(),time.time(),1920,945)


if __name__ == '__main__':
    unittest.main()
