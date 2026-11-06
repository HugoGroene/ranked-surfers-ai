"""Offline visible-native transport checks; never launches a browser."""
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from visible_native import VisibleNativeCapture, validate_ready, GuardUncertainError
from tools.ppo_electron_capture import StaleElectronFrame


class NativeTests(unittest.IsolatedAsyncioTestCase):
    def test_requires_actual_visible_window_and_matching_shared_transport(self):
        good = dict(width=1920,height=945,visible=True,offscreen=False,
                    transport='shared_memory',shared_name='test')
        validate_ready(good,1920,945)
        for change in [dict(visible=False),dict(offscreen=True),dict(width=960),
                       dict(transport='socket'),dict(shared_name=None)]:
            with self.assertRaises(GuardUncertainError):
                validate_ready(dict(good,**change),1920,945)

    def source(self):
        source = VisibleNativeCapture()
        source.backend='page'
        source.scale_errors=[]; source.last_timing={}; source.stale_frame_retries=0
        return source

    async def test_stale_readback_discarded_before_fresh_frame(self):
        source=self.source(); pixels=np.zeros((2,3,4),np.uint8)
        source._capture_once=AsyncMock(side_effect=[StaleElectronFrame('slow'),pixels])
        self.assertIs(await source.capture(),pixels)
        self.assertEqual(source.stale_frame_retries,1)
        self.assertEqual(source._capture_once.await_count,2)
        self.assertEqual(source.last_timing['capture_kind'],'visible_native_bitmap_shared_memory')

    async def test_repeated_stale_readback_still_stops(self):
        source=self.source()
        source._capture_once=AsyncMock(side_effect=StaleElectronFrame('slow'))
        with self.assertRaises(StaleElectronFrame): await source.capture()
        self.assertEqual(source._capture_once.await_count,3)

    async def test_render_scale_failure_stops_before_capture(self):
        source=self.source();source.scale_errors=['wrong render scale']
        source._capture_once=AsyncMock()
        with self.assertRaises(GuardUncertainError): await source.capture()
        source._capture_once.assert_not_awaited()


if __name__=='__main__':unittest.main()
