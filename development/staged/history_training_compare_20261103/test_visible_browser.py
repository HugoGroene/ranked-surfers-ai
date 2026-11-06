"""Offline visible-capture checks; no browser or game is launched."""
import base64
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'development/experiments/capture_pipeline')]
import cv2
import numpy as np
from visible_browser import VisibleBrowserCapture, StaleElectronFrame, GuardUncertainError, install_practice_start


class VisibleTests(unittest.IsolatedAsyncioTestCase):
    async def test_visible_launch_and_viewport(self):
        page = SimpleNamespace(bring_to_front=AsyncMock())
        session = SimpleNamespace(send=AsyncMock(side_effect=[{'windowId':7},{},{},
            {'bounds':{'windowState':'maximized'}}]))
        context = SimpleNamespace(new_page=AsyncMock(return_value=page),
            new_cdp_session=AsyncMock(return_value=session))
        browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
        playwright = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(return_value=browser)))
        with tempfile.TemporaryDirectory() as folder, patch('adapter.install_scale', AsyncMock(return_value=[])), patch('visible_browser.install_practice_start'):
            source = await VisibleBrowserCapture.launch(playwright, Path(folder)/'capture', 1920, 945)
            self.assertFalse(playwright.chromium.launch.call_args.kwargs['headless'])
            self.assertEqual(browser.new_context.call_args.kwargs['viewport'], {'width':1920,'height':945})
            self.assertEqual(session.send.call_args_list[2].args,
                ('Browser.setWindowBounds', {'windowId':7,'bounds':{'windowState':'maximized'}}))
            await source.close()
            await source.close()
            browser.close.assert_awaited_once()

    async def test_practice_uses_exact_button_without_screen_coordinates(self):
        import rs_practice_controller as ctrl
        button = SimpleNamespace(is_visible=AsyncMock(return_value=True),
            is_enabled=AsyncMock(return_value=True), evaluate=AsyncMock())
        handle=SimpleNamespace(evaluate=button.evaluate,dispose=AsyncMock())
        button.element_handle=AsyncMock(return_value=handle)
        from unittest.mock import Mock
        buttons = SimpleNamespace(count=AsyncMock(return_value=1), nth=Mock(return_value=button))
        frame = SimpleNamespace(get_by_role=Mock(return_value=buttons))
        page = SimpleNamespace(url='http://localhost:8000/', frames=[frame])
        with patch.object(ctrl, 'click_practice'):
            install_practice_start(Path('unused'))
            await ctrl.click_practice(page)
            frame.get_by_role.assert_called_once_with('button', name='Practice', exact=True)
            button.evaluate.assert_awaited_once()
            page.url='https://rankedsurfers.com/'
            with self.assertRaises(GuardUncertainError): await ctrl.click_practice(page)

    def source(self, image):
        source = VisibleBrowserCapture()
        source.width, source.height = 3, 2
        source.serial = source.reads = source.stale_frame_retries = 0
        source.last_timing = {}; source.scale_errors = []
        encoded = cv2.imencode('.png', image)[1].tobytes()
        source.session = SimpleNamespace(send=AsyncMock(return_value={'data':base64.b64encode(encoded).decode()}))
        return source

    async def test_lossless_pixels_and_timing(self):
        pixels = np.arange(18, dtype=np.uint8).reshape(2,3,3)
        source = self.source(pixels)
        with patch('visible_browser.time.perf_counter', side_effect=[10.,10.02]):
            image = await source.capture()
        np.testing.assert_array_equal(image[:,:,:3], pixels)
        self.assertTrue(np.all(image[:,:,3] == 255))
        self.assertEqual(source.last_timing['electron_serial'], 1)
        self.assertNotIn('paint_callback_at', source.last_timing)

    async def test_slow_capture_rejected(self):
        source = self.source(np.zeros((2,3,3), np.uint8))
        with patch('visible_browser.time.perf_counter', side_effect=[10.,10.08]):
            with self.assertRaises(StaleElectronFrame): await source.capture()
        self.assertEqual(source.reads, 0)
        self.assertEqual(source.last_timing, {})

    async def test_wrong_geometry_rejected(self):
        source = self.source(np.zeros((4,3,3), np.uint8))
        with self.assertRaises(GuardUncertainError): await source.capture()


if __name__ == '__main__': unittest.main()
