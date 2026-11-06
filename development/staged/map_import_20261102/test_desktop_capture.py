import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import unittest
from unittest.mock import AsyncMock,Mock,patch
from types import SimpleNamespace
from desktop_capture import Capture,crop_client
import numpy as np

class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_title_focus_race_retried_without_reading_wrong_window(self):
        capture=Capture();capture.open=Mock(side_effect=[False,False,True])
        page=SimpleNamespace(bring_to_front=AsyncMock(),evaluate=AsyncMock())
        try:
            with patch('desktop_capture.asyncio.sleep',AsyncMock()):await capture.ready(page)
            self.assertEqual(capture.open.call_count,3)
            page.bring_to_front.assert_awaited_once()
        finally:await capture.close()

    async def test_foreground_wait_bounded(self):
        capture=Capture();capture.open=Mock(return_value=False)
        page=SimpleNamespace(bring_to_front=AsyncMock(),evaluate=AsyncMock())
        try:
            with patch('desktop_capture.time.monotonic',side_effect=[0,61]):
                with self.assertRaises(RuntimeError):await capture.ready(page)
            capture.open.assert_not_called()
        finally:await capture.close()

    def test_crop_excludes_toolbar_and_rejects_hidden_canvas(self):
        image=np.zeros((100,200,4),np.uint8);image[20:]=7
        crop=crop_client(image,dict(x=0,y=0,width=20,height=10),dict(width=200,height=80))
        self.assertTrue((crop==7).all())
        with self.assertRaises(RuntimeError):
            crop_client(image,dict(x=0,y=70,width=20,height=20),dict(width=200,height=80))

if __name__=='__main__':unittest.main()
