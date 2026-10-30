import base64
import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import AsyncMock
from tools.ppo_cdp_capture import CdpCanvasCapture
from rs_v612_gameplay_guard import GuardUncertainError


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def test_guard_composition_is_pixel_identical_for_both_profiles(self):
        from tools.ppo_headless_env import compose_guard_frame
        rng=np.random.default_rng(4)
        for width,height,left in [(1920,945,0),(1904,937,8)]:
            image=rng.integers(0,256,(height,width,3),np.uint8)
            old=np.zeros((1080,1920,4),np.uint8)
            old[:,:,3]=255
            old[87:87+height,left:left+width,:3]=image
            np.testing.assert_array_equal(compose_guard_frame(image,left,87),old)
    async def test_clip_validated_once_and_preserves_full_canvas_quality(self):
        canvas=SimpleNamespace(bounding_box=AsyncMock(return_value=dict(x=0,y=0,width=1920,height=945)))
        session=SimpleNamespace(send=AsyncMock(return_value=dict(data=base64.b64encode(b'image').decode())))
        source=await CdpCanvasCapture.create(session,canvas,1920,945)
        self.assertEqual(await source.capture(),b'image')
        await source.capture()
        canvas.bounding_box.assert_awaited_once()
        command,options=session.send.await_args.args
        self.assertEqual(command,'Page.captureScreenshot')
        self.assertEqual(options['quality'],90)
        self.assertEqual(options['clip'],dict(x=0,y=0,width=1920,height=945,scale=1))
        self.assertFalse(options['captureBeyondViewport'])

    async def test_wrong_canvas_geometry_is_rejected_before_capture(self):
        canvas=SimpleNamespace(bounding_box=AsyncMock(return_value=dict(x=10,y=0,width=1910,height=945)))
        session=SimpleNamespace(send=AsyncMock())
        with self.assertRaises(GuardUncertainError):
            await CdpCanvasCapture.create(session,canvas,1920,945)
        session.send.assert_not_awaited()

    async def test_invalid_empty_and_missing_images_stop(self):
        for response in [dict(data='!invalid'),dict(data=''),{}]:
            with self.subTest(response=response):
                source=CdpCanvasCapture(SimpleNamespace(send=AsyncMock(return_value=response)),{})
                with self.assertRaises(GuardUncertainError):
                    await source.capture()


if __name__ == '__main__':
    unittest.main()
