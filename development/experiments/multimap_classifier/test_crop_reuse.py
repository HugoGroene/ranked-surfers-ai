import unittest
from types import SimpleNamespace
import numpy as np
from crop_reuse import matching_crop,install


class CropTests(unittest.TestCase):
    def raw(self):
        prepared=(np.zeros((3,384,384),np.uint8),np.zeros((945,1920,3),np.uint8))
        shot=np.zeros((1080,1920,4),np.uint8)
        return SimpleNamespace(pump=SimpleNamespace(delivered_prepared=(10.,5,prepared)),
            _v2_last_bgra=shot,grabber=SimpleNamespace(image=shot),
            _headless_capture_timing=dict(request_started_at=10.,electron_serial=5),
            _framing_profile='canvas_v1',_canvas_profile_preview_saved=True,
            _road_preview_saved=True,target_wh=(384,384)),prepared

    def test_exact_delivery_reuses_original_arrays(self):
        raw,prepared=self.raw()
        self.assertIs(matching_crop(raw),prepared)

    def test_mismatch_or_geometry_never_reused(self):
        for key,value in [('request_started_at',11.),('electron_serial',6)]:
            raw,_=self.raw();raw._headless_capture_timing[key]=value
            self.assertIsNone(matching_crop(raw))
        for key,value in [('_background_geometry_invalid','changed'),('_road_preview_saved',False),
                          ('_canvas_profile_preview_saved',False),('_framing_profile','legacy')]:
            raw,_=self.raw();setattr(raw,key,value)
            self.assertIsNone(matching_crop(raw))

    def test_missing_or_wrong_shape_falls_back(self):
        raw,_=self.raw();raw.pump.delivered_prepared=(10.,5,None)
        self.assertIsNone(matching_crop(raw))
        raw,_=self.raw();raw.grabber.image=raw.grabber.image.copy()
        self.assertIsNone(matching_crop(raw))
        raw,_=self.raw();raw.target_wh=(192,192)
        self.assertIsNone(matching_crop(raw))

    def test_installed_wrapper_preserves_geometry_rejection_and_fallback(self):
        from rs_v612_gameplay_guard import GuardUncertainError
        raw,prepared=self.raw()
        raw.region=dict(left=0,top=0,width=1920,height=1080)
        raw._background_canvas=dict(left=0,top=87,width=1920,height=945)
        fallback=object()
        raw._native_frames=lambda:fallback
        async def close():pass
        raw._close_browser=close
        trainer=SimpleNamespace(create_live_env=lambda:(None,raw))
        install(trainer)
        factory=trainer.create_live_env
        install(trainer)
        self.assertIs(trainer.create_live_env,factory)
        trainer.create_live_env()
        actual=raw._native_frames()
        self.assertIs(actual[0],prepared[0])
        self.assertIs(raw.last_source_native,prepared[1])
        raw._background_canvas['left']=1
        with self.assertRaises(GuardUncertainError):raw._native_frames()
        raw._headless_capture_timing['electron_serial']=6
        self.assertIs(raw._native_frames(),fallback)


if __name__=='__main__':unittest.main()
