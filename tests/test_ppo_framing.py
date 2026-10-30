import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.ppo_framing import canvas_observation
from tools.ppo_background_display import canvas_from_display


class FramingTests(unittest.TestCase):
    def test_desktop_and_browser_paths_are_pixel_identical(self):
        rng = np.random.default_rng(42)
        canvas = rng.integers(0,256,(945,1920,4),dtype=np.uint8)
        desktop = np.zeros((1080,1920,4),dtype=np.uint8)
        desktop[87:1032] = canvas
        visible = canvas_from_display(desktop,dict(left=0,top=0,width=1920,height=1080),
                                      dict(left=0,top=87,width=1920,height=945))
        a,native = canvas_observation(canvas)
        b,other = canvas_observation(visible)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(native,other)
        self.assertEqual(a.shape,(3,288,512))
        np.testing.assert_array_equal(native[0,0],canvas[0,0,[2,1,0]])
        np.testing.assert_array_equal(native[-1,-1],canvas[-1,-1,[2,1,0]])

    def test_reject_changed_camera_aspect(self):
        with self.assertRaises(ValueError):
            canvas_observation(np.zeros((1080,1920,4),np.uint8))

    def test_clean_policy_excludes_desktop_without_changing_guard_source(self):
        from tools.ppo_background_display import BackgroundDisplayMixin
        env = BackgroundDisplayMixin()
        env._framing_profile = 'canvas_v1'
        env.region = dict(left=0,top=0,width=1920,height=1080)
        env._background_canvas = dict(left=0,top=87,width=1920,height=945)
        raw = np.full((1080,1920,4),255,np.uint8)
        raw[87:1032,:,:3] = (1,2,3)
        env._v2_last_bgra = raw
        env.target_wh = (512,288)
        env.resize_method = 'linear'
        with tempfile.TemporaryDirectory() as directory:
            env.output_dir = Path(directory)
            policy,native = env._native_frames()
            self.assertIs(env._v2_last_bgra,raw)
            self.assertTrue(np.all(native == (3,2,1)))
            self.assertTrue((env.output_dir/'canvas_v1_policy.png').is_file())
            self.assertEqual(json.loads((env.output_dir/'framing_profile.json').read_text())['profile'],'canvas_v1')

    def test_bc_profile_selection_does_not_mix_existing_data(self):
        from rs_v612_demo_bc import approved_runs
        from rs_v61_features import FEATURE_VERSION,AUX_DIM
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,profile in [('old',None),('new','canvas_v1')]:
                folder = root/'runs'/('demo_'+name)
                folder.mkdir(parents=True)
                m = dict(status='approved',schema=1,feature_version=FEATURE_VERSION,aux_dim=AUX_DIM)
                if profile:
                    m['capture'] = dict(framing_profile=profile)
                (folder/'manifest.json').write_text(json.dumps(m))
            self.assertEqual([p.name for p,m in approved_runs(root)],['demo_old'])
            self.assertEqual([p.name for p,m in approved_runs(root,'canvas_v1')],['demo_new'])


class HeadlessTimeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_headless_reload_does_not_enter_native_fullscreen(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from tools.ppo_episode_reset import reload_after_time_limit
        env = SimpleNamespace(args=SimpleNamespace(url='http://localhost:8000'),
                              _headless_reload_after_time_limit=AsyncMock(),
                              _enter_game_fullscreen=AsyncMock())
        await reload_after_time_limit(env)
        env._headless_reload_after_time_limit.assert_awaited_once()
        env._enter_game_fullscreen.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
