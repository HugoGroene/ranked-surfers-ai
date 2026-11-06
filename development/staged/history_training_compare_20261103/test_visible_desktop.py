"""Offline capture/geometry/timing checks; no game or desktop screenshots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from visible_desktop import DesktopReader, summarize_render, collect_render
from visible_native import VisibleNativeCapture
from tools.ppo_electron_capture import ElectronCapture, StaleElectronFrame
from rs_v612_gameplay_guard import GuardUncertainError


class DesktopTests(unittest.TestCase):
    def reader(self):
        r=DesktopReader.__new__(DesktopReader)
        r.width,r.height=3,2
        r.initial_region=dict(left=10,top=20,width=3,height=2)
        r.region=Mock(return_value=r.initial_region)
        r.grabber=Mock()
        r.grabber.grab.return_value=SimpleNamespace(width=3,height=2,raw=bytearray(24))
        return r

    def test_focus_loss_during_read_rejects_pixels(self):
        r=self.reader()
        r.region.side_effect=[r.initial_region,GuardUncertainError('focus lost')]
        with self.assertRaises(GuardUncertainError):r.read()

    def test_window_move_rejected_before_read(self):
        r=self.reader();r.region.return_value=dict(r.initial_region,left=11)
        with self.assertRaises(GuardUncertainError):r.read()
        r.grabber.grab.assert_not_called()

    def test_mss_bytes_are_owned_not_reused(self):
        r=self.reader();result=r.read()
        r.grabber.grab.return_value.raw[0]=255
        self.assertEqual(result[0],0)

    def test_real_region_checks_foreground_and_size(self):
        r=DesktopReader.__new__(DesktopReader)
        r.hwnd=123;r.width=1920;r.height=945;r.user=Mock()
        r.user.IsWindowVisible.return_value=True;r.user.IsIconic.return_value=False
        r.user.GetForegroundWindow.return_value=456
        with self.assertRaises(GuardUncertainError):r.region()
        r.user.GetClientRect.assert_not_called()
        r.user.GetForegroundWindow.return_value=123
        def rect(hwnd, ptr):
            ptr._obj.right=1920;ptr._obj.bottom=944
            return True
        r.user.GetClientRect.side_effect=rect
        with self.assertRaises(GuardUncertainError):r.region()

    def test_render_summary_does_not_call_it_game_fps(self):
        bins=[0]*1001;bins[7]=90;bins[30]=10
        s=summarize_render(dict(count=100,total_ms=930,max_ms=30,
                               over25=10,over50=0,histogram_ms=bins))
        self.assertEqual(s['p95_interval_ms'],30)
        self.assertEqual(s['p50_interval_ms'],7)
        self.assertIn('Not verified Unity',s['note'])
        self.assertEqual(summarize_render(None)['status'],'NO_SAMPLES')


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_uses_game_iframe_not_smooth_outer_page(self):
        def sample(ms):
            bins=[0]*1001;bins[ms]=100
            return dict(count=100,total_ms=ms*100,max_ms=ms,over25=0,over50=0,histogram_ms=bins)
        outer=SimpleNamespace(evaluate=AsyncMock(return_value=dict(
            timing=sample(7),url='http://localhost:8000/',game_canvas=False)))
        game=SimpleNamespace(evaluate=AsyncMock(return_value=dict(
            timing=sample(33),url='http://localhost:8000/game/arena.html',game_canvas=True)))
        result=await collect_render(SimpleNamespace(frames=[outer,game]))
        self.assertEqual(result['measured_context'],'unity_iframe')
        self.assertEqual(result['p95_interval_ms'],33)

    async def test_desktop_bypasses_capture_page_and_keeps_bgra(self):
        source=VisibleNativeCapture();source.backend='desktop'
        source.width=3;source.height=2;source.serial=source.reads=0
        source.desktop=SimpleNamespace(read=lambda:bytes(range(24)))
        with ThreadPoolExecutor(max_workers=1) as executor:
            source.executor=executor
            with patch.object(ElectronCapture,'_capture_once',side_effect=AssertionError('page capture')):
                image=await source._capture_once()
                self.assertEqual(image.shape,(2,3,4))
                np.testing.assert_array_equal(image[0,0],[0,1,2,3])
                self.assertEqual(source.last_timing['electron_serial'],1)
                await source._capture_once()
                self.assertEqual(source.last_timing['electron_serial'],2)

    async def test_slow_desktop_readback_still_rejected(self):
        source=VisibleNativeCapture();source.backend='desktop'
        source.width=3;source.height=2;source.serial=source.reads=0
        source.desktop=SimpleNamespace(read=lambda:bytes(24))
        with ThreadPoolExecutor(max_workers=1) as executor:
            source.executor=executor
            with patch('visible_native.time.perf_counter',side_effect=[10.,10.051]):
                with self.assertRaises(StaleElectronFrame):await source._capture_once()
        self.assertEqual(source.reads,0)


class TraceTests(unittest.TestCase):
    def test_trace_preserves_actions_and_separates_episodes(self):
        from visible_episode_trace import install
        import tempfile
        import json
        env=SimpleNamespace(reset=Mock(return_value=('obs',{})),
            step=Mock(return_value=('obs',1.,True,False,{'learning_dispatched_key':'ArrowUp'})),
            close=Mock())
        trainer=SimpleNamespace(create_live_env=Mock(return_value=(env,object())))
        install(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            monitored,_=trainer.create_live_env(None,Path(tmp))
            monitored.reset();self.assertEqual(monitored.step(3)[1],1.)
            monitored.reset();monitored.step(0);monitored.close()
            report=json.loads((Path(tmp)/'visible_episode_trace.json').read_text())
        self.assertEqual(report['episodes'][0]['requested_actions'],{'3':1})
        self.assertEqual(report['episodes'][1]['requested_actions'],{'0':1})
        self.assertEqual(report['episodes'][0]['dispatched_keys'],{'ArrowUp':1})


if __name__=='__main__':unittest.main()
