import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import numpy as np
from rs_v6_ppo_features import encode_hazards
from rs_v61_features import screen_columns,encode_aux,ActionContext
from tools.ppo_detector_features import DetectorFeatureCache
from tools.ppo_worker_preview import preview_control,PreviewPublisher
from tools.ppo_shared_inference import validate_reply,RemotePolicy
from rs_v612_gameplay_guard import GuardUncertainError


class ScalingControlTests(unittest.TestCase):
    def snapshot(self):
        return dict(captured_at=10.,episode=2,detections=[dict(class_='coin')])

    def test_geometry_cache_matches_uncached_features_and_ages_out_without_reusing_votes(self):
        snapshot=dict(captured_at=10.,episode=2,detections=[{'class':'coin','box':[.1,.2,.3,.5],
            'conf':.8,'polygon':[[.1,.2],[.3,.2],[.3,.5]]}])
        cache=DetectorFeatureCache()
        for now,episode in [(10.01,2),(10.20,2),(10.301,2),(10.4,3),(9.99,2)]:
            hazards,age,fresh,columns=cache.read(snapshot,now=now,episode_tag=episode,max_stale_ms=300)
            expected=encode_hazards(snapshot,now=now,episode_tag=episode,max_stale_ms=300)
            np.testing.assert_array_equal(hazards,expected[0]);self.assertEqual((age,fresh),expected[1:])
            np.testing.assert_array_equal(columns,screen_columns(snapshot,now=now,episode_tag=episode,max_stale_ms=300))
            original=encode_aux(hazards,[1,3],.2,[.04],age,snapshot=snapshot,episode_tag=episode,now=now,action_context=ActionContext())
            cached=encode_aux(hazards,[1,3],.2,[.04],age,snapshot=snapshot,episode_tag=episode,now=now,action_context=ActionContext(),precomputed_columns=columns)
            np.testing.assert_array_equal(original,cached)
        self.assertGreater(cache.hits,0)
        before=cache.misses
        cache.read(dict(snapshot),now=10.1,episode_tag=2,max_stale_ms=300)
        self.assertEqual(cache.misses,before+1)

    def test_detector_off_does_not_load_or_warm_yolo_but_constructs_same_guarded_environment(self):
        import train_rs_v612_ppo as trainer
        with tempfile.TemporaryDirectory() as directory:
            args=trainer.parse_args(['--mode','check','--detector-mode','off'])
            args.mode='eval'
            raw=SimpleNamespace(observation_space=None,close=Mock())
            env_class=Mock(return_value=raw)
            with patch.object(trainer,'build_env_class',return_value=env_class),\
                 patch('ultralytics.YOLO',side_effect=AssertionError('YOLO must not load')),\
                 patch('rs_v5_shadow_segmentation.detect_v5',side_effect=AssertionError('YOLO must not warm')),\
                 patch('stable_baselines3.common.monitor.Monitor',side_effect=lambda value,**kw:value):
                monitored,actual=trainer.create_live_env(args,Path(directory))
            self.assertIs(actual,raw)
            self.assertIsNone(env_class.call_args.args[3])
            self.assertEqual(raw.detector_mode,'off')
            self.assertIs(monitored,raw)

    def test_preview_fps_validation_and_disabled_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);worker=folder/'worker_0';worker.mkdir()
            control=folder/'preview_control.json'
            control.write_text(json.dumps(dict(enabled=False,fps=30)))
            publisher=PreviewPublisher(worker,1)
            try:
                with patch('tools.ppo_worker_preview.write_preview') as encoder:
                    publisher.offer(np.zeros((12,8,8),np.uint8),{'phase':'collecting'})
                    publisher.close()
                    encoder.assert_not_called()
                self.assertEqual(preview_control(folder),dict(enabled=False,fps=30.))
                control.write_text(json.dumps(dict(enabled=True,fps=True)))
                self.assertEqual(preview_control(folder)['fps'],1.)
            finally:publisher.close()

    def test_preview_has_a_single_latest_frame_slot_and_owns_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);worker=folder/'worker_0';worker.mkdir()
            (folder/'preview_control.json').write_text(json.dumps(dict(enabled=True,fps=30)))
            publisher=PreviewPublisher(worker,30)
            # Block the encoder so producers must replace, rather than queue.
            import threading
            entered=threading.Event();release=threading.Event();frames=[]
            def encode(path,frame,*args,**kwargs):
                frames.append(frame.copy());entered.set();release.wait(2);return True
            with patch('tools.ppo_worker_preview.write_preview',side_effect=encode):
                source=np.ones((12,8,8),np.uint8)
                publisher.offer(source,{},now=1.,force=True)
                self.assertTrue(entered.wait(1))
                source[:]=2;publisher.offer(source,{},now=2.,force=True)
                source[:]=3;publisher.offer(source,{},now=3.,force=True)
                source[:]=99
                release.set();publisher.close()
            self.assertEqual(len(frames),2)
            self.assertTrue(np.all(frames[0]==1));self.assertTrue(np.all(frames[1]==3))
            self.assertEqual(publisher.dropped,1)

    def test_inference_rejects_late_wrong_request_and_wrong_policy_versions(self):
        good=dict(serial=3,version=2,policy_hash='d')
        validate_reply(good,3,2,'d',time.perf_counter()+1)
        for reply,deadline in [(good,0),({**good,'serial':4},time.perf_counter()+1),
                               ({**good,'version':3},time.perf_counter()+1),
                               ({**good,'policy_hash':'other'},time.perf_counter()+1),
                               ({**good,'error':'failed'},time.perf_counter()+1)]:
            with self.assertRaises(GuardUncertainError):validate_reply(reply,3,2,'d',deadline)

    def test_remote_policy_never_trains_or_accepts_another_version(self):
        policy=RemotePolicy(SimpleNamespace(version=2,digest='a'),{})
        policy.bind_version('a')
        with self.assertRaises(GuardUncertainError):policy.bind_version('b')
        with self.assertRaises(RuntimeError):policy.set_training_mode(True)

    def test_actual_preview_server_changes_fps_and_rejects_invalid_or_foreign_controls(self):
        from urllib.request import Request,urlopen
        from urllib.error import HTTPError
        from tools.ppo_worker_preview import start_preview
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            server=start_preview(folder,1,0,5)
            base=f'http://127.0.0.1:{server.server_port}'
            def post(value,origin=base):
                request=Request(base+'/preview-control',data=json.dumps(value).encode(),
                    headers={'Content-Type':'application/json','Origin':origin},method='POST')
                with urlopen(request,timeout=3) as response:return response.status
            try:
                self.assertNotIn('__PREVIEW_FPS__',(folder/'index.html').read_text(encoding='utf8'))
                self.assertEqual(post(dict(enabled=False,fps=30)),204)
                self.assertEqual(preview_control(folder),dict(enabled=False,fps=30.))
                self.assertEqual(post(dict(enabled=True)),204)
                self.assertEqual(preview_control(folder)['fps'],30.)
                for value in [dict(enabled=True,fps=0),dict(enabled=True,fps=31),dict(enabled=True,fps=True),[]]:
                    with self.subTest(control=value),self.assertRaises(HTTPError) as failure:post(value)
                    self.assertEqual(failure.exception.code,400)
                with self.assertRaises(HTTPError) as failure:post(dict(enabled=False),origin='https://example.com')
                self.assertEqual(failure.exception.code,403)
                self.assertTrue(preview_control(folder)['enabled'])
            finally:server.shutdown();server.server_close()
