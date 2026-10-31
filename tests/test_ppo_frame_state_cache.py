from types import SimpleNamespace
from unittest.mock import Mock
import unittest
from tools.ppo_frame_state_cache import install_frame_state_cache

class CacheTests(unittest.TestCase):
    def env(self):
        raw=SimpleNamespace(_v2_last_bgra=object(),last_capture_end=1.,screen_classifier=object(),
            state_confidence=.9,_learning_capture_serial=1,
            _read_screen_state=Mock(return_value=('GAMEPLAY',{'GAMEPLAY':1.})))
        original=raw._read_screen_state;install_frame_state_cache(raw)
        return raw,original

    def test_same_capture_reused_and_returned_probabilities_are_owned(self):
        raw,original=self.env();a=raw._read_screen_state();a[1]['GAMEPLAY']=0.
        self.assertEqual(raw._read_screen_state(),('GAMEPLAY',{'GAMEPLAY':1.}))
        self.assertEqual(original.call_count,1)

    def test_every_capture_model_or_confidence_change_invalidates(self):
        for field,new in [('_v2_last_bgra',object()),('last_capture_end',2.),
                ('screen_classifier',object()),('state_confidence',.95),('_learning_capture_serial',2)]:
            with self.subTest(field=field):
                raw,original=self.env();raw._read_screen_state();setattr(raw,field,new)
                raw._read_screen_state();self.assertEqual(original.call_count,2)

    def test_missing_capture_does_not_cache(self):
        raw,original=self.env();raw.last_capture_end=None
        raw._read_screen_state();raw._read_screen_state()
        self.assertEqual(original.call_count,2)
