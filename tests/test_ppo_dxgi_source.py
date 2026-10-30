from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from tools.ppo_dxgi_source import TimestampedDxgiSource


class Clock:
    value = 10.
    def now(self): return self.value
    def sleep(self, value): self.value += value


class DxgiFreshnessTests(unittest.TestCase):
    def test_new_readback_timestamp_cannot_hide_old_prepared_frame(self):
        from tools.ppo_prepared_guard import PreparedGameplayReading
        image = object()
        reading = PreparedGameplayReading(id(image), 10., 'GAMEPLAY', {}, 0., presented_at=9.79, source_acquired_at=9.8)
        self.assertFalse(reading.valid_for(image, 10., 10.001))
        current = PreparedGameplayReading(id(image), 10., 'GAMEPLAY', {}, 0., presented_at=9.98, source_acquired_at=9.999)
        self.assertTrue(current.valid_for(image, 10., 10.001))

    def source(self):
        clock = Clock()
        camera = Mock()
        camera._duplicator = SimpleNamespace(latest_frame_time=10., accumulated_frames=1)
        camera.grab.return_value = np.zeros((6, 8, 4), np.uint8)
        return TimestampedDxgiSource(camera, 8, 6, clock=clock.now, sleep=clock.sleep), clock, camera

    def test_cached_frame_never_becomes_fresh_gameplay_by_advancing_readback_clock(self):
        source, clock, camera = self.source()
        first = source.read()
        self.assertTrue(first.fresh)
        camera.grab.return_value = None
        clock.value += .2
        self.assertIsNone(source.read(timeout=.01))
        ui = source.read(allow_static_ui=True)
        self.assertFalse(ui.fresh)
        self.assertEqual(ui.present_at, first.present_at)
        self.assertEqual(ui.serial, first.serial)
        self.assertEqual(ui.received_at, first.received_at)

    def test_pointer_only_update_does_not_count_as_new_game_content(self):
        source, clock, camera = self.source()
        initial = source.read()
        camera._duplicator.accumulated_frames = 0
        camera._duplicator.latest_frame_time = 10.001
        self.assertIsNone(source.read(timeout=.01))
        cached = source.read(allow_static_ui=True)
        self.assertEqual(cached.serial, initial.serial)
        self.assertEqual(cached.present_at, initial.present_at)
        self.assertEqual(cached.received_at, initial.received_at)

    def test_duplicate_timestamp_does_not_count_as_new_frame(self):
        source, clock, camera = self.source()
        source.read()
        self.assertIsNone(source.read(timeout=.005))
        self.assertEqual(source.serial, 1)
        camera._duplicator.latest_frame_time = clock.value
        self.assertEqual(source.read().serial, 2)

    def test_future_regressed_and_invalid_shape_fail_closed(self):
        for case in ('future','regression','shape'):
            source, clock, camera = self.source()
            source.read()
            if case == 'future': camera._duplicator.latest_frame_time = 11.
            if case == 'regression': camera._duplicator.latest_frame_time = 9.
            if case == 'shape': camera.grab.return_value = np.zeros((5, 8, 4), np.uint8)
            with self.subTest(case=case), self.assertRaises(RuntimeError): source.read()

    def test_old_initial_frame_is_not_accepted_for_gameplay(self):
        source, clock, camera = self.source()
        camera._duplicator.latest_frame_time = 8.
        self.assertIsNone(source.read(timeout=.005))
        self.assertFalse(source.read(allow_static_ui=True).fresh)


if __name__ == '__main__':
    unittest.main()
