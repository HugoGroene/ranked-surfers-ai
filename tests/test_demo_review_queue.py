import unittest

import numpy as np

from tools.build_demo_review import frame_window, session_runs, suggested_window


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0, 10.05, .05)
        self.valid = np.ones(len(self.times), dtype=bool)
        self.valid[-1] = False
        self.actions = np.zeros(len(self.times), dtype=int)
        self.actions[60] = 1
        self.actions[100] = 2

    def test_suggestion_keeps_short_run_with_useful_labels(self):
        t, a, v = self.times[:161], self.actions[:161], self.valid[:161]
        w = suggested_window(t, v, a)
        self.assertIsNotNone(w)
        self.assertEqual(w['action_counts']['LEFT'], 1)
        self.assertEqual(w['action_counts']['RIGHT'], 1)

    def test_bounds_snap_without_including_label_after_cut(self):
        self.actions[101] = 3
        w = frame_window(self.times, self.valid, self.actions, 2.151, 5.023)
        self.assertEqual(w['start'], self.times[44])
        self.assertEqual(w['end'], self.times[100])
        self.assertEqual(w['action_counts']['RIGHT'], 0)
        self.assertEqual(w['action_counts']['JUMP'], 0)

    def test_stops_suggestion_before_interior_gap(self):
        self.times[120:] += .5
        w = suggested_window(self.times, self.valid, self.actions)
        self.assertEqual(w['end'], self.times[119])

    def test_four_frame_history_crossing_gap_is_rejected(self):
        self.times[60:] += .5
        self.actions[61] = 2
        self.assertIsNone(frame_window(self.times, self.valid, self.actions,
                                      self.times[61], self.times[70]))
        self.assertIsNotNone(frame_window(self.times, self.valid, self.actions,
                                         self.times[63], self.times[110]))

    def test_none_only_and_too_short_are_not_suggested(self):
        self.assertIsNone(suggested_window(self.times[:60], self.valid[:60], self.actions[:60]))
        self.assertIsNone(frame_window(self.times, self.valid, self.actions, 6, 8))

    def test_nonmonotonic_timestamps_rejected(self):
        self.times[50] = self.times[49]
        self.assertIsNone(frame_window(self.times, self.valid, self.actions, 2.15, 7))

    def test_session_scope_comes_from_saved_log_entries(self):
        log = '[RUN 2] started\n[AUTOSAVE] Saved candidate run 1: ai_data/runs/demo_20261028_160800_895699 | valid labels {}'
        self.assertEqual(session_runs(log), [(1, 'demo_20261028_160800_895699')])
        with self.assertRaises(ValueError):
            session_runs(log + '\n' + log)
