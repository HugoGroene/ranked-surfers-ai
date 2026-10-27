"""Offline tests: recorder 1.3.1 read-only diagnostics and tools/analyze_demo_run.py.

python -m unittest test_rs_v612_demo_diagnostics -v
No browser, GPU, game or Windows keyboard access required.
"""
from collections import deque
import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import rs_v612_demo as demo
from rs_v612_demo import (RECORDER_VERSION, WINDOWS_KEYS, action_labels, canvas_monitor_match,
                          curate, frame_interval_stats, keyboard_delivery_stats, label_diagnostics,
                          monitor_windows_arrows, save_replay, summarize_capture_timing,
                          validity_stats)
from rs_v612_demo_bc import Replay, approved_runs


def load_tool():
    path = Path(__file__).resolve().parent/'tools'/'analyze_demo_run.py'
    spec = importlib.util.spec_from_file_location('analyze_demo_run', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kd(at, code, source='win32_key_state'):
    return {'at': at, 'type': 'keydown', 'code': code, 'repeat': False, 'source': source}


def fake_timing(t):
    return {'loop_start': t-.030, 'focus_done': t-.025, 'grab_start': t-.020, 'grab_end': t,
            'classify_done': t+.004, 'preprocess_done': t+.007, 'jpeg_done': t+.020,
            'iteration_end': t+.021, 'sleep_requested': .010, 'sleep_end': t+.035}


def make_samples(times, *, timing=False):
    samples = []
    for i, t in enumerate(times):
        s = {'at': float(t), 'frame': np.full((3, 288, 512), i % 255, dtype=np.uint8),
             'detection': None}
        if timing:
            s['timing'] = fake_timing(float(t))
        samples.append(s)
    return samples


MONITOR_CAPTURE = {'region': {'left': 0, 'top': 0, 'width': 1920, 'height': 1080},
                   'canvas_monitor_index': 1, 'keyboard_monitor_alive': True,
                   'keyboard_monitor_error': None}


def save_test_run(root, *, timing=True, capture_info=None, n=40):
    """20 FPS candidate run with four gameplay keydowns and one pre-run key."""
    folder = Path(root)/'runs'/'demo_test'
    times = 200. + np.arange(n) * .05
    events = [kd(200.06, 'ArrowLeft'), kd(200.56, 'ArrowRight'), kd(200.81, 'ArrowUp'),
              kd(201.31, 'ArrowDown'), kd(199.0, 'KeyS')]  # last: before gameplay
    meta = save_replay(make_samples(times, timing=timing), events, folder,
                       status='candidate', capture_info=capture_info)
    return folder, meta, times, events


def folder_snapshot(folder):
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(folder).iterdir())}


class LabelDiagnosticsTests(unittest.TestCase):
    def test_v12_like_slow_capture_drops_every_keydown_as_long_interval(self):
        rng = np.random.default_rng(12)
        times = 100. + np.concatenate([[0.], np.cumsum(rng.uniform(.23, .39, 50))])
        codes = ['KeyA', 'KeyD', 'KeyS', 'KeyW']
        keys = np.linspace(times[0]+.05, times[-1]-.05, 16)
        events = [kd(float(t), codes[i % 4]) for i, t in enumerate(keys)]
        _, valid = action_labels(times, events)
        summary, outcomes = label_diagnostics(times, events)
        self.assertEqual(int(valid.sum()), 0)
        self.assertEqual(summary['keydowns_eligible'], 16)
        self.assertEqual(summary['keydowns_assigned'], 0)
        self.assertEqual(summary['keydown_yield'], 0.)
        self.assertEqual(summary['outcomes'], {'long_interval': 16})
        self.assertEqual(summary['trusted_source'], 'win32_key_state')

    def test_every_keydown_accounted_once_and_matches_action_labels(self):
        rng = np.random.default_rng(3)
        gaps = rng.uniform(.04, .06, 300)
        gaps[::17] = .14  # some intervals beyond the 130 ms bound
        times = 50. + np.concatenate([[0.], np.cumsum(gaps)])
        codes = list(demo.ARROWS)
        events = [kd(float(t), codes[int(rng.integers(0, 4))])
                  for t in np.sort(rng.uniform(times[0]-.5, times[-1]+.5, 120))]
        actions, valid = action_labels(times, events)
        summary, outcomes = label_diagnostics(times, events)
        self.assertEqual(sum(summary['outcomes'].values()), summary['keydowns_total'])
        self.assertEqual(summary['keydowns_total'], len(demo.trusted_keydowns(events)))
        self.assertNotIn('unexplained', summary['outcomes'])
        assigned = [o for o in outcomes if o['outcome'] == 'assigned']
        self.assertEqual(len(assigned), int(np.count_nonzero(valid & (actions != 0))))
        for o in assigned:
            self.assertTrue(valid[o['frame']])
            self.assertEqual(demo.ACTION_NAMES[int(actions[o['frame']])], o['action'])
            self.assertLessEqual(o['offset_after_frame_ms'], 130.)
        self.assertGreater(summary['outcomes'].get('long_interval', 0), 0)

    def test_specific_drop_reasons(self):
        times = [0., .05, .10, .22, .27, .32]
        events = [kd(-.10, 'ArrowLeft'),                             # before first frame
                  kd(.02, 'ArrowLeft'), kd(.03, 'ArrowRight'),       # two actions, one interval
                  kd(.11, 'ArrowUp'), kd(.17, 'ArrowUp'),            # same key twice, 120 ms gap
                  kd(.27, 'ArrowDown'),                              # exactly on a frame timestamp
                  kd(.32, 'ArrowLeft'), kd(.40, 'ArrowRight')]       # at/after last frame
        summary, _ = label_diagnostics(times, events)
        self.assertEqual(summary['outcomes'], {'before_first_frame': 1, 'conflicting_keys': 2,
                                               'duplicate_key': 2, 'frame_boundary': 1,
                                               'after_last_frame': 2})
        self.assertEqual(summary['keydowns_eligible'], 5)
        self.assertEqual(summary['keydowns_assigned'], 0)
        self.assertEqual(summary['merged_by_dedup'], 0)

    def test_dedup_merges_are_counted(self):
        # Two LEFT keydowns 30 ms apart are one physical press under the 45 ms rule;
        # the browser copy is ignored because the Win32 stream is trusted.
        events = [kd(.01, 'ArrowLeft'), kd(.04, 'KeyA'), kd(.02, 'ArrowLeft', 'browser_dom')]
        summary, _ = label_diagnostics([0., .05, .10], events)
        self.assertEqual(summary['keydowns_total'], 1)
        self.assertEqual(summary['merged_by_dedup'], 1)
        self.assertEqual(summary['keydowns_assigned'], 1)

    def test_terminal_and_menu_keys_outside_gameplay_are_not_eligible(self):
        times = np.arange(10) * .05 + 10.
        events = [kd(9.0, 'KeyS'), kd(10.12, 'ArrowLeft'), kd(11.0, 'KeyA')]
        summary, _ = label_diagnostics(times, events)
        self.assertEqual(summary['keydowns_eligible'], 1)
        self.assertEqual(summary['keydown_yield'], 1.)


class StatisticsTests(unittest.TestCase):
    def test_frame_intervals_and_stack_gap(self):
        times = np.array([0., .05, .10, .30, .35, .40, .45, .50])
        fi = frame_interval_stats(times)
        self.assertEqual(fi['intervals_over_max_lag'], 1)
        self.assertAlmostEqual(fi['fraction_intervals_within_max_lag'], 6/7, places=4)
        self.assertEqual(fi['intervals_over_1s'], 0)
        valid = np.ones(len(times), dtype=bool)
        vs = validity_stats(np.zeros(len(times), dtype=int), valid, times)
        # Samples 3, 4, 5 have the .10->.30 gap inside their 4-frame history.
        self.assertEqual(vs['valid_with_stack_gap'], 3)
        self.assertEqual(vs['valid_by_action']['NONE'], 8)

    def test_keyboard_delivery_pairs_win32_and_browser(self):
        events = [kd(1.000, 'ArrowLeft'), kd(1.020, 'ArrowLeft', 'browser_dom'),
                  kd(2.000, 'KeyD'), kd(1.995, 'ArrowRight', 'browser_dom'),
                  kd(3.000, 'ArrowUp', 'browser_dom')]
        stats = keyboard_delivery_stats(events)
        self.assertEqual(stats['matched_pairs'], 2)
        self.assertEqual(stats['browser_before_win32'], 1)
        self.assertEqual(stats['browser_without_win32'], 1)
        self.assertAlmostEqual(stats['browser_minus_win32_ms']['median'], 7.5, places=1)

    def test_capture_timing_summary(self):
        rows = [fake_timing(1.), fake_timing(2.)]
        del rows[1]['jpeg_done']
        s = summarize_capture_timing(rows)
        self.assertAlmostEqual(s['grab']['median'], 20., places=3)
        self.assertEqual(s['jpeg_encode']['count'], 1)
        self.assertAlmostEqual(s['sleep_overshoot']['median'], 4., places=3)
        self.assertEqual(summarize_capture_timing([])['grab'], {'count': 0})

    def test_canvas_monitor_match(self):
        monitors = [{'left': 0, 'top': 0, 'width': 3840, 'height': 1080},
                    {'left': 0, 'top': 0, 'width': 1920, 'height': 1080},
                    {'left': 1920, 'top': 0, 'width': 1920, 'height': 1080}]
        self.assertEqual(canvas_monitor_match({'left': 1921, 'top': 0, 'width': 1919, 'height': 1080},
                                              monitors), 2)
        self.assertIsNone(canvas_monitor_match({'left': 10, 'top': 120, 'width': 1900, 'height': 945},
                                               monitors))
        self.assertIsNone(canvas_monitor_match(None, monitors))


class KeyboardMonitorTests(unittest.TestCase):
    def test_measures_poll_period_and_edge_events(self):
        events, stop, stats = [], threading.Event(), {}
        calls = {'n': 0}
        per_poll = len(WINDOWS_KEYS)

        def fake_state(vk):
            poll = calls['n'] // per_poll
            calls['n'] += 1
            if poll >= 10:
                stop.set()
            return 0x8000 if (vk == WINDOWS_KEYS['ArrowLeft'] and 3 <= poll < 6) else 0

        with mock.patch('builtins.print'):
            monitor_windows_arrows(events, stop, interval=.001, stats=stats, get_state=fake_state)
        self.assertEqual([(e['code'], e['type']) for e in events],
                         [('ArrowLeft', 'keydown'), ('ArrowLeft', 'keyup')])
        self.assertTrue(all(e['source'] == 'win32_key_state' for e in events))
        self.assertGreaterEqual(len(stats['poll_intervals']), 9)
        self.assertIsNone(stats['error'])
        self.assertEqual(stats['requested_interval_ms'], 1.)

    def test_records_fatal_error(self):
        stats = {}
        with self.assertRaises(OSError):
            monitor_windows_arrows([], threading.Event(), stats=stats,
                                   get_state=mock.Mock(side_effect=OSError('denied')))
        self.assertIn('denied', stats['error'])


class SaveReplayDiagnosticsTests(unittest.TestCase):
    def test_additive_fields_sidecars_and_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = dict(MONITOR_CAPTURE)
            folder, meta, times, _ = save_test_run(tmp, capture_info=info)
            self.assertEqual(meta['recorder_version'], RECORDER_VERSION)
            self.assertEqual(meta['timestamp_origin_perf_counter'], float(times[0]))
            d = meta['diagnostics']
            self.assertEqual(d['keydowns']['keydowns_eligible'], 4)
            self.assertEqual(d['keydowns']['keydowns_assigned'], 4)
            self.assertEqual(d['validity']['valid_nonnone_samples'], 4)
            self.assertEqual(meta['capture'], info)
            self.assertAlmostEqual(meta['capture_timing_ms']['jpeg_encode']['median'], 13., places=3)
            # Existing manifest keys keep their meaning.
            valid = np.load(folder/'valid.npy')
            self.assertEqual(meta['valid_action_samples'], int(valid.sum()))
            self.assertEqual(meta['schema'], 1)
            on_disk = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(on_disk['diagnostics']['keydowns']['keydowns_assigned'], 4)
            sidecar = json.loads((folder/'label_diagnostics.json').read_text(encoding='utf-8'))
            self.assertEqual(len(sidecar['keydowns']), 5)
            with np.load(folder/'capture_timing.npz') as z:
                self.assertEqual(len(z['grab_end']), len(times))
                self.assertAlmostEqual(float(z['grab_end'][0]), 0., places=9)
            # Curation and BC loading still work on the new manifest.
            curate(folder, 'approved', '0.0-1.8')
            runs = approved_runs(Path(tmp))
            self.assertEqual(len(runs), 1)
            replay = Replay(*runs[0])
            image, aux, action = replay.sample(5)
            self.assertEqual(image.shape, (12, 288, 512))
            self.assertEqual(aux.shape, (demo.AUX_DIM,))
            # Replay memory-maps .npy files; release them so Windows can delete tmp.
            del replay, image, aux
            gc.collect()

    def test_without_timing_or_capture_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, meta, _, _ = save_test_run(tmp, timing=False)
            self.assertNotIn('capture_timing_ms', meta)
            self.assertNotIn('capture', meta)
            self.assertFalse((folder/'capture_timing.npz').exists())
            self.assertTrue((folder/'label_diagnostics.json').exists())

    def test_diagnostics_failure_never_loses_a_run(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(demo, 'label_diagnostics', side_effect=RuntimeError('boom')):
            folder, meta, _, _ = save_test_run(tmp)
            self.assertIn('boom', meta['diagnostics']['error'])
            self.assertTrue((folder/'frames.npz').is_file())
            self.assertTrue((folder/'manifest.json').is_file())
            self.assertEqual(len(np.load(folder/'actions.npy')), 40)


class ConsoleSummaryTests(unittest.TestCase):
    def test_print_save_diagnostics_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('builtins.print') as out:
            _, meta, _, _ = save_test_run(tmp)
            demo.print_save_diagnostics(meta)
            demo.print_save_diagnostics({'diagnostics': {'error': 'x'}})
            demo.print_save_diagnostics({'diagnostics': {'frame_intervals': 'corrupt'}})
            demo.print_save_diagnostics({})
        text = ' '.join(str(c.args[0]) for c in out.call_args_list)
        self.assertIn('keydowns assigned 4/4 eligible', text)
        self.assertIn('summary unavailable', text)

    def test_capture_stage_line(self):
        line = demo.capture_stage_line([fake_timing(1.), fake_timing(2.)],
                                       {'poll_intervals': deque([.0156, .0157])})
        self.assertIn('grab 20.0', line)
        self.assertIn('key poll 15.6', line)


class AnalyzeToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def _new_run(self, root):
        return save_test_run(root, capture_info=dict(MONITOR_CAPTURE))

    def test_new_run_uses_stored_origin_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, meta, _, _ = self._new_run(tmp)
            before = folder_snapshot(folder)
            report = self.tool.analyze_run(folder)
            self.assertEqual(folder_snapshot(folder), before)
            self.assertEqual(report['origin']['method'], 'manifest')
            self.assertTrue(report['labels_reproducible'])
            self.assertEqual(report['keydowns']['keydown_yield'], 1.)
            status = {c['criterion']: c['status'] for c in report['criteria']}
            self.assertEqual(status['median frame interval (ms)'], 'PASS')
            self.assertEqual(status['keydown yield (eligible gameplay keydowns)'], 'PASS')
            self.assertEqual(status['capture geometry (recorder captures canvas only)'], 'INFO')
            self.assertEqual(len(report['lateral_review']['assigned']), 2)
            self.assertIn('grab', report['capture_timing_ms'])
            if report['preview'].get('available'):
                # Documents a KNOWN V1.3 issue (not fixed here): make_preview_video
                # truncates fps with int(), so exact 50 ms frames (1/0.05000000001
                # -> 19 fps) play ~5% fast. The tool must detect this.
                self.assertEqual(report['preview']['fps'], 19.)
                self.assertAlmostEqual(report['preview']['speed_ratio'], 20/19, places=3)
                self.assertEqual(status['preview playback speed ratio'], 'FAIL')
            self.assertIn('criteria:', self.tool.format_report(report))

    def test_legacy_v13_run_recovers_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, meta, times, _ = self._new_run(tmp)
            # Emulate a pre-diagnostics V1.3 replay: strip additive fields/sidecars.
            for key in ('recorder_version', 'timestamp_origin_perf_counter', 'diagnostics',
                        'capture_timing_ms', 'capture'):
                meta.pop(key, None)
            (folder/'manifest.json').write_text(json.dumps(meta), encoding='utf-8')
            (folder/'label_diagnostics.json').unlink()
            (folder/'capture_timing.npz').unlink()
            report = self.tool.analyze_run(folder, skip_preview=True)
            self.assertEqual(report['origin']['method'], 'recovered_from_labels')
            self.assertLess(abs(report['origin']['value'] - float(times[0])), .05)
            self.assertTrue(report['labels_reproducible'])
            self.assertEqual(report['keydowns']['keydowns_assigned'], 4)
            self.assertEqual(report['recorder_version'], '1.3 (pre-diagnostics)')
            status = {c['criterion']: c['status'] for c in report['criteria']}
            self.assertEqual(status['capture geometry (recorder captures canvas only)'], 'N/A')

    def test_unrecoverable_origin_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'runs'/'demo_none'
            save_replay(make_samples(300. + np.arange(10) * .05), [], folder, status='candidate')
            m = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            m.pop('timestamp_origin_perf_counter')
            (folder/'manifest.json').write_text(json.dumps(m), encoding='utf-8')
            report = self.tool.analyze_run(folder, skip_preview=True)
            self.assertEqual(report['origin']['method'], 'unrecoverable')
            self.assertNotIn('keydowns', report)

    def test_refuses_to_write_report_inside_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _, _, _ = self._new_run(tmp)
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                self.tool.main([str(folder), '--json', str(folder/'report.json'), '--skip-preview'])
            self.assertFalse((folder/'report.json').exists())


if __name__ == '__main__':
    unittest.main()
