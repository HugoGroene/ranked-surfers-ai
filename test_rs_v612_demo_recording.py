"""Offline regression tests: no completed demonstration may be lost.

python -m unittest test_rs_v612_demo_recording -v
Covers autosave as candidate, runs starting before review, review commands
while a save is in flight (terminal focused), failed saves, shutdown with queued
saves, recoverable calibration, geometry-change logging and session logs.
No browser, game, GPU or keyboard access required.
"""
import asyncio
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest import mock
from datetime import datetime

import numpy as np

import rs_v612_demo as demo
from rs_v612_demo import (RunStore, build_capture_snapshot, calibrate_or_pause, open_session_log,
                          run_with_session_log, save_replay, unique_run_folder, view_changes)
from rs_v612_demo_bc import approved_runs


def kd(at, code):
    return {'at': at, 'type': 'keydown', 'code': code, 'repeat': False, 'source': 'win32_key_state'}


def run_data(start, n=30, keys=True):
    # 30 frames = 1.45 s, so G's default 0.7 s tail trim still leaves a segment.
    samples = [{'at': start + i*.05, 'frame': np.full((3, 288, 512), i, dtype=np.uint8),
                'detection': None} for i in range(n)]
    events = [kd(start + .06, 'ArrowLeft'), kd(start + .21, 'ArrowRight')] if keys else []
    return samples, events


def manifest(folder):
    return json.loads((Path(folder)/'manifest.json').read_text(encoding='utf-8'))


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, **kw):
        s = RunStore(self.root, log=self.logs.append, opener=kw.pop('opener', mock.Mock()), **kw)
        self.addCleanup(s.shutdown)
        return s


class AutosaveTests(StoreCase):
    def test_completed_run_is_saved_as_candidate_never_approved(self):
        s = self.store()
        rec = s.complete(*run_data(100.), reason='completed', capture_info={'region': None})
        s.shutdown()
        self.assertEqual(rec.state, 'saved')
        m = manifest(rec.folder)
        self.assertEqual(m['status'], 'candidate')
        self.assertEqual(m['reason'], 'completed')
        self.assertEqual(m['capture'], {'region': None})
        for name in ('frames.npz', 'aux.npy', 'actions.npy', 'valid.npy', 'timestamps.npy', 'events.jsonl'):
            self.assertTrue((rec.folder/name).is_file(), name)
        self.assertEqual(int(np.load(rec.folder/'actions.npy')[1]), 1)  # original labels kept
        self.assertEqual(len((rec.folder/'events.jsonl').read_text().splitlines()), 2)
        self.assertIsNone(rec.samples)  # memory released only after a successful save
        self.assertEqual(approved_runs(self.root), [])

    def test_new_run_before_review_keeps_both_runs(self):
        s = self.store()
        first = s.complete(*run_data(100.), reason='completed')
        second = s.complete(*run_data(200.), reason='completed')
        s.command('g').result(timeout=30)
        s.shutdown()
        self.assertNotEqual(first.folder, second.folder)
        self.assertEqual(manifest(first.folder)['status'], 'candidate')
        self.assertEqual(manifest(second.folder)['status'], 'approved')
        self.assertEqual(len(list((self.root/'runs').glob('demo_*'))), 2)

    def test_partial_and_forced_runs_are_also_preserved(self):
        s = self.store()
        for reason in ('geometry_changed_partial', 'focus_lost_partial', 'max_run_length'):
            s.complete(*run_data(100.), reason=reason)
        s.shutdown()
        reasons = sorted(manifest(r.folder)['reason'] for r in s.records)
        self.assertEqual(reasons, ['focus_lost_partial', 'geometry_changed_partial', 'max_run_length'])

    def test_too_short_run_is_reported_not_crashing(self):
        s = self.store()
        self.assertIsNone(s.complete(*run_data(100., n=3), reason='completed'))
        self.assertTrue(any('too short' in m for m in self.logs))


class ReviewCommandTests(StoreCase):
    def test_command_typed_while_save_in_flight_applies_after_save(self):
        gate = threading.Event()

        def slow_save(*args, **kw):
            self.assertTrue(gate.wait(10))
            return save_replay(*args, **kw)

        s = self.store(save=slow_save)
        rec = s.complete(*run_data(100.), reason='completed')
        future = s.command('g')  # terminal focused, save not finished yet
        self.assertFalse(future.done())
        self.assertEqual(s.pending(), 2)
        gate.set()
        self.assertEqual(future.result(timeout=30), rec.folder)
        self.assertEqual(manifest(rec.folder)['status'], 'approved')

    def test_trim_approve_preview_and_reject(self):
        opener = mock.Mock()
        s = self.store(opener=opener)
        rec = s.complete(*run_data(100.), reason='completed')
        s.command('t', '0.0-0.3').result(timeout=30)
        self.assertEqual(manifest(rec.folder)['valid_segments_seconds'], [[0.0, 0.3]])
        s.command('p').result(timeout=30)
        opener.assert_called_once()
        self.assertTrue(Path(opener.call_args[0][0]).exists())
        s.command('r').result(timeout=30)
        self.assertEqual(manifest(rec.folder)['status'], 'rejected')
        self.assertTrue((rec.folder/'frames.npz').is_file())  # R never deletes data

    def test_approval_refused_without_actions_keeps_candidate(self):
        s = self.store()
        rec = s.complete(*run_data(100., keys=False), reason='completed')
        self.assertIsNone(s.command('g').result(timeout=30))
        self.assertEqual(manifest(rec.folder)['status'], 'candidate')
        self.assertTrue(any('Could not apply G' in m for m in self.logs))

    def test_command_before_any_run(self):
        s = self.store()
        self.assertIsNone(s.command('s'))
        self.assertTrue(any('No completed run yet' in m for m in self.logs))


class FailureTests(StoreCase):
    def flaky(self, failures):
        calls = {'n': 0}

        def save(samples, events, folder, **kw):
            calls['n'] += 1
            if calls['n'] <= failures:
                folder.mkdir(parents=True)
                (folder/'partial.txt').write_text('x')
                raise OSError('disk full')
            return save_replay(samples, events, folder, **kw)
        return save

    def test_failed_autosave_keeps_run_and_s_retries_into_new_folder(self):
        s = self.store(save=self.flaky(1))
        rec = s.complete(*run_data(100.), reason='completed')
        s._futures[-1].result(timeout=30)
        self.assertEqual(rec.state, 'failed')
        self.assertEqual(len(rec.samples), 30)  # still buffered in memory
        self.assertTrue(any('was NOT saved' in m and 'S + Enter' in m for m in self.logs))
        failed_folder = next((self.root/'runs').glob('demo_*'))
        s.command('r').result(timeout=30)  # reject cannot drop an unsaved run
        self.assertEqual(len(rec.samples), 30)
        folder = s.command('s').result(timeout=30)
        self.assertEqual(rec.state, 'saved')
        self.assertNotEqual(folder, failed_folder)
        self.assertEqual(sorted(p.name for p in failed_folder.iterdir()), ['partial.txt'])
        self.assertEqual(manifest(folder)['status'], 'candidate')

    def test_shutdown_waits_for_queued_saves(self):
        gate = threading.Event()

        def slow_save(*args, **kw):
            gate.wait(.3)
            return save_replay(*args, **kw)

        s = self.store(save=slow_save)
        recs = [s.complete(*run_data(100. + 50*i), reason='completed') for i in range(3)]
        s.shutdown()
        self.assertTrue(all(r.state == 'saved' and (r.folder/'manifest.json').exists() for r in recs))
        self.assertTrue(any('Waiting for' in m for m in self.logs))
        self.assertTrue(any('3 run(s) saved' in m for m in self.logs))

    def test_shutdown_makes_final_attempt_for_unsaved_run(self):
        s = self.store(save=self.flaky(1))
        rec = s.complete(*run_data(100.), reason='completed')
        s.shutdown()
        self.assertEqual(rec.state, 'saved')
        self.assertEqual(manifest(rec.folder)['status'], 'candidate')

    def test_shutdown_reports_runs_that_could_not_be_saved(self):
        s = self.store(save=self.flaky(99))
        s.complete(*run_data(100.), reason='completed')
        s.shutdown()
        self.assertTrue(any('UNSAVED runs: [1]' in m for m in self.logs))


class FolderAndLogTests(unittest.TestCase):
    def test_unique_run_folder_never_reuses_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2026, 10, 28, 9, 0, 0, 123456)
            first = unique_run_folder(tmp, now=now)
            first.mkdir(parents=True)
            second = unique_run_folder(tmp, now=now)
            self.assertNotEqual(first, second)
            self.assertTrue(second.name.startswith(first.name))
            with self.assertRaises(FileExistsError):  # save_replay's final guard
                save_replay(run_data(1.)[0], [], first, status='candidate')

    def test_session_logs_are_unique_and_capture_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2026, 10, 28, 9, 0, 0, 1)
            p1, h1 = open_session_log(tmp, now=now)
            p2, h2 = open_session_log(tmp, now=now)
            h1.close(); h2.close()
            self.assertNotEqual(p1, p2)
            console = io.StringIO()
            with mock.patch.object(sys, 'stdout', console):
                run_with_session_log(tmp, lambda: print('hello session'))
            logs = sorted((Path(tmp)/'logs').glob('*.log'))
            self.assertEqual(len(logs), 3)
            newest = [p for p in logs if p not in (p1, p2)][0]
            self.assertIn('hello session', newest.read_text(encoding='utf-8'))
            self.assertIn('hello session', console.getvalue())

    def test_session_log_records_crash_traceback_and_restores_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = sys.stdout
            with mock.patch.object(sys, 'stdout', io.StringIO()):
                inner = sys.stdout
                with self.assertRaises(RuntimeError):
                    run_with_session_log(tmp, lambda: (_ for _ in ()).throw(RuntimeError('calibration boom')))
                self.assertIs(sys.stdout, inner)
            self.assertIs(sys.stdout, before)
            text = next((Path(tmp)/'logs').glob('*.log')).read_text(encoding='utf-8')
            self.assertIn('calibration boom', text)
            self.assertIn('Traceback', text)


class CalibrationTests(unittest.TestCase):
    def test_failed_calibration_pauses_instead_of_raising(self):
        logs = []

        async def failing(*args, **kw):
            raise RuntimeError('Cannot safely calibrate fast canvas capture.')

        with mock.patch.object(demo, 'calibrate_canvas_mss', failing):
            result = asyncio.run(calibrate_or_pause(None, None, None, log=logs.append))
        self.assertEqual(result, (None, None))
        self.assertTrue(any('PAUSED' in m and 'completed runs are safe' in m for m in logs))

    def test_successful_calibration_passes_through_threshold_unchanged(self):
        seen = {}

        async def ok(page, canvas, grabber, *, attempts, with_score):
            seen.update(attempts=attempts, with_score=with_score)
            return {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}, .99

        with mock.patch.object(demo, 'calibrate_canvas_mss', ok):
            region, score = asyncio.run(calibrate_or_pause(None, None, None, attempts=1))
        self.assertEqual(score, .99)
        self.assertEqual(seen, {'attempts': 1, 'with_score': True})

    def test_view_changes_names_fields(self):
        old = (1920, 945, 1.0, 0, 0, 0.0, 0.0, 1920.0, 945.0)
        new = (1920, 1080, 1.0, 0, 0, 0.0, 0.0, 1920.0, 1080.0)
        self.assertEqual(view_changes(old, new), ['inner_height: 945 -> 1080', 'canvas_height: 945.0 -> 1080.0'])
        self.assertEqual(view_changes(old, None), ['game canvas not visible'])

    def test_capture_snapshot_while_paused(self):
        snap = build_capture_snapshot({'recorder_version': 'x'}, region=None, calibration_score=None,
                                      monitors=[{'left': 0, 'top': 0, 'width': 1920, 'height': 1080}],
                                      canvas_view=None, keyboard_stats={}, keyboard_alive=True)
        self.assertIsNone(snap['region'])
        self.assertIsNone(snap['view_signature'])
        self.assertIsNone(snap['canvas_monitor_index'])
        json.dumps(snap)


class ConsoleTests(unittest.TestCase):
    def test_console_reader_stops_reading_after_quit(self):
        commands = queue.Queue()
        answers = iter(['s', 'x', 'q', 'never read'])
        with mock.patch('builtins.input', side_effect=lambda *_: next(answers)), \
                mock.patch('builtins.print'):
            demo.console_reader(commands, threading.Event())
        self.assertEqual([commands.get_nowait(), commands.get_nowait()], [('s', ''), ('q', '')])
        self.assertTrue(commands.empty())
        self.assertEqual(next(answers), 'never read')


if __name__ == '__main__':
    unittest.main()
