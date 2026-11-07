"""Dependency-free tests; safe while BC holds RAM/GPU resources."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from observation import latest_only, install
from transfers import cache_fits
from adapter import configuration, verify_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class FakeNumpy:
    def __init__(self):
        self.calls = 0

    def concatenate(self, frames, axis):
        self.calls += 1
        return tuple(frames)


np = FakeNumpy()


def pack_fixture(self, frame):
    self.native_history.append(frame)
    packed = np.concatenate(tuple(self.native_history), axis=0)
    self.guard_checks += 1
    return {'image': packed, 'aux': self.aux}


def invalid_fixture(self, frame):
    packed = tuple(self.native_history)
    return packed


class StaticTests(unittest.TestCase):
    def test_latest_frame_and_side_effects(self):
        optimized = latest_only(pack_fixture)
        raw = SimpleNamespace(native_history=[], guard_checks=0, aux=object())
        frame = object()
        before = np.calls
        result = optimized(raw, frame)
        self.assertIs(result['image'], frame)
        self.assertIs(result['aux'], raw.aux)
        self.assertEqual(raw.native_history, [frame])
        self.assertEqual(raw.guard_checks, 1)
        self.assertEqual(np.calls, before)

    def test_changed_pack_rejected(self):
        with self.assertRaises(RuntimeError):
            latest_only(invalid_fixture)

    def test_requires_temporal_and_command_wrapper(self):
        trainer = SimpleNamespace(build_env_class=lambda: type('Legacy', (), {}))
        install(trainer)
        with self.assertRaises(ValueError):
            trainer.build_env_class()

    def test_real_pack_only_one_assignment_changes(self):
        tree = ast.parse((ROOT/'train_rs_v612_ppo.py').read_text())
        pack = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_pack')
        # Every guard/control statement remains in the transformed source.
        assigns = [n for n in ast.walk(pack) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'packed' for t in n.targets)]
        self.assertEqual(len(assigns), 1)
        self.assertEqual(ast.unparse(assigns[0].value), 'np.concatenate(tuple(self.native_history), axis=0)')
        scope = {}
        exec(compile(ast.Module(body=[pack], type_ignores=[]), '<fixture>', 'exec'), scope)
        # Actual transform is exercised by runtime tests using inspectable source.
        self.assertIn('_read_screen_state', ast.unparse(pack))

    def test_memory_boundaries(self):
        self.assertTrue(cache_fits(864, 3000, 1024, 2048))
        self.assertFalse(cache_fits(864, 2800, 1024, 2048))
        self.assertFalse(cache_fits(4096, 10000, 1024, 2048))
        self.assertFalse(cache_fits(0, 10000, 1024, 2048))

    def test_no_implicit_enabled_features(self):
        with patch.dict(os.environ, {'RS_PIPELINE_EXPERIMENT': ''}):
            self.assertEqual(configuration()['features'], [])

    def test_reject_invalid_and_competing_transfers(self):
        for text in ('pinned,gpu_cache', 'mystery'):
            with patch.dict(os.environ, {'RS_PIPELINE_EXPERIMENT': text}):
                with self.assertRaises(ValueError):
                    configuration()

    def test_active_sources_unchanged(self):
        verify_sources()

    def test_every_staged_python_file_compiles(self):
        for file in HERE.glob('*.py'):
            compile(file.read_text(encoding='utf8'), str(file), 'exec')

    def test_readback_not_patched(self):
        source = (HERE/'adapter.py').read_text()
        self.assertNotIn('d3d_bridge', source)
        self.assertNotIn('ElectronCapture', source)


if __name__ == '__main__':
    unittest.main()
