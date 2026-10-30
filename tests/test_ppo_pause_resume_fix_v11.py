"""Offline-only checks for pause probe v1.1; do not open a browser."""
import ast
from pathlib import Path
import unittest
from tools.apply_pause_probe_resume_fix import apply_patch, OLD, NEW

ROOT = Path(__file__).resolve().parents[1]


class PauseProbeResumeFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_probe = (ROOT / 'tools' / 'probe_unity_pause.py').read_text(encoding='utf-8')

    def test_compiles_installed_probe(self):
        compile(self.original_probe, 'tools/probe_unity_pause.py', 'exec')

    def test_installed_probe_has_overlay_resume(self):
        self.assertIn(NEW, self.original_probe)
        self.assertNotIn(OLD, self.original_probe)

    def test_no_forced_click(self):
        self.assertNotIn('force=True', self.original_probe.replace('never use force=True', ''))

    def test_checks_pause_overlay_hidden_after_resume(self):
        self.assertIn("await overlay.wait_for(state='hidden'", self.original_probe)

    def test_patcher_refuses_second_install(self):
        with self.assertRaisesRegex(ValueError, 'already'):
            apply_patch(self.original_probe)

    def test_patcher_refuses_unknown_script(self):
        with self.assertRaisesRegex(ValueError, 'Expected unmodified'):
            apply_patch('def main():\n    pass\n')

    def test_new_selector_scoped_to_overlay(self):
        self.assertEqual(self.original_probe.count("overlay.get_by_role('button', name='Resume', exact=True)"), 1)

    def test_patcher_does_not_change_browser_logic(self):
        original_source = self.original_probe.replace(NEW, OLD)
        self.assertEqual(apply_patch(original_source), self.original_probe)

    def test_patcher_keeps_screenshot_and_motion_checks(self):
        self.assertIn("'resumed_scene': resumed", self.original_probe)
        self.assertIn("'resume_motion_seen':", self.original_probe)


if __name__ == '__main__':
    unittest.main()
