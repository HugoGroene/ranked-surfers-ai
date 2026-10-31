from pathlib import Path
from types import SimpleNamespace
import unittest
from tools.benchmark_ppo_detector import command


class DetectorBenchmarkTests(unittest.TestCase):
    def test_both_arms_use_same_source_budget_and_pipeline_with_only_detector_mode_changed(self):
        args=SimpleNamespace(checkpoint=Path('preserved.zip'),state_model=Path('screen.pt'),detector=Path('objects.pt'),
            workers=4,minutes=1,compact_observations=True,cache_detector_features=True,shared_inference=False,
            rollout_transport='shared_memory',opencv_threads=1,electron_host='per_worker')
        on=command(args,'on',Path('new_on'));off=command(args,'off',Path('new_off'))
        at=on.index('--detector-mode')+1;out=on.index('--report')+1
        expected=list(on);expected[at]='off';expected[out]='new_off'
        self.assertEqual(off,expected)
        self.assertIn('--benchmark-only',on)
        self.assertNotIn('--preview-port',on)
        self.assertEqual(on[on.index('--checkpoint')+1],'preserved.zip')
        self.assertNotIn('--shared-inference',on)

    def test_shared_resources_are_explicitly_selected(self):
        args=SimpleNamespace(checkpoint=Path('preserved.zip'),state_model=Path('screen.pt'),detector=Path('objects.pt'),
            workers=6,minutes=1,compact_observations=True,cache_detector_features=True,shared_inference=True,
            rollout_transport='shared_memory',opencv_threads=1,electron_host='shared')
        argv=command(args,'off',Path('new'))
        self.assertIn('--shared-inference',argv)
        self.assertEqual(argv[argv.index('--electron-host')+1],'shared')
        self.assertEqual(argv[argv.index('--detector-mode')+1],'off')
