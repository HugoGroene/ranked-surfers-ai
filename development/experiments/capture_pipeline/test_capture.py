import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from adapter import ARMS,ROOT,scaled_html
from batch import checkpoint_info,command,main

class CaptureTests(unittest.TestCase):
    def test_targeted_followup_plan_has_repeats_and_no_low_resolution_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'source.zip'
            with zipfile.ZipFile(checkpoint,'w') as z:z.writestr('data',json.dumps(dict(num_timesteps=123)))
            main(['--checkpoint',str(checkpoint),'--output',str(root/'plan'),
                  '--arms','bitmap_100','gpu_100','--workers','3','4','--repeats','2'])
            report=json.loads((root/'plan/comparison.json').read_text())
            jobs=report['plan'];self.assertEqual(report['status'],'PLANNED')
            self.assertEqual(len(jobs),14)
            self.assertTrue(all(j['kind']=='calibration' for j in jobs[:2]))
            for count in (3,4):
                selected=[j for j in jobs if j.get('workers')==count and j['arm']=='gpu_100']
                self.assertEqual(len(selected),2)
            self.assertTrue(all(j['arm'] in ('bitmap_100','gpu_100') for j in jobs))

    def test_scale_changes_only_expected_wrapper_anchors(self):
        original=(ROOT/'rankedsurfers-downloaded/game/releases/ae44e7782a4c8bbe5b10/arena.html').read_text()
        for scale in (.5,.75,1.):
            changed=scaled_html(original,scale)
            self.assertIn(f'renderBudget.configure(false,{scale})',changed)
            self.assertEqual(changed.count('instance.SendMessage'),original.count('instance.SendMessage'))
            restored=changed.replace(f'let renderScale = {scale};','let renderScale = 1;').replace(
                f'renderBudget.configure(false,{scale});','renderBudget.configure(d.preferences.autoResolution===true,ceiling);')
            self.assertEqual(restored,original)

    def test_changed_wrapper_fails_closed(self):
        with self.assertRaises(ValueError):scaled_html('unknown wrapper',.75)
        with self.assertRaises(ValueError):scaled_html('',0)

    def test_one_variable_per_arm(self):
        baseline=ARMS['bitmap_100']
        for name in ('bitmap_75','bitmap_50','gpu_100'):
            self.assertEqual(sum(ARMS[name][k]!=baseline[k] for k in baseline),1)
        self.assertEqual(ARMS['gpu_half_readback']['render_scale'],1.)
        self.assertEqual(ARMS['gpu_half_readback']['readback_width'],960)

    def test_benchmark_never_optimizes_or_enables_detector(self):
        cmd=command('gpu_100',3,1,Path('source.zip'),Path('out'))
        for flag in ('--benchmark-only','--profile-reaction','--telemetry'):self.assertIn(flag,cmd)
        self.assertEqual(cmd[cmd.index('--detector-mode')+1],'off')
        self.assertNotIn('--allow-input-migration',cmd)
        self.assertNotIn('--preview-port',cmd)

    def test_checkpoint_check_rejects_incomplete_and_preserves_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'latest.zip';p.write_bytes(b'')
            with self.assertRaises(zipfile.BadZipFile):checkpoint_info(p)
            with zipfile.ZipFile(p,'w') as archive:archive.writestr('data',json.dumps(dict(num_timesteps=123)))
            before=p.read_bytes();info=checkpoint_info(p)
            self.assertEqual(info['num_timesteps'],123);self.assertEqual(p.read_bytes(),before)

if __name__=='__main__':unittest.main()
