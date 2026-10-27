"""Offline curation and action-alignment checks: python -m unittest test_rs_v612_demo -v"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rs_v612_demo import parse_segments, action_labels, curate, save_replay, FEATURE_VERSION, AUX_DIM, nonzero_approved_actions
from rs_v612_demo_bc import approved_runs


class DemonstrationTests(unittest.TestCase):
    def test_good_intervals(self):
        self.assertEqual(parse_segments('0.2-4.8, 5.2-7.1', 9.0), [[.2,4.8],[5.2,7.1]])
        for bad in ('3-2', '0-4,3-6', '0-99', 'one-5', '-1-3'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_segments(bad, 8.)

    def test_causal_action_labelling_and_repeats(self):
        t = [5., 5.05, 5.10, 5.15, 5.20, 5.25]
        events = [
            {'at':5.023, 'type':'keydown','code':'ArrowLeft','repeat':False},
            {'at':5.038, 'type':'keydown','code':'ArrowLeft','repeat':True},
            {'at':5.07, 'type':'keyup','code':'ArrowLeft','repeat':False},
            {'at':5.18, 'type':'keydown','code':'ArrowRight','repeat':False},
        ]
        actions, valid = action_labels(t, events)
        self.assertEqual(actions.tolist(), [1,0,0,2,0,0])
        self.assertEqual(valid.tolist(), [True,True,True,True,True,False])

    def test_ambiguous_simultaneous_keys_excluded(self):
        t = [0.,.05,.1]
        events = [{'at':.02,'type':'keydown','code':code,'repeat':False}
                  for code in ('ArrowLeft','ArrowRight')]
        _, valid = action_labels(t, events)
        self.assertFalse(valid[0])

    def test_save_replay_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'runs'/'demo_01'
            frames = [np.zeros((3,288,512), dtype=np.uint8) + i for i in range(12)]
            samples = [{'at':100.+i*.05, 'frame':fr, 'detection':None}
                       for i,fr in enumerate(frames)]
            events = [{'at':100.07, 'type':'keydown', 'code':'ArrowLeft', 'repeat':False},
                      {'at':100.27, 'type':'keydown', 'code':'ArrowRight', 'repeat':False}]
            meta = save_replay(samples, events, folder, status='candidate')
            self.assertEqual(meta['status'], 'candidate')
            self.assertEqual(len(np.load(folder/'actions.npy')), 12)
            self.assertTrue((folder/'frames.npz').is_file())
            self.assertEqual(approved_runs(Path(tmp)), [])
            self.assertEqual(int(np.load(folder/'actions.npy')[1]), 1)
            self.assertEqual(int(np.load(folder/'actions.npy')[5]), 2)
            self.assertEqual(np.load(folder/'aux.npy').shape, (12,AUX_DIM))
            self.assertEqual(len(np.load(folder/'valid.npy')),12)
            curate(folder, 'approved', '0.0-0.4')
            self.assertEqual(len(approved_runs(Path(tmp))), 1)

    def test_no_keys_cannot_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'runs'/'demo_no_keys'
            frames = [np.zeros((3,288,512), dtype=np.uint8) for _ in range(15)]
            samples = [{'at':100.+i*.05, 'frame':fr, 'detection':None}
                       for i,fr in enumerate(frames)]
            with self.assertRaisesRegex(ValueError, 'No correctly labeled physical arrow'):
                save_replay(samples, [], folder, status='approved', segments=[[0.,.5]])
            self.assertFalse(folder.exists())
            meta = save_replay(samples, [], folder, status='candidate')
            self.assertEqual(meta['action_counts_valid'], {'NONE':14})
            with self.assertRaisesRegex(ValueError, 'no valid physical arrow'):
                curate(folder, 'approved', '0-0.5')

    def test_approved_action_count_respects_trim(self):
        acts = np.array([1, 0, 2, 0, 0])
        valid = np.ones(5, dtype=bool)
        secs = np.arange(5.)
        self.assertEqual(nonzero_approved_actions(acts, valid, secs, [[1.,2.]]), 0)
        self.assertEqual(nonzero_approved_actions(acts, valid, secs, [[2.,3.]]), 1)

    def test_approval_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'runs'/'demo_01'
            folder.mkdir(parents=True)
            m = {'schema':1, 'feature_version':FEATURE_VERSION, 'aux_dim':AUX_DIM,
                 'status':'candidate', 'duration_seconds':10., 'valid_segments_seconds':[]}
            manifest = folder/'manifest.json'
            manifest.write_text(json.dumps(m), encoding='utf-8')
            np.save(folder/'actions.npy', np.array([0,1,0], dtype=np.int64))
            np.save(folder/'timestamps.npy', np.array([0.,1.,2.]))
            np.save(folder/'valid.npy', np.array([True,True,True]))
            before = approved_runs(root)
            self.assertEqual(before, [])
            after = curate(folder, 'approved', '0.5-8.6')
            self.assertEqual(after['valid_segments_seconds'], [[.5,8.6]])
            self.assertEqual(len(approved_runs(root)), 1)
            curate(folder, 'rejected')
            self.assertEqual(approved_runs(root), [])
            with self.assertRaises(ValueError):
                curate(folder, 'approved', '10-99')


if __name__ == '__main__':
    unittest.main()
