import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from history_spacing import offsets,history_class
from road_config import configuration
from overnight_batch import cases,phase_plan

class SpacingTests(unittest.TestCase):
    def test_batch_budget_stops_before_launch_and_allows_complete_stage(self):
        from road_batch import check_stage_budget,BatchBudgetReached
        job={'command':['python','--minutes','25']}
        with patch('road_batch.time.monotonic',return_value=100):
            check_stage_budget(job,None)
            check_stage_budget(job,1780)
            with self.assertRaises(BatchBudgetReached):check_stage_budget(job,1779)

    def test_schedules_and_freshest_frame(self):
        for pattern in ('uniform','golden','logarithmic'):
            target=offsets(pattern,4,.5)
            self.assertEqual(target[0],.5);self.assertEqual(target[-1],0)
            self.assertTrue(np.all(np.diff(target)<0))
            h=history_class(pattern)(4,.5)
            origin=time.perf_counter()
            h.observation(np.zeros((3,1,1),np.uint8),origin)
            for n in range(1,61):h.add(np.full((3,1,1),n,np.uint8),origin+n/60,h.token())
            image,aux=h.observation(np.full((3,1,1),99,np.uint8),origin+1)
            self.assertEqual(image[-1,0,0],99)
            np.testing.assert_array_equal(aux[4:],1)
            np.testing.assert_allclose(aux[:4]*.56,target,atol=1/60+1e-6)
            h.clear()
            image,aux=h.observation(np.ones((3,1,1),np.uint8),time.perf_counter())
            np.testing.assert_array_equal(aux[4:],[0,0,0,1])

    def test_consecutive_tracks_packs_not_background_frames(self):
        h=history_class('consecutive')(4,.5);t=time.perf_counter()
        for n,delta in enumerate((0,.04,.09,.16)):
            h.add(np.full((3,1,1),88,np.uint8),t+delta,h.token())
            image,aux=h.observation(np.full((3,1,1),n+1,np.uint8),t+delta)
        np.testing.assert_array_equal(image[::3,0,0],[1,2,3,4])
        np.testing.assert_allclose(aux[:4]*.56,[.16,.12,.07,0],atol=1e-6)
        h.clear();h.add(np.full((3,1,1),99,np.uint8),t+2,0)
        image,aux=h.observation(np.ones((3,1,1),np.uint8),time.perf_counter())
        np.testing.assert_array_equal(aux[4:],[0,0,0,1])

    def test_spacing_contracts_differ(self):
        values=[configuration('road192_spatial',spacing=p) for p in ('uniform','consecutive','golden','logarithmic')]
        self.assertEqual(len({str(v) for v in values}),4)
        with self.assertRaises(ValueError):configuration('road192_spatial',6,spacing='golden')

    def test_plan_is_controlled_and_training_precedes_eval(self):
        self.assertEqual(len(cases()),10)
        with tempfile.TemporaryDirectory() as tmp:
            plan=phase_plan(Path(tmp),[123,456],25,50,6)
        jobs=plan['jobs'];self.assertEqual(len(jobs),40)
        self.assertEqual(len({j['output'] for j in jobs}),40)
        self.assertTrue(all(j['mode']=='train' for j in jobs[:20]))
        self.assertTrue(all(j['mode']=='eval' for j in jobs[20:]))
        for train,evaluation in zip(jobs[:20],jobs[20:]):
            self.assertEqual(train['name'],evaluation['name'])
            for flag in ('--arm','--spacing','--span-ms','--capture-backend','--seed'):
                self.assertEqual(train['command'][train['command'].index(flag)+1],evaluation['command'][evaluation['command'].index(flag)+1])
            self.assertIn(str(Path(train['output'])/'latest.zip'),evaluation['command'])
            self.assertNotIn('--checkpoint',train['command'])
            self.assertNotIn('--bc-root',train['command'])

if __name__=='__main__':unittest.main()
