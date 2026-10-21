#!/usr/bin/env python3
"""Offline, dependency-light V6.1 feature/reward tests (no browser or GPU)."""
import unittest
from collections import deque
import numpy as np
from rs_v5_shadow_segmentation import V5_CLASSES
from rs_v6_ppo_features import HAZARD_DIM, encode_hazards
from rs_v61_features import (AUX_DIM, V6_AUX_DIM, ActionContext,
                              encode_aux, screen_columns, SCREEN_COLUMN_DIM)
from rs_v61_rewards import ProgressRewards


class Features(unittest.TestCase):
    def setUp(self):
        self.snapshot = {'episode':12,'captured_at':20.,'detections':[
            {'class':'coin','conf':.9,'box':[.27,.3,.48,.8]},
            {'class':'train_front','conf':.75,'box':[.79,.1,.98,.7]},
        ]}

    def test_aux_shape_and_range(self):
        hazards,age,fresh=encode_hazards(self.snapshot,now=20.08,episode_tag=12)
        self.assertTrue(fresh)
        ctx=ActionContext()
        aux=encode_aux(hazards,deque([2,3],maxlen=4),.8,deque([.032],maxlen=3),age,
                       snapshot=self.snapshot,episode_tag=12,now=20.08,
                       action_context=ctx)
        self.assertEqual(AUX_DIM,557)
        self.assertEqual(V6_AUX_DIM,467)
        self.assertEqual(aux.shape,(AUX_DIM,))
        self.assertTrue(np.isfinite(aux).all())
        self.assertGreaterEqual(float(aux.min()),0.)
        self.assertLessEqual(float(aux.max()),1.)
        np.testing.assert_array_equal(aux[V6_AUX_DIM+SCREEN_COLUMN_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM+4],0)

    def test_columns_cross_image_boundaries(self):
        out=screen_columns(self.snapshot,episode_tag=12,now=20.08).reshape(9,3,3)
        self.assertGreater(out[V5_CLASSES.index('coin'),0,0],0.)
        self.assertGreater(out[V5_CLASSES.index('coin'),1,0],0.)
        self.assertEqual(float(out[V5_CLASSES.index('coin'),2,0]),0.)
        self.assertGreater(out[V5_CLASSES.index('train_front'),2,0],0.)
        self.assertEqual(out.shape,(9,3,3))

    def test_stale_and_episode_isolation(self):
        stale=screen_columns(self.snapshot,episode_tag=12,now=20.6)
        different=screen_columns(self.snapshot,episode_tag=99,now=20.08)
        self.assertEqual(stale.shape,(SCREEN_COLUMN_DIM,))
        self.assertFalse(stale.any())
        self.assertFalse(different.any())
        hz,age,fresh=encode_hazards(self.snapshot,now=20.6,episode_tag=12)
        self.assertFalse(fresh)
        self.assertFalse(hz.any())

    def test_lane_is_unverified_and_command_timing(self):
        x=ActionContext()
        x.note_action(1,10.0)
        self.assertEqual(x.commanded_lane,0)
        self.assertEqual(x.lane_confidence,0.)
        np.testing.assert_array_equal(x.vector(10.2)[:4],0.)
        self.assertAlmostEqual(float(x.vector(10.2)[4+1]),.2,places=5)
        y=ActionContext(assume_center=True)
        self.assertEqual(y.commanded_lane,1)
        self.assertLessEqual(y.lane_confidence,.25)
        y.note_action(1,10.)
        self.assertLess(y.lane_confidence,.25)
        y.note_action(1,10.01)
        self.assertEqual(y.commanded_lane,0)
        self.assertTrue(np.isfinite(y.vector(10.02)).all())


class Rewards(unittest.TestCase):
    def test_dense_progress_and_death(self):
        r=ProgressRewards()
        val,parts=r.step(.1,False)
        self.assertAlmostEqual(val,.08,places=6)
        val,parts=r.step(.51,False)
        self.assertEqual(parts['milestones'],.15)
        self.assertGreater(parts['survival'],0)
        val,parts=r.step(.8,True)
        self.assertEqual(parts['death'],-1.)
        self.assertEqual(parts['milestones'],0.)

    def test_repeat_duration_cannot_farm_reward(self):
        r=ProgressRewards()
        r.step(.5,False)
        val,parts=r.step(.5,False)
        self.assertEqual(val,0.)
        self.assertEqual(parts['survival'],0.)
        self.assertEqual(parts['milestones'],0.)
        r.reset()
        val,parts=r.step(.5,False)
        self.assertEqual(parts['milestones'],.15)

    def test_nonfinite_rejected(self):
        r=ProgressRewards()
        with self.assertRaises(ValueError):r.step(float('nan'),False)


if __name__=='__main__':
    unittest.main()
