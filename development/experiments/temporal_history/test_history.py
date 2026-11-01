import unittest
import numpy as np
from framebuffer import HistoryConfig, HistoryUnavailable, TimestampedFrameBuffer, require_matching_contract
from rates import RateGate
from try_rates import simulate


class HistoryTests(unittest.TestCase):
    def filled(self, **kwargs):
        b = TimestampedFrameBuffer(HistoryConfig(**kwargs))
        b.reset('episode')
        for i in range(61):
            b.append(np.full((3,2,3),i,np.uint8),i/60,segment='episode')
        return b

    def test_all_frame_counts_spans_and_rates(self):
        for n in range(6,13):
            for span in (.6,1.2):
                for decision in (20,30,60):
                    with self.subTest(n=n,span=span,decision=decision):
                        r = simulate(HistoryConfig(frames=n,span_seconds=span),60,decision,20,(3,2,3))
                        self.assertEqual(r['accepted'],20)
                        self.assertEqual(r['shape'],[n*3,2,3])

    def test_pixels_order_and_actual_ages(self):
        obs = self.filled().sample(1.001)
        self.assertEqual(len(set(obs.timestamps)),8)
        self.assertEqual(obs.timestamps[-1],1.)
        for i,t in enumerate(obs.timestamps):
            self.assertTrue(np.all(obs.image[i*3:(i+1)*3] == round(t*60)))
        np.testing.assert_allclose(obs.ages_seconds,1.001-np.array(obs.timestamps),atol=1e-7)

    def test_stale_future_and_nonfinite(self):
        b = self.filled()
        for t in (1.1,.99,float('nan')):
            with self.assertRaises(HistoryUnavailable): b.sample(t)

    def test_reset_and_segment_mismatch(self):
        b = self.filled()
        b.reset('resume')
        self.assertEqual(b.stored_bytes,0)
        with self.assertRaises(HistoryUnavailable): b.sample(2)
        with self.assertRaises(ValueError): b.append(np.zeros((3,2,3),np.uint8),2,segment='episode')

    def test_owned_pixels(self):
        b = TimestampedFrameBuffer()
        b.reset(1)
        f = np.zeros((3,2,3),np.uint8)
        for i in range(61):
            f.fill(i)
            b.append(f,i/60,segment=1)
        f.fill(255)
        self.assertEqual(b.sample(1).image[-1,0,0],60)

    def test_impossible_span_not_padded(self):
        r = simulate(HistoryConfig(frames=12,span_seconds=.1),30,30,10,(3,2,3))
        self.assertEqual(r['accepted'],0)

    def test_historical_gap_rejected(self):
        with self.assertRaises(HistoryUnavailable):
            self.filled(max_selection_error_seconds=.005).sample(1)

    def test_duplicate_or_backwards_capture_rejected(self):
        b = self.filled()
        for t in (1,.9,float('nan')):
            with self.assertRaises(ValueError): b.append(np.zeros((3,2,3),np.uint8),t,segment='episode')

    def test_memory_bound(self):
        b = self.filled(max_frames=12)
        self.assertLessEqual(b.stored_bytes,12*18)
        with self.assertRaises(HistoryUnavailable): b.sample(1)

    def test_contract_rejects_old_checkpoint(self):
        c = HistoryConfig().contract()
        require_matching_contract(c,c)
        for other in ({},HistoryConfig(span_seconds=.8).contract()):
            with self.assertRaises(ValueError): require_matching_contract(other,c)

    def test_rate_gate_skips_missed_deadlines(self):
        gate = RateGate(60)
        self.assertTrue(gate.due(0))
        self.assertFalse(gate.due(.001))
        self.assertTrue(gate.due(1))
        self.assertFalse(gate.due(1))
        with self.assertRaises(ValueError): gate.due(.5)

    def test_invalid_rates(self):
        for r in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): RateGate(r)


if __name__ == '__main__':
    unittest.main()
