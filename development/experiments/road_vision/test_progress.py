import unittest
from batch_progress import BatchProgress

class ProgressTests(unittest.TestCase):
    def test_throttled_progress_tracks_eval_and_outcomes(self):
        now=[0.];lines=[]
        p=BatchProgress(4,clock=lambda:now[0],emit=lambda line,**k:lines.append(line))
        p.phase='trials'
        job={'name':'sample','mode':'eval','command':['--episodes','200']}
        p(job,'START');self.assertIn('episodes=0/200',lines[-1])
        now[0]=5;p(job,'OUTPUT','[V61 EVAL] episode 12/200: test')
        self.assertEqual(len(lines),1)
        now[0]=31;p(job,'OUTPUT','other output')
        self.assertIn('episodes=12/200',lines[-1])
        for event in ('PASSED','FAILED','SKIPPED','RETAINED'):p(job,event)
        self.assertIn('stages=4/4 finished',lines[-1])
        self.assertIn('failed=1 skipped=1',lines[-1])

    def test_training_wall_time_is_not_claimed_as_active_training(self):
        now=[0.];lines=[]
        p=BatchProgress(2,clock=lambda:now[0],emit=lambda line,**k:lines.append(line))
        job={'name':'train','mode':'train','command':['--minutes','30']}
        p(job,'START');now[0]=61;p(job,'OUTPUT','test')
        self.assertIn('stage_wall=00:01:01',lines[-1])
        self.assertIn('training_budget=30m (excludes startup)',lines[-1])

if __name__=='__main__':unittest.main()
