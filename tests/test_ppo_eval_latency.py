import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from development.experiments.multimap_classifier.eval_latency import install


class EvalLatencyTests(unittest.TestCase):
    def test_timing_uses_policy_source_and_never_counts_none_as_a_command(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=NS(episode_number=1,output_dir=folder,step_timings=[],
                   _learning_observation_capture={'request_started_at':10.,'image_received_at':10.01},
                   _reaction_guard_capture={'request_started_at':10.015})
            seen=[]
            def step(action):
                seen.append(action)
                raw.step_timings.append(dict(press_timestamp=10.02 if action else None,press_roundtrip_ms=5))
                return ('observation',1,False,False,{})
            monitored=NS(step=step,close=lambda:seen.append('closed'))
            trainer=NS(create_live_env=lambda:(monitored,raw));install(trainer)
            env,_=trainer.create_live_env()
            self.assertEqual(env.step(3)[0],'observation');env.step(0)
            raw.episode_number=2;env.step(0);env.close()
            report=json.loads((Path(folder)/'evaluation_latency.json').read_text())
            self.assertEqual(seen,[3,0,0,'closed']);self.assertEqual(report['commands'],1)
            self.assertEqual(report['cadence']['window_samples'],1)
            self.assertAlmostEqual(report['latency']['metrics']['policy_request_to_command_ms']['median_ms'],20)


if __name__=='__main__':unittest.main()
