import unittest
from tools.run_ppo_live_checks import summarize,settle_startup
from unittest.mock import Mock


class LiveCheckReportTests(unittest.TestCase):
    def test_lead_in_only_sends_none_and_returns_fresh_observation(self):
        result=('fresh',0,False,False,{})
        step=Mock(return_value=result)
        self.assertEqual(settle_startup(step,1,Mock(side_effect=[0,0,.5,1])),result)
        self.assertEqual([call.args for call in step.call_args_list],[(0,),(0,)])

    def test_terminal_lead_in_stops_without_more_actions(self):
        step=Mock(return_value=('terminal',0,True,False,{}))
        with self.assertRaises(RuntimeError):
            settle_startup(step,1,Mock(side_effect=[0,0]))
        step.assert_called_once_with(0)

    def test_collection_never_claims_verified_movement(self):
        report=summarize([{}]*10,10,None)
        self.assertEqual(report['status'],'COLLECTED_REVIEW_REQUIRED')
        self.assertFalse(report['movement_verified'])
        self.assertFalse(report['unity_clock_verified'])

    def test_partial_and_error_are_incomplete(self):
        for trials,error in [([{}],None),([{}]*10,'timeout')]:
            self.assertEqual(summarize(trials,10,error)['status'],'INCOMPLETE')
