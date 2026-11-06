import unittest
from summarize_probes import phase_latencies


def row(episode,latency=None,kind='policy'):
    return dict(episode=episode,kind=kind,reaction_latency=(
        dict(policy_request_to_command_ms=latency) if latency is not None else None))


class PhaseTests(unittest.TestCase):
    def test_initial_reset_resume_and_normal_are_separate(self):
        phases=phase_latencies([row(1,170),row(1,35),row(1,kind='passive_resync'),
                               row(1,45),row(2,80),row(2,34)])
        self.assertEqual(phases,dict(initial_decision=[170.],post_reset=[80.],post_resume=[45.],ordinary=[35.,34.]))

    def test_none_first_decision_does_not_relabel_later_movement(self):
        phases=phase_latencies([row(1),row(1,35),row(2),row(2,34)])
        self.assertEqual(phases['ordinary'],[35.,34.])
        self.assertEqual(phases['initial_decision'],[])
        self.assertEqual(phases['post_reset'],[])

    def test_reset_takes_precedence_over_resume(self):
        phases=phase_latencies([row(1,35),row(1,kind='passive_resync'),row(2,80),row(2,34)])
        self.assertEqual(phases['post_reset'],[80.])
        self.assertEqual(phases['post_resume'],[])


if __name__=='__main__':unittest.main()
