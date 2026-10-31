import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.plot_ppo_survival import build_figure,end_plot_mode,finish_plot,plot_run


class SurvivalPlotTests(unittest.TestCase):
    def data(self):
        return dict(requested_seconds=300,window_episodes=100,completed_episodes=123,
            overall_mean_seconds=4.,time_limited_episodes=1,
            points=[dict(seconds=10,mean_seconds=3.2),dict(seconds=20,mean_seconds=4.5)])

    def test_auto_uses_dashboard_and_benchmark_status_and_explicit_modes_override(self):
        self.assertEqual(end_plot_mode('auto',None),'show')
        self.assertEqual(end_plot_mode('auto',8001),'off')
        self.assertEqual(end_plot_mode('auto',None,True),'off')
        for mode in ['show','save','off']:
            self.assertEqual(end_plot_mode(mode,8001,True),mode)
        with self.assertRaises(ValueError):end_plot_mode('unknown',None)

    def test_exact_saved_points_full_axis_and_empty_data(self):
        figure=build_figure(self.data(),dict(workers=1,detector_mode='off',status='COMPLETE'))
        try:
            axis=figure.axes[0]
            self.assertEqual(list(axis.lines[0].get_xdata()),[10/60,20/60])
            self.assertEqual(list(axis.lines[0].get_ydata()),[3.2,4.5])
            self.assertEqual(axis.get_xlim(),(0.,5.))
            self.assertIn('123 completed episodes',axis.get_title())
            self.assertIn('overall mean 4.00s',axis.get_title())
        finally:plt.close(figure)
        data=self.data();data['points']=[];data['completed_episodes']=0;data['overall_mean_seconds']=None
        figure=build_figure(data,{})
        try:
            self.assertEqual(len(figure.axes[0].lines),0)
            self.assertIn('No completed episodes',figure.axes[0].texts[0].get_text())
        finally:plt.close(figure)

    def test_saves_before_showing_and_preserves_existing_results_and_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            original=json.dumps(self.data())
            (folder/'survival.json').write_text(original)
            (folder/'runtime.json').write_text('{"workers":1,"detector_mode":"off"}')
            (folder/'survival_curve.png').write_bytes(b'preserved')
            def show(**kwargs):
                self.assertTrue(kwargs['block'])
                self.assertTrue((folder/'survival_curve_2.png').read_bytes().startswith(b'\x89PNG'))
            with patch.object(plt,'show',side_effect=show) as display:
                result=plot_run(folder,show=True)
                self.assertEqual(result.name,'survival_curve_2.png')
                display.assert_called_once()
            self.assertEqual((folder/'survival_curve.png').read_bytes(),b'preserved')
            self.assertEqual((folder/'survival.json').read_text(),original)
            with patch.object(plt,'show') as display:
                self.assertEqual(plot_run(folder,show=False).name,'survival_curve_3.png')
                display.assert_not_called()

    def test_disabled_plot_does_no_io_and_plot_errors_do_not_fail_training(self):
        with patch('tools.plot_ppo_survival.plot_run') as render:
            self.assertIsNone(finish_plot('missing','off',None))
            render.assert_not_called()
            render.side_effect=RuntimeError('GUI unavailable')
            self.assertIsNone(finish_plot('missing','show',None))

    def test_invalid_axis_or_nonfinite_points_are_rejected(self):
        for requested in [0,-1,float('nan')]:
            data=self.data();data['requested_seconds']=requested
            with self.assertRaises(ValueError):build_figure(data,{})
        for point in [dict(seconds=10,mean_seconds=float('nan')),
                      dict(seconds=301,mean_seconds=4)]:
            data=self.data();data['points']=[point]
            with self.assertRaises(ValueError):build_figure(data,{})
