import unittest
from unittest.mock import AsyncMock,Mock,patch
from types import SimpleNamespace
import stability_case as case

class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_asset_connection_recovers(self):
        route=SimpleNamespace(request=SimpleNamespace(method='GET'),fetch=AsyncMock(
            side_effect=[RuntimeError('connect ECONNREFUSED 127.0.0.1:8010'),'asset']))
        with patch.object(case.asyncio,'sleep',new_callable=AsyncMock):
            self.assertEqual(await case.fetch_local(route,'http://127.0.0.1:8010/asset'),'asset')
        self.assertEqual(route.fetch.await_count,2)

    async def test_local_asset_failure_is_bounded(self):
        route=SimpleNamespace(request=SimpleNamespace(method='GET'),fetch=AsyncMock(
            side_effect=RuntimeError('connect ECONNREFUSED 127.0.0.1:8010')))
        with patch.object(case.asyncio,'sleep',new_callable=AsyncMock):
            with self.assertRaises(RuntimeError):await case.fetch_local(route,'unused')
        self.assertEqual(route.fetch.await_count,3)

    async def test_writes_and_unrelated_errors_are_not_retried(self):
        for method,error in [('POST','connect ECONNREFUSED 127.0.0.1:8010'),('GET','invalid response')]:
            route=SimpleNamespace(request=SimpleNamespace(method=method),fetch=AsyncMock(
                side_effect=RuntimeError(error)))
            with self.assertRaises(RuntimeError):await case.fetch_local(route,'unused')
            self.assertEqual(route.fetch.await_count,1)

    async def test_recovers_consumed_paint_timeout(self):
        source=SimpleNamespace()
        call=AsyncMock(side_effect=[case.GuardUncertainError('No fresh offscreen paint within 500ms'),'fresh'])
        with patch.object(case,'original_capture_once',call):
            self.assertEqual(await case.fresh_capture(source),'fresh')
        self.assertEqual(call.await_count,2)
        self.assertEqual(source.multimap_paint_retries,1)

    async def test_retries_are_bounded(self):
        call=AsyncMock(side_effect=case.GuardUncertainError('No fresh offscreen paint within 500ms'))
        with patch.object(case,'original_capture_once',call):
            with self.assertRaises(case.GuardUncertainError):await case.fresh_capture(SimpleNamespace())
        self.assertEqual(call.await_count,3)

    async def test_other_errors_propagate(self):
        for error in (case.GuardUncertainError('invalid clock'),TimeoutError('socket')):
            call=AsyncMock(side_effect=error)
            with patch.object(case,'original_capture_once',call):
                with self.assertRaises(type(error)):await case.fresh_capture(SimpleNamespace())
            self.assertEqual(call.await_count,1)

    def test_preview_cap(self):
        with patch.object(case,'old_control',return_value=dict(enabled=True,fps=30)):
            self.assertEqual(case.preview_control('unused')['fps'],1)

    def test_only_selected_worker_installs_visible_backend(self):
        visible=Mock();ready=Mock();trainer=SimpleNamespace()
        modules={'visible_native':SimpleNamespace(install=visible),
                 'visible_startup':SimpleNamespace(install=ready),
                 'train_rs_v612_ppo':trainer}
        with patch.dict('os.environ',{'RS_MULTIMAP_VISIBLE_WORKER':'0'}),patch.dict('sys.modules',modules),patch.object(case,'original_worker') as original:
            case.worker_main(2,None,{},None,None)
            visible.assert_not_called();ready.assert_not_called()
            case.worker_main(0,None,{},None,None)
            visible.assert_called_once();ready.assert_called_once_with(trainer)
            self.assertEqual(original.call_count,2)

    def test_latest_selector_excludes_visible_diagnostics(self):
        import tempfile,json
        from pathlib import Path
        import evaluate_both
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,diagnostic in [('001',False),('002',True)]:
                run=root/'reports'/('multimap_stability_'+name)
                target=run/'stability/train';target.mkdir(parents=True)
                (run/'plan.json').write_text(json.dumps(dict(diagnostic_only=diagnostic)))
                (target/'latest.zip').write_bytes(b'test')
                (target/'parallel_training.json').write_text(json.dumps(dict(status='COMPLETE',
                    final_checkpoint_transitions=512,optimized_transitions=512,finalization_errors=[])))
            with patch.object(evaluate_both,'ROOT',root):
                self.assertIn('multimap_stability_001',str(evaluate_both.latest_checkpoint()))

    def test_halloween_odd_worker_does_not_install_tokyo_route(self):
        adapter=SimpleNamespace(install_scale=Mock())
        original_scale=adapter.install_scale
        with patch.dict('os.environ',{'RS_MULTIMAP_MAP_MODE':'halloween','RS_MULTIMAP_VISIBLE_WORKER':''}),patch.dict('sys.modules',{'adapter':adapter}),patch.object(case,'original_worker') as original:
            case.worker_main(1,None,{},None,None)
        self.assertIs(adapter.install_scale,original_scale)
        original.assert_called_once()

    def test_halloween_plot_pools_all_four_workers(self):
        import tempfile,json
        from pathlib import Path
        import stability
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index in range(4):
                worker=root/f'train/worker_{index}';worker.mkdir(parents=True)
                (worker/'survival_episodes.jsonl').write_text(json.dumps(dict(at=index,duration=index+1))+'\n')
            stability.plots(root,workers=4,show=False,map_mode='halloween')
            data=json.loads((root/'per_map_summary.json').read_text())
            self.assertEqual(list(data),['halloween'])
            self.assertEqual(data['halloween']['episodes'],4)
            self.assertEqual(data['halloween']['overall_mean'],2.5)

if __name__=='__main__':unittest.main()
