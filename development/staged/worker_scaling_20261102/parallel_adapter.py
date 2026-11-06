"""Spawn-safe opt-in five-frame environment for the existing shared PPO learner."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HYBRID=HERE.parent/'hybrid_history_20261102'
ROAD=ROOT/'development/experiments/road_vision'
sys.path[:0]=[str(ROOT),str(HYBRID),str(ROAD)]
import run_probe

DEFAULT_CHECKPOINT=ROOT/'reports/hybrid_capacity_20261102_193304_470308/performance/road384_recent3_context2/train/latest.zip'


def sources():
    result=run_probe.sources()
    for file in HERE.glob('*.py'):
        result[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def configuration():
    from road_config import configuration as base
    c=base('road384_spatial',4,1000,60,30)
    c['temporal'].update(frames=5,spacing='hybrid',layout='recent3_context2',recent_frames=3,
        context_offsets_seconds=[.5,1.],padding='zero_with_age_and_validity',
        history_schema='recent_plus_context_v1')
    return c


def install():
    import torch
    import cv2
    import train_rs_v612_ppo as trainer
    if getattr(trainer,'_hybrid_parallel_ready',False):return trainer
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from capture_backend import install_backend
    install_backend('gpu_100')
    from run_trial import install_rate_bridge
    install_rate_bridge()
    from hybrid import HybridHistory
    import history_spacing
    original_history=history_spacing.history_class
    class SelectedHistory(HybridHistory):
        def __init__(self,frames,span):
            if frames!=5 or span!=1.:raise ValueError('Wrong worker history contract')
            super().__init__(3,(.5,1.))
    history_spacing.history_class=lambda pattern: SelectedHistory if pattern=='hybrid' else original_history(pattern)

    from tools import ppo_headless_env as headless
    original_headless=headless.install
    def headless_once(target,capture='playwright',*,profile=False,electron_transport='socket',electron_window=None):
        if target is not trainer or capture!='electron' or electron_transport!='shared_memory' or electron_window is not None:
            raise ValueError('Worker benchmark requires separate Electron hosts and shared-memory capture')
        if getattr(target,'_hybrid_parallel_ready',False):return
        original_headless(target,capture,profile=True,electron_transport=electron_transport)
    headless.install=headless_once
    from road_adapter import install as road_install
    road_install(trainer,configuration())
    from tools.ppo_settled_learning import SettledPPO
    original_collect=SettledPPO.collect_rollouts
    def collect(self,*args,**kwargs):
        started=time.perf_counter()
        try:return original_collect(self,*args,**kwargs)
        finally:
            raw=getattr(self,'_settled_raw',None)
            if raw is not None:
                if not hasattr(raw,'_scaling_rollouts'):raw._scaling_rollouts=[]
                raw._scaling_rollouts.append(dict(seconds=time.perf_counter()-started,
                    timesteps=self.num_timesteps))
    SettledPPO.collect_rollouts=collect
    original_parse=trainer.parse_args
    def parse(*args,**kwargs):
        a=original_parse(*args,**kwargs)
        a.resolution=(384,384);a.resize_method='area'
        a.fps=30.;a.rate_mode='capped';a.torch_threads=1
        return a
    trainer.parse_args=parse
    original_build=trainer.build_env_class
    def build():
        base=original_build()
        class HybridWorkerEnv(base):
            async def _close_browser(self):
                closed={'status':'STOPPED'}
                try:
                    await super()._close_browser()
                    closed['status']='COMPLETE'
                except BaseException as exc:
                    closed['error']=repr(exc)
                    raise
                finally:
                    (self.output_dir/'hybrid_history.json').write_text(json.dumps(self.temporal.report(),indent=2))
                    (self.output_dir/'rollout_timings.json').write_text(json.dumps(getattr(self,'_scaling_rollouts',[]),indent=2))
                    (self.output_dir/'hybrid_close.json').write_text(json.dumps(closed,indent=2))
        return HybridWorkerEnv
    trainer.build_env_class=build
    trainer._hybrid_parallel_ready=True
    return trainer


def contract():
    from types import SimpleNamespace
    from tools.ppo_observation_contract import requested_contract
    return requested_contract(SimpleNamespace(framing_profile='canvas_v1',detector_mode='off'))


def inspect_spawn(checkpoint,connection):
    """CPU/read-only preflight target, also used to verify Windows spawn behavior."""
    try:
        trainer=install()
        from tools.ppo_observation_contract import validate_checkpoint
        validate_checkpoint(checkpoint,contract())
        # Exercise the worker's second headless install without stacking adapters.
        from tools.ppo_headless_env import install as headless
        headless(trainer,'electron',profile=True,electron_transport='shared_memory')
        base=trainer.build_env_class()
        if sum(c.__name__=='HeadlessEnv' for c in base.__mro__)!=1:
            raise AssertionError('Duplicate/missing headless adapter')
        a=trainer.parse_args(['--mode','train','--run-name','offline_worker_check',
            '--resolution','512x288','--rate-mode','uncapped','--framing-profile','canvas_v1','--detector-mode','off'])
        assert a.resolution==(384,384) and a.rate_mode=='capped' and a.fps==30
        connection.send({'status':'PASS','contract':contract(),'rate_mode':a.rate_mode,'fps':a.fps})
    except BaseException as exc:connection.send({'status':'FAIL','error':repr(exc)})
    finally:connection.close()
