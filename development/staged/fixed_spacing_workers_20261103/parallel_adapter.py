"""Spawn-safe opt-in fixed-spacing four-frame environment for shared PPO."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HYBRID=HERE.parent/'hybrid_history_20261102'
ROAD=ROOT/'development/experiments/road_vision'
sys.path[:0]=[str(ROOT),str(HYBRID),str(ROAD)]
import run_probe

LAYOUTS={'fixed4':(0.,.05,.25,.75)}
DEFAULT_CHECKPOINT=None  # Each layout gets its own fresh checkpoint; never migrate implicitly.


def selected():
    name=os.environ.get('RS_FIXED_LAYOUT','fixed4')
    if name not in LAYOUTS:raise ValueError('Unknown fixed spacing layout')
    return name,LAYOUTS[name]


def make_history(frames,span):
    from hybrid import HybridHistory
    _,ages=selected()
    if frames!=len(ages) or abs(span-ages[-1])>1e-9:raise ValueError('Wrong worker history contract')
    return HybridHistory(1,ages[1:])


def history_report(history):
    import numpy as np
    report=history.report()
    targets=[t*1000 for t in reversed(selected()[1])]
    report['requested_slot_ages_ms']=targets
    report['last_1000_target_lateness_ms']=[]
    for i,target in enumerate(targets):
        values=[row[i]-target for row in history.age_samples if row[i] is not None]
        report['last_1000_target_lateness_ms'].append(dict(count=len(values),
            median=float(np.median(values)) if values else None,
            p95=float(np.percentile(values,95)) if values else None,
            max=max(values) if values else None))
    return report


def sources():
    result=run_probe.sources()
    for file in HERE.glob('*.py'):
        result[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def configuration():
    from road_config import configuration as base
    name,ages=selected()
    c=base('road384_spatial',4,ages[-1]*1000,60,30)
    c['temporal'].update(frames=len(ages),spacing='hybrid',layout=name,recent_frames=1,
        context_offsets_seconds=list(ages[1:]),padding='zero_with_age_and_validity',
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
    import history_spacing
    original_history=history_spacing.history_class
    history_spacing.history_class=lambda pattern: make_history if pattern=='hybrid' else original_history(pattern)

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
        try:
            result=original_collect(self,*args,**kwargs)
            raw=getattr(self,'_settled_raw',None)
            if raw is not None and os.environ.get('RS_FIXED_INPUTS')=='1':
                from input_evidence import InputEvidence
                if not hasattr(raw,'_scaling_evidence'):
                    _,ages=selected()
                    raw._scaling_evidence=InputEvidence(len(ages),ages[-1]+.06)
                raw._scaling_evidence.collect(self.rollout_buffer)
            return result
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
                    (self.output_dir/'hybrid_history.json').write_text(json.dumps(history_report(self.temporal),indent=2))
                    (self.output_dir/'rollout_timings.json').write_text(json.dumps(getattr(self,'_scaling_rollouts',[]),indent=2))
                    (self.output_dir/'hybrid_close.json').write_text(json.dumps(closed,indent=2))
                    if hasattr(self,'_scaling_evidence'):
                        self._scaling_evidence.write(self.output_dir/'policy_inputs')
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
