"""Original visible MSS path, frozen policy: capture/command timing only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types


class TimedGrabber:
    def __init__(self,delegate):
        self.delegate=delegate
        self.timing={}
        self.durations=[]

    def __getattr__(self,name):
        return getattr(self.delegate,name)

    def grab(self,region):
        began=time.perf_counter()
        image=self.delegate.grab(region)
        ended=time.perf_counter()
        self.timing=dict(request_started_at=began,image_received_at=ended)
        self.durations.append((ended-began)*1000)
        return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=30,choices=range(10,61))
    args=parser.parse_args()
    args.report.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    import train_rs_v612_ppo as trainer
    from tools.ppo_reaction_metrics import warm_policy,command_latency,latency_summary
    source_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    settings=trainer.parse_args(['--mode','eval','--resume',str(args.checkpoint),
        '--framing-profile','legacy','--reuse-observation-guard','--capture-mode','fast',
        '--fps','30','--rate-mode','uncapped','--press-delay-ms','5','--startup-timeout','20',
        '--state-model','checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt',
        '--detector','ai_data/v5_hazards/runs/segment_round04/weights/best.pt'])
    monitored=raw=None
    samples=[]
    result=dict(status='STARTING',model_updated=False,framing_profile='legacy',source_sha256=source_hash,
                seconds=args.seconds,steps=0,episodes=0)
    try:
        monitored,raw=trainer.create_live_env(settings,args.report)
        grabber=TimedGrabber(raw.grabber)
        raw.grabber=grabber
        capture=raw._capture
        async def timed_capture(self):
            data=await capture()
            self._headless_capture_timing=dict(grabber.timing)  # Same timestamp schema; MSS source.
            return data
        raw._capture=types.MethodType(timed_capture,raw)
        step_browser=raw._step_browser
        depth=[0]
        async def timed_step(self,action):
            policy=dict(getattr(self,'_learning_observation_capture',{}) or {})
            before=self.step_timings[-1] if self.step_timings else None
            outer=depth[0]==0
            depth[0]+=1
            try:
                return await step_browser(action)
            finally:
                depth[0]-=1
                after=self.step_timings[-1] if self.step_timings else None
                if outer and after is not None and after is not before:
                    sample=command_latency(policy,getattr(self,'_reaction_guard_capture',{}),
                        after.get('press_timestamp'),after.get('press_roundtrip_ms'))
                    if sample:
                        samples.append(sample)
        raw._step_browser=types.MethodType(timed_step,raw)
        model=PPO.load(str(args.checkpoint),env=monitored,device='cuda' if torch.cuda.is_available() else 'cpu')
        warm_policy(model.policy,model.observation_space)
        obs,_=monitored.reset()
        grabber.durations.clear()
        started=time.perf_counter()
        print('VISIBLE MEASUREMENT READY: keep Chromium focused and unobstructed for 30 seconds.',flush=True)
        while time.perf_counter()-started < args.seconds:
            action,_=model.predict(obs,deterministic=False)
            obs,_,done,truncated,_=monitored.step(int(action))
            result['steps']+=1
            if done or truncated:
                result['episodes']+=1
                if time.perf_counter()-started < args.seconds:
                    obs,_=monitored.reset()
        result.update(status='COMPLETE',wall_seconds=time.perf_counter()-started)
    except Exception as exc:
        result.update(status='STOPPED',error=repr(exc))
        raise
    finally:
        result['reaction_latency']=latency_summary(samples)
        if raw is not None:
            values=getattr(raw.grabber,'durations',[])
            if values:
                result['mss_capture_ms']=dict(count=len(values),median_ms=float(np.median(values)),
                    p95_ms=float(np.percentile(values,95)),max_ms=float(max(values)))
            result['guard_hard_stopped']=raw._guard_hard_stopped
        if monitored is not None:
            monitored.close()
        elif raw is not None:
            raw.close()
        result['source_unchanged']=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==source_hash
        (args.report/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
