"""Isolated short real-PPO stability probe. No migration of existing checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ROAD=ROOT/'development/experiments/road_vision'
sys.path[:0]=[str(ROOT),str(ROAD)]
import road_config
from hybrid import HybridHistory,LAYOUTS

def sources():
    result=road_config.source_hashes()
    for file in HERE.glob('*.py'):result[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',choices=list(LAYOUTS),required=True)
    p.add_argument('--context-ms',type=float,nargs='+',help='Optional increasing context ages; changes checkpoint contract')
    p.add_argument('--size',type=int,choices=[256,384],required=True)
    p.add_argument('--mode',choices=['train','eval'],default='train')
    p.add_argument('--minutes',type=float,default=3)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--expected-sources',type=Path)
    p.add_argument('--input-previews',action='store_true',help='Smoke only: retain six real actor observations, export after closing game')
    a=p.parse_args()
    if a.expected_sources and json.loads(a.expected_sources.read_text())!=sources():p.error('Sources changed')
    recent,old=LAYOUTS[a.layout]
    if a.context_ms is not None:old=tuple(t/1000 for t in a.context_ms)
    HybridHistory(recent,old)  # Validate before launching any browser.
    frames=recent+len(old)
    if frames not in (4,5,6,7,8):p.error('Probe supports four through eight total frames')
    if a.mode=='eval' and not a.checkpoint:p.error('Evaluation requires a matching checkpoint')
    if a.mode=='train' and a.checkpoint:p.error('Fresh probe only')
    import run_road_trial as trial
    import history_spacing
    original_config=trial.configuration
    def config(arm,count=4,span_ms=500,capture_hz=60,decision_hz=30,spacing='uniform'):
        if count!=frames:raise ValueError('Requested configuration frame count differs from layout')
        # Legacy experiment validation skipped five frames. Validate its other
        # fields through a supported shape; only this opt-in adapter supplies five.
        c=original_config(arm,4 if count==5 else count,span_ms,capture_hz,decision_hz,spacing)
        c['temporal'].update(frames=frames,spacing='hybrid',layout=a.layout,recent_frames=recent,
            context_offsets_seconds=list(old),padding='zero_with_age_and_validity',
            history_schema='recent_plus_context_v1')
        return c
    trial.configuration=config;trial.source_hashes=sources
    original_history=history_spacing.history_class
    def factory(pattern):
        if pattern!='hybrid':return original_history(pattern)
        class SelectedHistory(HybridHistory):
            def __init__(self,count,span):
                if count!=frames:raise ValueError('History channel count mismatch')
                super().__init__(recent,old)
        return SelectedHistory
    history_spacing.history_class=factory
    # Decorate environment finalization to preserve history-availability evidence.
    import road_adapter
    from input_evidence import InputEvidence
    evidence=InputEvidence(frames,max(old,default=1.)+.06) if a.input_previews else None
    original_install=road_adapter.install
    def install(trainer,config,**kwargs):
        original_install(trainer,config,**kwargs)
        base_build=trainer.build_env_class
        def build():
            base=base_build()
            class ProbeEnv(base):
                async def _close_browser(self):
                    try:
                        await super()._close_browser()
                    finally:
                        (self.output_dir/'hybrid_history.json').write_text(json.dumps(self.temporal.report(),indent=2))
            return ProbeEnv
        trainer.build_env_class=build
        if evidence is not None:
            original_callback=trainer.build_callback
            def callback(*args,**kwargs):
                result=original_callback(*args,**kwargs)
                end=result._on_rollout_end
                def on_end():
                    end()  # Existing guarded pause happens before copying evidence.
                    evidence.collect(result.model.rollout_buffer)
                result._on_rollout_end=on_end
                return result
            trainer.build_callback=callback
    road_adapter.install=install
    # Resource sampling starts only once the child has created its output folder.
    # It follows this process and descendants, avoiding other desktop processes.
    import run_trial
    original_train=run_trial.train
    def train(*args,**kwargs):
        from tools.ppo_resource_sampler import ResourceSampler
        sampler=ResourceSampler(Path(args[2])/'system_usage.jsonl',[os.getpid()])
        sampler.start()
        try:return original_train(*args,**kwargs)
        finally:
            sampler.close()
            if evidence is not None:evidence.write(Path(args[2])/'policy_inputs')
    run_trial.train=train
    argv=['run_probe','--mode',a.mode,'--arm',f'road{a.size}_spatial','--frames',str(frames),
          '--span-ms',str(1000*max(old,default=1.)),'--spacing','uniform',
          '--capture-hz','60','--decision-hz','30','--capture-backend','gpu_100',
          '--seed',str(a.seed),'--minutes',str(a.minutes),'--episodes',str(a.episodes),
          '--output',str(a.output)]
    if a.checkpoint:argv+=['--checkpoint',str(a.checkpoint)]
    sys.argv=argv
    trial.main()

if __name__=='__main__':main()
