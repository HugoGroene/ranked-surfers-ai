"""Fresh road-crop PPO, optional end-to-end BC initialization, matching evaluation."""
import argparse
import json
from pathlib import Path
import time
from road_config import ROOT,PRESETS,configuration,source_hashes,digest,write_json

def args_for(trainer,options,config):
    from run_trial import trainer_args
    a=trainer_args(trainer,options)
    # Parse unchanged production arguments first; opt-in process-local contract
    # explicitly replaces policy geometry only. No production CLI relaxation.
    a.resolution=tuple(config['size']);a.resize_method='area'
    return a

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['train','eval'],required=True)
    p.add_argument('--arm',choices=list(PRESETS),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--span-ms',type=float,default=500)
    p.add_argument('--spacing',choices=['consecutive','uniform','golden','logarithmic'],default='uniform')
    p.add_argument('--capture-hz',type=float,default=60)
    p.add_argument('--decision-hz',type=float,default=30)
    p.add_argument('--minutes',type=float,default=30)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--state-model',type=Path,default=ROOT/'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt')
    p.add_argument('--expected-sources',type=Path)
    p.add_argument('--crop-previews',action='store_true')
    p.add_argument('--capture-backend',choices=['native_bitmap','bitmap_100','gpu_100'],default='native_bitmap')
    p.add_argument('--bc-root',type=Path,help='Optional approved canvas_v1 demos; legacy recordings refused')
    p.add_argument('--allow-downsampled-replays',action='store_true',help='Acknowledge BC uses already-downsampled 512x288 recordings')
    p.add_argument('--bc-epochs',type=int,default=5)
    o=p.parse_args()
    if not 0<o.minutes<=2880 or o.episodes<1 or not 0<=o.seed<2**32:p.error('Invalid budgets/seed')
    if not 1<=o.bc_epochs<=30:p.error('BC epochs must be 1..30')
    if o.mode=='train' and o.checkpoint:p.error('Fresh initialization only; no silent checkpoint migration')
    if o.mode=='eval' and (not o.checkpoint or not o.checkpoint.is_file()):p.error('Evaluation needs checkpoint')
    if o.bc_root and o.mode!='train':p.error('BC runs only before training')
    if not o.state_model.is_file():p.error('Screen state model missing')
    config=configuration(o.arm,o.frames,o.span_ms,o.capture_hz,o.decision_hz,o.spacing)
    if o.bc_root and o.spacing!='uniform':p.error('Nonuniform/pack-history BC replay is not implemented; use scratch trials')
    o.output=o.output.resolve()
    if not o.output.is_relative_to(ROOT/'reports') or o.output.exists():p.error('Use a new output folder under reports')
    hashes=source_hashes()
    if o.expected_sources and json.loads(o.expected_sources.read_text())!=hashes:p.error('Source manifest mismatch')
    if o.bc_root:
        from road_bc import inspect_demos
        inspect_demos(o.bc_root,o.allow_downsampled_replays)  # Before creating a browser.
    import os
    os.chdir(ROOT)
    import cv2
    cv2.setNumThreads(1)
    import train_rs_v612_ppo as trainer
    from capture_backend import install_backend
    install_backend(o.capture_backend)
    from run_trial import install_rate_bridge,train
    from road_adapter import install
    install_rate_bridge();install(trainer,config,crop_previews=o.crop_previews)
    a=args_for(trainer,o,config)
    from tools.ppo_observation_contract import requested_contract,validate_checkpoint
    contract=requested_contract(a)
    source_hash=digest(o.checkpoint) if o.checkpoint else None
    if o.mode=='eval':validate_checkpoint(o.checkpoint,contract)
    import train_rs_v2_verify as v2
    v2.RUN_ROOT=o.output.parent
    o.output.parent.mkdir(parents=True,exist_ok=True)
    # BC completes BEFORE launching the live environment; its time is reported
    # separately from the equal PPO wall budget. The live model starts at t=0.
    bc_state=None
    stage_started=time.perf_counter()
    if o.mode=='train':
        o.output.mkdir()
        if o.bc_root:
            from road_bc import pretrain
            bc_state=pretrain(trainer,a,config,contract,o.bc_root,o.output/'bc',
                              o.bc_epochs,o.allow_downsampled_replays)
            make=trainer.make_model
            def initialized(env,args):
                model=make(env,args);model.policy.load_state_dict(bc_state);return model
            trainer.make_model=initialized
        write_json(o.output/'trial_config.json',dict(config=config,contract=contract,source_hashes=hashes,
            initialization='end_to_end_bc' if o.bc_root else 'scratch',seed=o.seed,settings=vars(a),capture_backend=o.capture_backend))
        success=train(trainer,a,o.output,contract,o.minutes)
        report_path=o.output/'trial_report.json'
        report=json.loads(report_path.read_text())
        report['initialization']='end_to_end_bc' if o.bc_root else 'scratch'
        report['bc_seconds']=(json.loads((o.output/'bc/bc_report.json').read_text())['wall_seconds'] if o.bc_root else 0.)
        report['total_stage_seconds']=time.perf_counter()-stage_started
        report['road_vision']=config
        report['capture_backend']=o.capture_backend
        write_json(report_path,report)
    else:
        success=trainer.evaluate(a)
        # evaluate returned only after its environment close completed.
        report_path=o.output/'v61_eval_report.json'
        report=json.loads(report_path.read_text());report['finalization_errors']=[]
        write_json(report_path,report)
        write_json(o.output/'trial_config.json',dict(config=config,contract=contract,source_hashes=hashes,
            checkpoint_sha256=source_hash,seed=o.seed,capture_backend=o.capture_backend))
        if digest(o.checkpoint)!=source_hash:raise RuntimeError('Source checkpoint changed')
    if source_hashes()!=hashes:raise RuntimeError('Source changed during trial')
    raise SystemExit(0 if success else 1)

if __name__=='__main__':main()
