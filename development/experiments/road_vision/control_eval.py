"""Frozen four-frame policy versus constant actions, with the same model forward."""
import argparse
import json
from pathlib import Path
import time
from road_config import ROOT,PRESETS,configuration,source_hashes,digest,write_json

DEFAULT_CHECKPOINT=ROOT/'reports/temporal_batch_20261101_180312_752232/wide4_60_seed123/train/latest.zip'
CONTROLS={'none':0,'jump':3,'roll':4}

def override_prediction(original,control):
    def predict(model,observation,**kwargs):
        action,probs=original(model,observation,**kwargs)
        # Keep the original forward/copy workload even for a fixed-action control.
        # Do not report the learned distribution as the constant controller's.
        return (action,probs) if control=='learned' else (CONTROLS[control],None)
    return predict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',choices=['learned',*CONTROLS],required=True)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--minutes',type=float,default=8)
    p.add_argument('--crop-evidence',action='store_true')
    o=p.parse_args()
    o.output=o.output.resolve()
    if not o.output.is_relative_to(ROOT/'reports') or o.output.exists():p.error('Use a new reports directory')
    if not 1<=o.episodes<=100 or not 0<o.minutes<=15:p.error('Use 1..100 episodes and <=15 minutes')
    import os
    os.chdir(ROOT)
    import cv2
    cv2.setNumThreads(1)
    import train_rs_v612_ppo as trainer
    from run_trial import trainer_args,install_rate_bridge
    from live_adapter import install
    from experiment_config import configuration as temporal_configuration
    from menu_confirmation import install as install_confirmation
    install_confirmation();install_rate_bridge();install(trainer,temporal_configuration(4,500,60,30))
    o.mode='eval';o.seed=123;o.decision_hz=30
    o.state_model=ROOT/'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt'
    a=trainer_args(trainer,o)
    from tools.ppo_observation_contract import requested_contract,validate_checkpoint
    validate_checkpoint(o.checkpoint,requested_contract(a))
    before=digest(o.checkpoint);hashes=source_hashes()
    import train_rs_v2_verify as v2
    v2.RUN_ROOT=o.output.parent
    import tools.ppo_eval_prediction as prediction
    prediction.predict_with_optional_probabilities=override_prediction(prediction.predict_with_optional_probabilities,o.control)
    original_create=trainer.create_live_env
    evidence_frames=[]
    def create(*args,**kwargs):
        monitored,raw=original_create(*args,**kwargs)
        if not o.crop_evidence:return monitored,raw
        step=monitored.step;last=0.;saved=0
        def evidence(action):
            nonlocal last,saved
            result=step(action);now=time.perf_counter()
            native=getattr(raw,'last_source_native',None)
            if saved<12 and now-last>=.75 and native is not None:
                # Bounded copies of existing frames; PNG encoding happens after close.
                evidence_frames.append((dict(index=saved,episode=raw.episode_number,action=int(action),
                    step=raw.training_steps,terminal=bool(result[2] or result[3]),review_required=True),native.copy()))
                saved+=1;last=time.perf_counter()
            return result
        monitored.step=evidence
        return monitored,raw
    trainer.create_live_env=create
    success=trainer.evaluate(a)
    if evidence_frames:
        from road_crop import crop_rgb,preview
        folder=o.output/'crop_evidence';folder.mkdir(exist_ok=True)
        with (folder/'frames.jsonl').open('x') as f:
            for record,native in evidence_frames:
                index=record['index'];f.write(json.dumps(record)+'\n')
                if not cv2.imwrite(str(folder/f'{index:02d}_native.png'),cv2.cvtColor(native,cv2.COLOR_RGB2BGR)):
                    raise OSError('Could not save native crop evidence')
                for arm in PRESETS:
                    c=configuration(arm)
                    preview(native,c,folder/f'{index:02d}_{arm}_bounds.png')
                    pixels=crop_rgb(native,c).transpose(1,2,0)
                    if not cv2.imwrite(str(folder/f'{index:02d}_{arm}_input.png'),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)):
                        raise OSError('Could not save policy crop evidence')
    report_path=o.output/'v61_eval_report.json'
    report=json.loads(report_path.read_text())
    report.update(control=o.control,inference_workload_preserved=True,finalization_errors=[],
        crop_evidence_io=o.crop_evidence,performance_benchmark=False,
        checkpoint_unchanged=digest(o.checkpoint)==before,
        source_unchanged=hashes==source_hashes(),menu_confirmation='bounded_extra_evidence_v1')
    write_json(report_path,report)
    write_json(o.output/'trial_config.json',dict(control=o.control,checkpoint=str(o.checkpoint),
        checkpoint_sha256=before,source_hashes=hashes,temporal=temporal_configuration(4,500,60,30)))
    if not report['checkpoint_unchanged'] or not report['source_unchanged']:raise RuntimeError('Inputs changed')
    raise SystemExit(0 if success else 1)

if __name__=='__main__':main()
