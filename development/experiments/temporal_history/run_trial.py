"""Isolated single-worker scratch training/evaluation. Launch only explicitly."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from experiment_config import ROOT, configuration, source_hashes, file_hash
sys.path.insert(0,str(ROOT))


def install_rate_bridge(trainer=None):
    """Keep experimental rates out of the inherited integer-only 2..30 CLI."""
    if trainer is None:
        import train_rs_v3_ppo as trainer
    if getattr(trainer.v2_args,'_temporal_rate_bridge',False) is True:return
    original=trainer.v2_args
    def bridged(a):
        import argparse
        import math
        rate=float(a.fps)
        if not math.isfinite(rate) or not 1 <= rate <= 60:
            raise ValueError('Experimental decision rate must be 1..60 Hz')
        legacy=argparse.Namespace(**vars(a))
        legacy.fps=30  # Parse unchanged legacy defaults/validation first.
        cfg=original(legacy)
        cfg.fps=rate  # Runtime rate is restored without rounding/clamping.
        return cfg
    bridged._temporal_rate_bridge=True
    trainer.v2_args=bridged


def trainer_args(trainer, options):
    common = ['--mode',options.mode,'--run-name',options.output.name,
        '--resolution','512x288','--framing-profile','canvas_v1','--canvas-policy-input',
        '--background-display','1','--detector-mode','off',
        '--reuse-observation-guard','--press-delay-ms','5','--fps','30','--rate-mode','capped',
        '--state-model',str(options.state_model),'--episode-max-seconds','300',
        '--max-minutes',str(options.minutes),'--seed',str(options.seed),'--torch-threads','1',
        '--steps','12000000','--n-steps','128','--batch-size','32','--n-epochs','5',
        '--learning-rate','0.0003','--gamma','0.995','--ent-coef','0.002']
    if options.mode == 'train':
        common += ['--pause-ppo-optimizer','--resync-rollouts','--settled-learning',
                   '--audit-reward-timing','--save-every-minutes','5']
    else:
        common += ['--resume',str(options.checkpoint),'--eval-policy','deterministic',
                   '--eval-episodes',str(options.episodes)]
    a = trainer.parse_args(common)
    # Experimental validator allows up to 60 Hz. No production CLI change.
    a.fps = options.decision_hz
    return a


def train(trainer,a,output,contract,minutes):
    from tools.ppo_settled_learning import SettledPPO,audit_settled_ledger
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    from tools.ppo_observation_contract import attach_contract
    from tools.ppo_run_integrity import atomic_json,finalize_run
    from tools.ppo_reaction_metrics import warm_policy
    from tools.ppo_frame_state_cache import install_frame_state_cache

    monitored=raw=model=callback=None
    report=dict(status='RUNNING',initialization='scratch',requested_minutes=minutes)
    started=time.perf_counter()
    error=None
    class TrialPPO(SettledPPO):
        def train(self):
            self._trial_update_active=True
            super().train()
            self._trial_update_active=False
    try:
        monitored,raw=trainer.create_live_env(a,output)
        install_frame_state_cache(raw)
        model=trainer.make_model(monitored,a)
        attach_contract(model,contract)
        model.__class__=TrialPPO
        model._trial_update_active=False
        warm_policy(model.policy,model.observation_space)
        model.configure_settled(raw,output,.05,minutes)
        save_atomic_checkpoint(model,output/'latest.zip')
        with (output/'initial.zip').open('xb') as handle:
            handle.write((output/'latest.zip').read_bytes())
        callback=trainer.build_callback(a,output,raw=raw)
        original_start=callback._on_training_start
        def budget_start():
            report['budget_started_perf']=time.perf_counter()
            model._settled_deadline=report['budget_started_perf']+minutes*60
            original_start()
        callback._on_training_start=budget_start
        model.learn(total_timesteps=a.steps,callback=callback,progress_bar=False)
        report['training_seconds']=time.perf_counter()-report['budget_started_perf']
        stats=dict(model._settled_stats)
        report['learning_transitions']=stats
        report['audit']=audit_settled_ledger(output/'learning_transitions.jsonl')
        report['status']=('COMPLETE' if stats['stop_reason']=='wall_time_at_optimized_rollout_boundary'
            and report['audit']['status']=='PASS_SETTLED_LEARNING' and not raw._guard_hard_stopped else 'STOPPED')
    except BaseException as exc:
        error=exc
        report.update(status='STOPPED',error=repr(exc))
        traceback.print_exc()
    finally:
        report['wall_seconds']=time.perf_counter()-started
        if model is not None and hasattr(model,'_settled_stats'):
            report['learning_transitions']=dict(model._settled_stats)
            report['optimized_transitions']=model._settled_stats['optimized_transitions']
            report['unoptimized_collected_transitions']=max(0,
                model._settled_stats['collected_transitions']-model._settled_stats['optimized_transitions'])
            report['optimized_per_second']=report['optimized_transitions']/max(.001,report.get('training_seconds',report['wall_seconds']))
        def save():
            if model is not None and hasattr(model,'_settled_stats') and not model._trial_update_active:
                save_atomic_checkpoint(model,output/'latest.zip')
            elif model is not None:
                report['checkpoint_note']='Setup or update interrupted; retained any last complete checkpoint'
        def stats_report():
            if raw is not None:
                value=trainer.build_report(a,raw,steps=raw.training_steps,started=started,callback=callback)
                atomic_json(output/'v61_report.json',value)
                from tools.ppo_reaction_metrics import latency_summary
                report['reaction_latency']=latency_summary(list(getattr(raw,'_reaction_samples',[])))
                report['guard_hard_stopped']=bool(getattr(raw,'_guard_hard_stopped',False))
        cleanup=[('statistics',stats_report)]
        if callback is not None:
            for name in ('log_handle','reward_step_audit_handle'):
                handle=getattr(callback,name,None)
                if handle is not None:cleanup.append((name,handle.close))
        if monitored is not None:cleanup.append(('environment',monitored.close))
        elif raw is not None:cleanup.append(('environment',raw.close))
        finalize_run(report,save,cleanup,lambda:atomic_json(output/'trial_report.json',report))
    return report['status']=='COMPLETE' and error is None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['train','eval'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--span-ms',type=float,default=500)
    p.add_argument('--capture-hz',type=float,default=60)
    p.add_argument('--decision-hz',type=float,default=30)
    p.add_argument('--minutes',type=float,default=30)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--state-model',type=Path,default=ROOT/'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt')
    p.add_argument('--expected-sources',type=Path)
    options=p.parse_args()
    config=configuration(options.frames,options.span_ms,options.capture_hz,options.decision_hz)
    if not 0 < options.minutes <= 2880 or options.episodes < 1 or not 0 <= options.seed < 2**32:
        p.error('Invalid budget, seed or evaluation count')
    if options.mode=='train' and options.checkpoint is not None:p.error('Scratch experiment does not accept checkpoint initialization')
    if options.mode=='eval' and (options.checkpoint is None or not options.checkpoint.is_file()):p.error('Evaluation requires a checkpoint')
    if not options.state_model.is_file():p.error('Missing screen-state model')
    hashes=source_hashes()
    if options.expected_sources and json.loads(options.expected_sources.read_text())!=hashes:
        p.error('Source changed since batch planning; refusing mixed-source comparison')
    options.output=options.output.resolve()
    if not options.output.is_relative_to(ROOT/'reports'):
        p.error('Trial output must be a new directory under reports')
    if options.output.exists():p.error('Output exists; choose a new directory')
    import os
    os.chdir(ROOT)
    import cv2
    cv2.setNumThreads(1)
    import train_rs_v612_ppo as trainer
    install_rate_bridge()
    from live_adapter import install
    install(trainer,config)
    a=trainer_args(trainer,options)
    from tools.ppo_observation_contract import requested_contract,validate_checkpoint
    from tools.ppo_run_integrity import atomic_json
    contract=requested_contract(a)
    if options.mode=='eval':validate_checkpoint(options.checkpoint,contract)
    # Evaluate uses its existing output/report implementation, redirected locally.
    import train_rs_v2_verify as v2
    v2.RUN_ROOT=options.output.parent
    options.output.parent.mkdir(parents=True,exist_ok=True)
    if options.mode=='train':options.output.mkdir()
    source_checkpoint_hash=file_hash(options.checkpoint) if options.checkpoint else None
    if options.mode=='train':
        atomic_json(options.output/'trial_config.json',dict(config=config,seed=options.seed,contract=contract,
            settings=vars(a),initialization='scratch',source_hashes=hashes))
        success=train(trainer,a,options.output,contract,options.minutes)
    else:
        success=trainer.evaluate(a)
        atomic_json(options.output/'trial_config.json',dict(config=config,seed=options.seed,contract=contract,
            source_hashes=hashes,checkpoint_sha256=source_checkpoint_hash))
        if file_hash(options.checkpoint)!=source_checkpoint_hash:raise RuntimeError('Evaluation checkpoint changed')
    if hashes!=source_hashes():raise RuntimeError('Sources changed while trial was running')
    raise SystemExit(0 if success else 1)


if __name__=='__main__':main()
