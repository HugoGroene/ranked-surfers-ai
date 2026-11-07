"""Measured command-history runtime defaults; checkpoint contract unchanged."""
import json
import os
from pathlib import Path


def settings():
    # Explicit staged experiments own their switches, including a true baseline.
    enabled='RS_PIPELINE_EXPERIMENT' not in os.environ and os.environ.get('RS_PIPELINE_FAST_PATH','1')=='1'
    return dict(enabled=enabled,pack=enabled,pinned=enabled,resync_spacing=enabled,
                rate_wait=os.environ.get('RS_RATE_WAIT','precise') if enabled else None,
                pinned_limit_mib=96)


def install(trainer):
    config=settings()
    if not config['enabled'] or getattr(trainer,'_measured_fast_path',False):return
    from development.experiments.latency_tail_20261107.pacing import install as install_pacing
    from development.staged.pipeline_optimizations_20261106.observation import install as install_pack
    from development.staged.pipeline_optimizations_20261106.transfers import UpdateTransfers
    from development.experiments.latency_tail_20261107.adam_steps import restore_cpu_steps
    from development.experiments.latency_tail_20261107.resync_spacing import install as install_resync_spacing
    from tools.ppo_settled_learning import SettledPPO
    install_pacing(trainer,config['rate_wait']);install_pack(trainer)
    install_resync_spacing(trainer)
    original_train=SettledPPO.train
    def train(model):
        buffer=model.rollout_buffer
        if not hasattr(model,'_settled_stats') or buffer.n_envs!=1 or not buffer.full or buffer.generator_ready:
            return original_train(model)
        counters=restore_cpu_steps(model.policy.optimizer)
        if counters['moved']:
            print('[ADAM CPU STEPS] '+json.dumps(counters),flush=True)
        transfer=None
        try:
            transfer=UpdateTransfers(buffer,'pinned',model.batch_size,pinned_limit_mib=96)
            model.rollout_buffer=transfer
            result=original_train(model)
            if hasattr(model,'_settled_output'):
                with (Path(model._settled_output)/'pipeline_runtime.jsonl').open('a') as f:
                    f.write(json.dumps(dict(config,transfer_mode=transfer.mode,transfer_reason=transfer.reason,
                                            adam_cpu_steps=counters))+'\n')
            return result
        finally:
            model.rollout_buffer=buffer
            if transfer is not None:transfer.close()
    SettledPPO.train=train
    trainer._measured_fast_path=True
    print('[PIPELINE RUNTIME] '+json.dumps(config),flush=True)
