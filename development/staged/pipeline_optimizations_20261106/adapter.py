"""Explicit process-local experimental install; never imported by production."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def configuration():
    features = set(filter(None, os.environ.get('RS_PIPELINE_EXPERIMENT', '').split(',')))
    if not features <= {'pack', 'metrics', 'pinned', 'gpu_cache'}:
        raise ValueError('Unknown pipeline optimization')
    if {'pinned', 'gpu_cache'} <= features:
        raise ValueError('Choose only one transfer experiment')
    return dict(features=sorted(features), rate_wait=os.environ.get('RS_RATE_WAIT'),gpu_limit_mib=1024, reserve_mib=2048,
                pinned_limit_mib=96, experimental=True)


def verify_sources():
    baseline = json.loads((HERE/'source_baseline.json').read_text())
    for relative, expected in baseline.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed; revalidate staged optimization: '+relative)


def install():
    config = configuration()
    verify_sources()
    import command_adapter
    trainer = command_adapter.install()
    if getattr(trainer, '_pipeline_experiment_installed', False):
        return
    features = set(config['features'])
    if config['rate_wait']:
        from development.experiments.latency_tail_20261107.pacing import install as install_pacing
        install_pacing(trainer,config['rate_wait'])
    if 'pack' in features:
        from development.staged.pipeline_optimizations_20261106.observation import install as install_pack
        install_pack(trainer)
    from tools.ppo_settled_learning import SettledPPO
    original_train = SettledPPO.train

    def train(model):
        if not hasattr(model, '_settled_stats'):
            return original_train(model)
        # Early-stop semantics require immediate KL. Keep that path untouched.
        if model.target_kl is not None:
            return original_train(model)
        from development.staged.pipeline_optimizations_20261106.optimizer import deferred_train
        from development.staged.pipeline_optimizations_20261106.transfers import UpdateTransfers
        original_buffer = model.rollout_buffer
        transfer = None
        try:
            mode = next(iter(features & {'pinned', 'gpu_cache'}), None)
            if mode:
                transfer = UpdateTransfers(original_buffer, mode, model.batch_size,
                    gpu_limit_mib=config['gpu_limit_mib'], reserve_mib=config['reserve_mib'],
                    pinned_limit_mib=config['pinned_limit_mib'])
                model.rollout_buffer = transfer
            result = (deferred_train if 'metrics' in features else original_train)(model)
            # One small record per update, outside action dispatch.
            if hasattr(model, '_settled_output'):
                with (Path(model._settled_output)/'pipeline_experiment.jsonl').open('a') as f:
                    f.write(json.dumps(dict(config, transfer_mode=transfer.mode if transfer else 'standard',
                        transfer_reason=transfer.reason if transfer else None))+'\n')
            return result
        finally:
            model.rollout_buffer = original_buffer
            if transfer is not None:
                transfer.close()

    if features & {'metrics', 'pinned', 'gpu_cache'}:
        SettledPPO.train = train
    original_sources = command_adapter.sources

    def sources():
        result = original_sources()
        for path in [*HERE.glob('*.py'), HERE/'source_baseline.json']:
            result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        path=ROOT/'development/experiments/latency_tail_20261107/pacing.py'
        result[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    command_adapter.sources = sources
    # run_training imports this symbol before our explicit installation.
    import sys
    if 'run_training' in sys.modules:
        sys.modules['run_training'].sources = sources
    trainer._pipeline_experiment_installed = True
    print('[PIPELINE EXPERIMENT] '+json.dumps(config), flush=True)
