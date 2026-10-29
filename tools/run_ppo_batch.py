"""Sequential local PPO stages in one persistent, foreground browser window.

Manifest: {"stages": [{"args": ["--mode", "eval", ...]}]}
The args are normal train_rs_v612_ppo flags. No shell commands are accepted.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import re

from tools.ppo_browser_session import PersistentBrowserSession


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def manifest_runs(manifest):
    stages = manifest.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('At least one stage is required')
    names = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError('Each stage must be an object')
        args = stage.get('args')
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise ValueError('Each stage needs an argument list')
        for flag in ('--mode', '--run-name'):
            if args.count(flag) != 1 or args.index(flag)+1 >= len(args):
                raise ValueError(f'Each stage requires exactly one {flag}')
        if args[args.index('--mode')+1] not in ('train', 'eval'):
            raise ValueError('Only train/eval stages are supported')
        if stage.get('reaction_audit') and args[args.index('--mode')+1] != 'eval':
            raise ValueError('Reaction instrumentation is evaluation-only')
        name = args[args.index('--run-name')+1]
        if Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('Run names must be single directory names')
        if name in names:
            raise ValueError('Stage run names must be unique')
        names.append(name)
    return names


def with_run_prefix(manifest, prefix):
    names = manifest_runs(manifest)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', prefix):
        raise ValueError('Run prefix must be 1-40 letters, digits, underscores or hyphens')
    result = copy.deepcopy(manifest)
    mapping = {}
    for index, (stage, original) in enumerate(zip(result['stages'], names), 1):
        args = stage['args']
        mode = args[args.index('--mode')+1]
        mapping[original] = f'{prefix}_{index:02d}_{mode}'
        args[args.index('--run-name')+1] = mapping[original]
    root = Path('ai_data/rl_v2_runs').resolve()
    for stage in result['stages']:
        args = stage['args']
        if '--resume' in args:
            at = args.index('--resume')+1
            source = Path(args[at])
            if source.parent.name in mapping and source.parent.parent.resolve() == root:
                args[at] = str(source.parent.with_name(mapping[source.parent.name])/source.name)
    return result


def main():
    import train_rs_v612_ppo as trainer
    from tools.ppo_reward_timing_audit import audit_training_log, audit_reward_steps, read_training_log

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run-prefix', default=None,
                        help='Give all stages fresh names and remap their inter-stage checkpoint references')
    options = parser.parse_args()
    if options.report.exists():
        raise ValueError('Report path exists; choose a new output directory')
    manifest = json.loads(options.manifest.read_text())
    if options.run_prefix:
        manifest = with_run_prefix(manifest, options.run_prefix)
    names = manifest_runs(manifest)
    for name in names:
        if (Path('ai_data/rl_v2_runs')/name).exists():
            raise ValueError(f'Stage output already exists: {name}')
    options.report.mkdir(parents=True)
    state = {'manifest': manifest, 'completed': [], 'status': 'RUNNING'}
    source_hashes = {}
    session = PersistentBrowserSession()

    def save():
        state['browser_session'] = session.summary()
        (options.report/'batch.json').write_text(json.dumps(state, indent=2, default=str)+'\n')

    save()
    try:
        for stage in manifest['stages']:
            args = trainer.parse_args(stage['args'])
            if args.resume:
                source_hashes[args.resume] = hashlib.sha256(args.resume.read_bytes()).hexdigest()
            started = time.perf_counter()
            log_path = options.report/(args.run_name+'.log')
            with log_path.open('w', encoding='utf8') as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    print('[BATCH STAGE]', args.run_name, args.mode, flush=True)
                    if args.mode == 'train':
                        trainer.train(args, browser_session=session)
                    elif stage.get('reaction_audit', False):
                        if args.eval_policy != 'deterministic' or args.stochastic_eval or args.resume is None:
                            raise ValueError('Reaction audit requires a deterministic checkpoint evaluation')
                        from tools.audit_ppo_reaction_latency import Audit, instrument_evaluation, save_audit
                        audit = Audit()
                        try:
                            with instrument_evaluation(trainer, audit):
                                trainer.evaluate(args, browser_session=session)
                        finally:
                            dest = Path('ai_data/rl_v2_runs')/args.run_name
                            if dest.exists():
                                save_audit(audit, dest, source_hashes[args.resume],
                                    hashlib.sha256(args.resume.read_bytes()).hexdigest() == source_hashes[args.resume])
                    else:
                        trainer.evaluate(args, browser_session=session)
            directory = Path('ai_data/rl_v2_runs')/args.run_name
            report_path = directory/('v61_report.json' if args.mode == 'train' else 'v61_eval_report.json')
            if not report_path.exists():
                raise RuntimeError(f'{args.run_name}: no completed report; inspect startup/focus log')
            report = json.loads(report_path.read_text())
            if report['detector']['v612_live_input_guard']['guard_hard_stopped']:
                raise RuntimeError(f'{args.run_name}: guard stopped; subsequent stages blocked')
            if args.mode == 'train':
                if not report['completed_requested_steps']:
                    raise RuntimeError(f'{args.run_name}: incomplete training; subsequent stages blocked')
                if args.audit_reward_timing:
                    effective_path = directory/'v61_effective_ppo_settings.json'
                    rollout = (json.loads(effective_path.read_text())['effective_settings']['n_steps']
                               if effective_path.exists() else args.n_steps)
                    stability = audit_training_log(read_training_log(log_path),
                        min_resumes=max(1, args.steps//rollout-1), min_steps=args.steps)
                    replay = audit_reward_steps((directory/'ppo_reward_step_audit.jsonl').read_text(),
                                               minimum_steps=args.steps)
                    audits = {'stability': stability, 'reward_replay': replay}
                    (options.report/(args.run_name+'_audits.json')).write_text(json.dumps(audits, indent=2)+'\n')
                    if stability['status'] != 'PASS_STABILITY_CANDIDATE' or replay['status'] != 'PASS_REWARD_REPLAY':
                        raise RuntimeError(f'{args.run_name}: audit failed; subsequent stages blocked')
            elif report['eval_episodes'] != args.eval_episodes:
                raise RuntimeError(f'{args.run_name}: incomplete evaluation; subsequent stages blocked')
            state['completed'].append({'run_name': args.run_name, 'mode': args.mode,
                'wall_seconds': time.perf_counter()-started, 'report': str(report_path)})
            save()
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state['status'] = 'COMPLETE'
    except BaseException as exc:
        state['status'], state['error'] = 'STOPPED', repr(exc)
        raise
    finally:
        session.close()
        state['source_checkpoints_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == digest
                                                   for p, digest in source_hashes.items())
        save()
        if not state['source_checkpoints_unchanged']:
            raise RuntimeError('A source checkpoint changed')


if __name__ == '__main__':
    main()
