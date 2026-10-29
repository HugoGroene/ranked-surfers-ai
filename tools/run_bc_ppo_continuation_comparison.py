"""Run the user-authorized local, pause-aware BC/PPO continuation comparison.

All source models remain read-only. Each stage has a unique run/output path.
Stops before subsequent stages if training, reward audits or evaluation fail.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/ppo_bc_continuation_ab_20261028'
STEPS = 4096
SOURCES = [
    ('pilot', 'ai_data/demonstrations_v13/bc_runs/bc_20261028_152607/model.zip',
     'ca99e521298e783eec6f5b11de11fa21eaa03631842801457e4ea3974643c5bc'),
    ('macro_bc', 'ai_data/demonstrations_v13/bc_runs/bc_control_macro_lr3e4_20261028/latest.zip', None),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')


def run_stage(name, args):
    command = [sys.executable, '-B', '-u', *args]
    write(REPORT/(name+'_command.json'), command)
    print('[EXPERIMENT START]', name, flush=True)
    started = time.perf_counter()
    with (REPORT/(name+'.log')).open('w', encoding='utf8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace')
        try:
            first_pause = first_resume = False
            for line in child.stdout:
                log.write(line)
                log.flush()
                show = any(token in line for token in ('[PPO FOCUS]', '[V612 PROGRESS]',
                    '[V612 REPORT]', '[PPO REWARD AUDIT]', 'Traceback', 'SAFETY STOP',
                    'UI safety stop', 'FAILED', 'Ran ', '\nOK'))
                if '[PPO PAUSE] Freeze verified' in line and not first_pause:
                    show = first_pause = True
                if '[PPO PAUSE] Resumed after optimizer' in line and not first_resume:
                    show = first_resume = True
                if '[V61 EVAL] episode ' in line:
                    episode = int(line.split('episode ', 1)[1].split('/', 1)[0])
                    show = episode % 10 == 0
                if show:
                    print(line.rstrip(), flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect its log')
    print(f'[EXPERIMENT END] {name} wall_seconds={time.perf_counter()-started:.1f}', flush=True)


def main():
    import torch
    from stable_baselines3 import PPO
    import train_rs_v3_ppo  # noqa: F401 - saved extractor class

    if REPORT.exists():
        raise ValueError('Experiment report directory exists; do not overwrite')
    for name, _, _ in SOURCES:
        for mode in ('train', 'eval'):
            if (ROOT/f'ai_data/rl_v2_runs/v612_ab_{name}_{mode}_20261028').exists():
                raise ValueError('Run directory already exists')
    REPORT.mkdir(parents=True)
    torch.set_num_threads(1)
    expected_macro = json.loads((ROOT/Path(SOURCES[1][1]).parent/'bc_report.json').read_text())['output_sha256']
    models = []
    source_info = []
    for name, file, expected in SOURCES:
        path = ROOT/file
        digest = sha(path)
        assert digest == (expected or expected_macro), name
        model = PPO.load(str(path), device='cpu')
        assert all(p.requires_grad for p in model.policy.parameters())
        settings = {k: getattr(model, k) for k in ('n_steps', 'batch_size', 'n_epochs',
                    'learning_rate', 'gamma', 'gae_lambda', 'ent_coef', 'seed')}
        settings['num_timesteps'] = model.num_timesteps
        settings['clip_range'] = float(model.clip_range(1.))
        models.append(model)
        source_info.append({'name': name, 'checkpoint': file, 'sha256': digest, 'settings': settings})
    assert source_info[0]['settings'] == source_info[1]['settings'], 'Loaded PPO settings differ'
    assert source_info[0]['settings']['n_steps'] == 128
    a, b = (m.policy.state_dict() for m in models)
    changed = [k for k in a if not torch.equal(a[k], b[k])]
    assert len(changed) == 4 and all(k.startswith(('mlp_extractor.policy_net.', 'action_net.')) for k in changed)
    del models, a, b
    status = {'steps_per_training_run': STEPS, 'eval_episodes_each': 100,
              'source_models': source_info, 'initial_differing_tensors': changed,
              'stage_order': ['pilot train', 'pilot eval', 'macro BC train', 'macro BC eval'],
              'limitations': 'One seed, sequential unpaired maps; a short diagnostic, not an overnight decision',
              'completed': []}
    write(REPORT/'experiment.json', status)
    run_stage('pause_runtime_tests', ['-m', 'unittest', 'discover', '-s', 'tests',
                                     '-p', 'test_ppo_pause_runtime_v179.py', '-q'])
    run_stage('reward_clock_self_test', ['-m', 'tools.ppo_reward_timing_audit', '--self-test'])
    common = ['--resolution', '512x288', '--fps', '30', '--rate-mode', 'uncapped',
              '--press-delay-ms', '5', '--capture-mode', 'fast', '--resize-method', 'linear',
              '--state-model', 'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt',
              '--state-confidence', '0.90', '--detector', 'ai_data/v5_hazards/runs/segment_round04/weights/best.pt',
              '--imgsz', '960', '--conf', '0.20', '--detect-every', '3', '--max-stale-ms', '300',
              '--episode-max-seconds', '15', '--seed', '123']
    try:
        for source in source_info:
            name = source['name']
            train_name = f'v612_ab_{name}_train_20261028'
            eval_name = f'v612_ab_{name}_eval_20261028'
            train_dir = ROOT/'ai_data/rl_v2_runs'/train_name
            run_stage(name+'_train', ['train_rs_v612_ppo.py', '--mode', 'train',
                '--run-name', train_name, '--resume', source['checkpoint'], '--steps', str(STEPS),
                '--max-minutes', '30', '--save-every-minutes', '4', '--pause-ppo-optimizer',
                '--resync-rollouts', '--audit-reward-timing', *common])
            trained = json.loads((train_dir/'v61_report.json').read_text())
            if not trained['completed_requested_steps'] or trained['steps'] != STEPS:
                raise RuntimeError(f'{name} did not complete the equal step budget')
            run_stage(name+'_stability_audit', ['-m', 'tools.ppo_reward_timing_audit',
                '--log', str(REPORT/(name+'_train.log')), '--min-resumes', '31', '--min-steps', str(STEPS)])
            run_stage(name+'_reward_replay', ['-m', 'tools.ppo_reward_timing_audit',
                '--reward-steps', str(train_dir/'ppo_reward_step_audit.jsonl'), '--min-steps', str(STEPS)])
            output_model = train_dir/'latest.zip'
            output_sha = sha(output_model)
            run_stage(name+'_eval', ['train_rs_v612_ppo.py', '--mode', 'eval', '--run-name', eval_name,
                '--resume', str(output_model.relative_to(ROOT)), '--eval-policy', 'deterministic',
                '--eval-episodes', '100', '--max-minutes', '45', *common])
            evaluation = json.loads((ROOT/'ai_data/rl_v2_runs'/eval_name/'v61_eval_report.json').read_text())
            if evaluation['eval_episodes'] != 100 or evaluation['detector']['v612_live_input_guard']['guard_hard_stopped']:
                raise RuntimeError(f'{name} evaluation was incomplete or guard-stopped')
            assert sha(output_model) == output_sha
            assert sha(ROOT/source['checkpoint']) == source['sha256']
            status['completed'].append({'branch': name, 'training_run': train_name,
                'evaluation_run': eval_name, 'output_sha256': output_sha})
            write(REPORT/'experiment.json', status)
        status['status'] = 'COMPLETE'
    except BaseException as exc:
        status['status'], status['error'] = 'STOPPED', repr(exc)
        raise
    finally:
        for source in source_info:
            assert sha(ROOT/source['checkpoint']) == source['sha256']
        status['source_hashes_unchanged'] = True
        write(REPORT/'experiment.json', status)
    print('[EXPERIMENT COMPLETE]', REPORT, flush=True)


if __name__ == '__main__':
    main()
