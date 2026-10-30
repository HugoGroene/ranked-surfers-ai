#!/usr/bin/env python3
"""Add opt-in READ-ONLY real PPO reward-step JSONL; no reward/policy/guard changes.

Run only in Claude clone with installed v1.5.3 and no training running:
 python -B -m tools.install_ppo_reward_step_audit_v16 --check
 python -B -m tools.install_ppo_reward_step_audit_v16 --apply
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_reward_timing_audit_v16.bak')
MARKER = "p.add_argument('--audit-reward-timing'"


def replace_one(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f'Expected exactly one known source anchor ({count}): {old[:120]!r}')
    return source.replace(old, new, 1)


def patch(source):
    if MARKER in source:
        raise ValueError('Reward audit already installed; no second installation')
    for marker in ('[PPO RESYNC TRACE]', '--pause-ppo-optimizer',
                   'compensate_pause_clocks(', 'def pause_for_ppo_optimizer(self):'):
        if marker not in source:
            raise ValueError('Requires installed optimizer pause v1.5.3: ' + marker)

    s = replace_one(source,
        "    p.add_argument('--save-every-minutes',type=float,default=4.)\n",
        "    p.add_argument('--save-every-minutes',type=float,default=4.)\n"
        "    p.add_argument('--audit-reward-timing', action='store_true',\n"
        "                   help='OPT-IN: write read-only per-step reward/timing evidence for offline audit')\n")

    s = replace_one(s,
        "    a=p.parse_args()\n",
        "    a=p.parse_args()\n"
        "    if a.audit_reward_timing and not (a.mode == 'train' and a.pause_ppo_optimizer and a.resync_rollouts):\n"
        "        p.error('--audit-reward-timing requires --mode train --pause-ppo-optimizer --resync-rollouts')\n")

    s = replace_one(s,
        "            self.log_handle = None\n",
        "            self.log_handle = None\n"
        "            self.reward_step_audit_handle = None\n")

    s = replace_one(s,
        "            self.log_handle = self.log_path.open('a',encoding='utf-8')\n",
        "            self.log_handle = self.log_path.open('a',encoding='utf-8')\n"
        "            if a.audit_reward_timing:\n"
        "                self.reward_step_audit_handle = (output/'ppo_reward_step_audit.jsonl').open('w', encoding='utf-8')\n"
        "                print('[PPO REWARD STEP AUDIT] Recording actual env reward components (observation only).', flush=True)\n")

    old = ("            for info in self.locals.get('infos', []):\n"
           "                for key, dest in (('reward_survival','survival'),('reward_milestones','milestones'),\n")
    new = ("            if a.audit_reward_timing:\n"
           "                # Read env->Monitor->DummyVecEnv INFO, after the actual\n"
           "                # policy action. Absolutely no actions/rewards/dones changed.\n"
           "                audited_infos = self.locals.get('infos', [])\n"
           "                audited_dones = np.asarray(self.locals.get('dones', []), dtype=bool).ravel()\n"
           "                audited_rewards = np.asarray(self.locals.get('rewards', []), dtype=float).ravel()\n"
           "                if len(audited_infos) != 1 or len(audited_dones) != 1 or len(audited_rewards) != 1:\n"
           "                    raise RuntimeError('[PPO REWARD STEP AUDIT] Expected single env INFO/DONE/reward')\n"
           "                audit_info = audited_infos[0]\n"
           "                required = ('duration', 'reward_survival', 'reward_milestones',\n"
           "                            'reward_death', 'reward_v61')\n"
           "                if any(k not in audit_info for k in required):\n"
           "                    raise RuntimeError('[PPO REWARD STEP AUDIT] Missing raw reward fields')\n"
           "                audit_row = {\n"
           "                    'step': int(self.num_timesteps),\n"
           "                    'episode_duration_s': float(audit_info['duration']),\n"
           "                    'done': bool(audited_dones[0]),\n"
           "                    'truncated': bool(audit_info.get('TimeLimit.truncated', False)),\n"
           "                    'environment_reward': float(audited_rewards[0]),\n"
           "                    'reward_survival': float(audit_info['reward_survival']),\n"
           "                    'reward_milestones': float(audit_info['reward_milestones']),\n"
           "                    'reward_death': float(audit_info['reward_death']),\n"
           "                    'reward_v61': float(audit_info['reward_v61']),\n"
           "                }\n"
           "                self.reward_step_audit_handle.write(json.dumps(audit_row)+'\\n')\n"
           "                self.reward_step_audit_handle.flush()\n"
           "            for info in self.locals.get('infos', []):\n"
           "                for key, dest in (('reward_survival','survival'),('reward_milestones','milestones'),\n")
    s = replace_one(s, old, new)

    s = replace_one(s,
        "        def _on_training_end(self):\n"
        "            if self.log_handle is not None:\n"
        "                self.log_handle.close()\n",
        "        def _on_training_end(self):\n"
        "            if self.reward_step_audit_handle is not None:\n"
        "                self.reward_step_audit_handle.close()\n"
        "            if self.log_handle is not None:\n"
        "                self.log_handle.close()\n")

    compile(s, str(TARGET), 'exec')
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--apply', action='store_true')
    args = p.parse_args()
    raw = TARGET.read_bytes()
    crlf = b'\r\n' in raw
    original = raw.decode('utf-8').replace('\r\n', '\n')
    modified = patch(original)
    if args.check:
        print(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                fromfile='a/train_rs_v612_ppo.py', tofile='b/train_rs_v612_ppo.py')))
        print('DRY RUN: no files changed. New audit switch defaults OFF.')
        return
    if BACKUP.exists():
        p.error('Existing backup; refusing to overwrite: '+str(BACKUP))
    BACKUP.write_bytes(raw)
    temp = TARGET.with_name(TARGET.name+'.ppo_reward_v16_tmp')
    try:
        temp.write_bytes((modified.replace('\n','\r\n') if crlf else modified).encode('utf-8'))
        os.replace(temp, TARGET)
    finally:
        if temp.exists():
            temp.unlink()
    print('Installed opt-in, observation-only reward-step audit v1.6.')
    print('Existing rewards, timing adjustment, classifier, PPO actions/checkpoints unchanged.')
    print('Backup:', BACKUP)


if __name__ == '__main__':
    main()
