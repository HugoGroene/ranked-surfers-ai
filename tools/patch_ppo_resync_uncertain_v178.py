#!/usr/bin/env python3
"""Source-hash-locked, helper-only PPO post-resume uncertain-frame retry fix."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'tools/ppo_resync_transient_v172.py'
ORIGINAL_SHA = 'f2805628c41ed64041cb29a72097029ea03a4a3ada454a63ed3d84d373ec32bd'
PAYLOAD = ROOT/'tools/_ppo_resync_uncertain_v178_payload/ppo_resync_transient_v172.py'
ORIGINAL_PAYLOAD = ROOT/'tools/_ppo_resync_uncertain_v178_payload/ppo_resync_transient_v172_original.txt'
BACKUP_SUFFIX = '.pre_ppo_resync_uncertain_v178.bak'


def sha(data):
    return hashlib.sha256(data).hexdigest()

def normalize_eol(data):
    if b'\r' in data.replace(b'\r\n', b''):
        raise RuntimeError('Unexpected mixed/CR line endings in helper')
    return data.replace(b'\r\n', b'\n')


def prepare(root=ROOT):
    target = root/MODULE
    trainer = root/'train_rs_v612_ppo.py'
    if not trainer.is_file():
        raise RuntimeError('Trainer absent')
    trainer_src = trainer.read_text(encoding='utf8')
    if ('def refresh_after_ppo_optimizer(self):' not in trainer_src or
        'from tools.ppo_resync_transient_v172 import (' not in trainer_src):
        raise RuntimeError('Trainer does not use the expected post-resume helper')
    before = target.read_bytes()
    canonical = normalize_eol(before)
    if sha(canonical) != ORIGINAL_SHA:
        raise RuntimeError(f'Unexpected helper normalized SHA256 {sha(canonical)}; expected {ORIGINAL_SHA}. No changes.')
    baseline = ORIGINAL_PAYLOAD.read_bytes()
    candidate = PAYLOAD.read_bytes()
    after = candidate.replace(b'\n', b'\r\n') if b'\r\n' in before else candidate
    if canonical != baseline:
        raise RuntimeError('Packaged original does not match the installed helper')
    if after == before:
        raise RuntimeError('Replacement identical to original')
    compile(after, str(target), 'exec')
    before_txt = canonical.decode('utf8')
    after_expected = before_txt.replace('''        # Only a near-GAMEPLAY uncertain state with no independent MENU evidence
        # and no substantial ML MENU probability gets this bounded grace.
        if (state == 'UNCERTAIN' and .70 <= p_game < conf and p_menu < .10
                and score < threshold * .30 and self.skipped < MAX_TRANSIENT_SKIPS):
''','''        # A brief UNCERTAIN frame is not evidence of gameplay or death.
        # In particular, a tunnel/camera transition can push p_game below .70.
        # Passive retries may wait for independent proof, but DO NOT permit a
        # movement key: four consecutive high-confidence GAMEPLAY captures are
        # still required. MENU evidence, other states and exhausted retries
        # still abort. Never count an UNCERTAIN frame toward the image stack.
        if (state == 'UNCERTAIN' and p_game < conf and p_menu < .10
                and score < threshold * .30 and self.skipped < MAX_TRANSIENT_SKIPS):
''').replace("'near-gameplay UNCERTAIN; no menu evidence'", "'transient UNCERTAIN; no corroborated menu evidence'")
    if normalize_eol(after) != after_expected.encode('utf8'):
        raise RuntimeError('Replacement alters unexpected helper code')
    backup = target.with_name(target.name+BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f'Backup {backup.name} exists; refusing second installation')
    return target, backup, before, after


def apply(root=ROOT):
    target, backup, before, after = prepare(root)
    with backup.open('xb') as f:
        f.write(before); f.flush(); os.fsync(f.fileno())
    fd, temp_path = tempfile.mkstemp(prefix='.ppo-v178-', suffix='.py', dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(after); f.flush(); os.fsync(f.fileno())
        os.replace(temp_path,target)
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)
    return backup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--apply', action='store_true')
    a=p.parse_args()
    try:
        if a.check:
            prepare()
            print('[PPO RESYNC UNCERTAIN V178] DRY RUN PASS; exact original helper matched. No files changed.')
        else:
            backup=apply()
            print('[PPO RESYNC UNCERTAIN V178] Installed bounded passive uncertain-frame retries.')
            print(f'[PPO RESYNC UNCERTAIN V178] Backup: {backup.name}; trainer/model/rewards unchanged.')
    except Exception as e:
        p.exit(2, '[PPO RESYNC UNCERTAIN V178] REFUSED: '+str(e)+'\n')

if __name__=='__main__': main()
