"""Instrument an unchanged local evaluation; never trains or saves weights.

Use the normal train_rs_v612_ppo eval flags. Records actual policy-source capture
timestamps independently of newer guard captures. Dispatch is not Unity response.
"""
import copy
import contextlib
import hashlib
import json
from pathlib import Path
import time

import numpy as np


class GrabAudit:
    def __init__(self, original, audit):
        self.original, self.audit = original, audit

    def __getattr__(self, name):
        return getattr(self.original, name)

    def grab(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.original.grab(*args, **kwargs)
        end = time.perf_counter()
        self.audit.capture_id += 1
        self.audit.latest_grab = {'id': self.audit.capture_id, 'grab_start': start, 'grab_end': end}
        return result


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'count': 0}
    return {'count': len(values), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)),
            'p99_ms': float(np.percentile(values, 99)), 'max_ms': max(values)}


def row_metrics(row):
    source = row['observation']['capture']
    command = row.get('command_start')
    inner = row.get('inner_start')
    return {
        'policy_inference_ms': 1000*(row['prediction_end']-row['prediction_start']),
        'policy_rgb_age_at_predict_ms': 1000*(row['prediction_start']-source['grab_end']),
        'policy_rgb_age_at_command_ms': 1000*(command-source['grab_end']) if command else None,
        'prediction_end_to_command_ms': 1000*(command-row['prediction_end']) if command else None,
        'guard_to_inner_ms': 1000*(inner-row['step_start']) if inner else None,
        'keypress_roundtrip_ms': 1000*(row['command_end']-command) if command and row.get('command_end') else None,
        'policy_rgb_age_at_press_return_ms': 1000*(row['command_end']-source['grab_end']) if command and row.get('command_end') else None,
        'guard_rgb_age_at_command_ms': 1000*(command-row['captures'][-2]['grab_end'])
            if command and len(row['captures']) >= 2 and row['captures'][-1]['phase'] == 'post_action' else None,
        'detector_source_age_at_predict_ms': 1000*(row['prediction_start']-row['observation']['detector_at'])
            if row['observation'].get('detector_at') is not None else None,
    }


class Audit:
    def __init__(self):
        self.latest_grab = self.observation = self.pending = self.active = None
        self.capture_id = 0
        self.rows = []
        self._detach_callbacks = []

    def attach(self, raw):
        original_grabber = raw.grabber
        proxy = GrabAudit(original_grabber, self)
        raw.grabber = proxy
        original_pack, original_capture, original_step = raw._pack, raw._capture, raw.step
        keyboard = raw.page.keyboard
        original_press = keyboard.press

        def pack(*args, **kwargs):
            started = time.perf_counter()
            result = original_pack(*args, **kwargs)
            self.observation = {'capture': copy.deepcopy(self.latest_grab),
                                'packed_start': started, 'packed_end': time.perf_counter(),
                                'detector_at': raw.last_detection.get('captured_at') if raw.last_detection else None}
            return result

        async def capture(*args, **kwargs):
            result = await original_capture(*args, **kwargs)
            if self.active is not None:
                self.active['captures'].append({**self.latest_grab,
                    'capture_processed_end': time.perf_counter(),
                    'phase': 'post_action' if self.active.get('inner_start') else 'guard'})
            return result

        async def press(key, *args, **kwargs):
            row = self.active
            if row is not None and key in ('ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'):
                row['command_start'] = time.perf_counter()
                row['key'] = key
                result = await original_press(key, *args, **kwargs)
                row['command_end'] = time.perf_counter()
                return result
            return await original_press(key, *args, **kwargs)

        def step(action):
            if self.pending is None or self.pending['observation']['capture'] is None:
                raise RuntimeError('Missing actual policy source capture; refusing untracked action')
            row, self.pending = self.pending, None
            row.update(action=int(action), episode=int(raw.episode_number),
                       step_start=time.perf_counter(), captures=[])
            self.active = row
            try:
                result = original_step(action)
                row['done'], row['truncated'] = bool(result[2]), bool(result[3])
                return result
            finally:
                row['step_end'] = time.perf_counter()
                row['metrics'] = row_metrics(row)
                self.rows.append(row)
                self.active = None

        raw._pack, raw._capture, raw.step = pack, capture, step
        keyboard.press = press

        def detach():
            if raw.grabber is proxy:
                raw.grabber = original_grabber
            session = getattr(raw, '_persistent_browser_session', None)
            if session is not None and session.resources.get('grabber') is proxy:
                session.resources['grabber'] = original_grabber
            if keyboard.press is press:
                keyboard.press = original_press
            for name, wrapper, original in (('_pack', pack, original_pack),
                                             ('_capture', capture, original_capture),
                                             ('step', step, original_step)):
                if getattr(raw, name) is wrapper:
                    setattr(raw, name, original)

        self._detach_callbacks.append(detach)

    def detach(self):
        for callback in reversed(self._detach_callbacks):
            callback()
        self._detach_callbacks.clear()

    def summary(self):
        keys = list(self.rows[0]['metrics']) if self.rows else []
        groups = {'all_steps': self.rows,
                  'movement_commands': [r for r in self.rows if r.get('command_start')],
                  'prepared_guard_commands': [r for r in self.rows if r.get('command_start') and
                      not any(c['phase'] == 'guard' for c in r['captures'])],
                  'single_guard_capture_commands': [r for r in self.rows if r.get('command_start') and
                      sum(c['phase'] == 'guard' for c in r['captures']) == 1],
                  'multi_guard_capture_commands': [r for r in self.rows if r.get('command_start') and
                      sum(c['phase'] == 'guard' for c in r['captures']) > 1]}
        return {name: {'steps': len(rows), 'metrics': {key: distribution(
                    r['metrics'].get(key) for r in rows) for key in keys}}
                for name, rows in groups.items()}


@contextlib.contextmanager
def instrument_evaluation(trainer, audit):
    from stable_baselines3 import PPO
    from rs_v2_timing_probe_ab import TimedEnv

    original_create, original_predict, original_inner = trainer.create_live_env, PPO.predict, TimedEnv._step_browser

    def create(*values, **kwargs):
        monitored, raw = original_create(*values, **kwargs)
        audit.attach(raw)
        return monitored, raw

    def predict(model, *values, **kwargs):
        source = copy.deepcopy(audit.observation)
        start = time.perf_counter()
        result = original_predict(model, *values, **kwargs)
        audit.pending = {'observation': source, 'prediction_start': start,
                         'prediction_end': time.perf_counter()}
        return result

    async def inner(raw, *values, **kwargs):
        if audit.active is not None:
            audit.active['inner_start'] = time.perf_counter()
        return await original_inner(raw, *values, **kwargs)

    trainer.create_live_env, PPO.predict, TimedEnv._step_browser = create, predict, inner
    try:
        yield
    finally:
        trainer.create_live_env, PPO.predict, TimedEnv._step_browser = original_create, original_predict, original_inner
        audit.detach()


def save_audit(audit, dest, checkpoint_hash, unchanged):
    result = {'scope': 'Desktop-grab completion to Playwright dispatch/return; not Unity input application',
              'checkpoint_sha256': checkpoint_hash, 'checkpoint_unchanged': unchanged,
              'clock': 'perf_counter; capture times measure MSS grab, not desktop presentation',
              'instrumentation': 'In-memory wrappers; no extra browser calls or model forwards; no weights saved',
              'summary': audit.summary()}
    (dest/'reaction_latency.json').write_text(json.dumps(result, indent=2)+'\n')
    with (dest/'reaction_latency_steps.jsonl').open('w') as handle:
        for row in audit.rows:
            handle.write(json.dumps(row)+'\n')
    print('[REACTION AUDIT]', json.dumps(result['summary'].get('movement_commands', {})), flush=True)


def main():
    import train_rs_v612_ppo as trainer

    args = trainer.parse_args()
    if args.mode != 'eval' or args.eval_policy != 'deterministic' or args.stochastic_eval:
        raise ValueError('Audit supports unchanged deterministic evaluation only')
    audit = Audit()
    checkpoint_hash = hashlib.sha256(args.resume.read_bytes()).hexdigest()
    try:
        with instrument_evaluation(trainer, audit):
            trainer.evaluate(args)
    finally:
        dest = Path('ai_data/rl_v2_runs')/args.run_name
        if dest.exists():
            unchanged = hashlib.sha256(args.resume.read_bytes()).hexdigest() == checkpoint_hash
            save_audit(audit, dest, checkpoint_hash, unchanged)
            if not unchanged:
                raise RuntimeError('Checkpoint hash changed')


if __name__ == '__main__':
    main()
