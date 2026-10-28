#!/usr/bin/env python3
"""Independent read-only audit of PPO pause/resume log, plus reward-clock regression.

Run from Claude development clone:
 python -B -m tools.ppo_reward_timing_audit --self-test
 python -B -m tools.ppo_reward_timing_audit --log ai_data/ppo_pause_<run>.log --min-resumes 15 --min-steps 2048

No Playwright, actions, model, checkpoint or training code is modified.
"""
import argparse
import json
import math
from pathlib import Path
import re


class AuditFailure(ValueError):
    pass


def _lines(payload):
    return payload.splitlines() if isinstance(payload, str) else list(payload)


def read_training_log(path):
    """Read Windows PowerShell Tee-Object UTF-16LE or UTF-8 logs safely."""
    data = Path(path).read_bytes()
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16')
    if data.startswith(b'\xef\xbb\xbf'):
        return data.decode('utf-8-sig')
    # PowerShell's Tee-Object on Windows PowerShell may produce BOM-less UTF-16LE.
    sample = data[: min(4096, len(data))]
    if len(sample) > 20 and sample[1::2].count(0) > len(sample[1::2]) * .55:
        return data.decode('utf-16-le')
    return data.decode('utf-8', errors='replace')


def audit_training_log(payload, *, min_resumes=3, min_steps=512, age_limit_ms=100.0):
    """Validate reported pause/resume and observation invariants, not Unity internals.

    Accept one additional verified freeze at final rollout end: SB3 calls
    _on_rollout_end before its final PPO.train(), but does not start another
    rollout to resume. The browser closes when learning completes.
    """
    lines = _lines(payload)
    # Accept pasted PowerShell transcripts as well as raw Tee-Object logs.
    # Diff previews and Select-String summaries repeat tokens but are NOT
    # training events; only the last actual [V612 TRAIN]..[V612 REPORT] block
    # is eligible to establish successful completion.
    starts = [j for j, line in enumerate(lines) if line.startswith('[V612 TRAIN]')]
    if starts:
        block_start = starts[-1]
        endings = [j for j in range(block_start, len(lines))
                   if lines[j].startswith('[V612 REPORT]')]
        if endings:
            lines = lines[block_start:endings[0]+1]
        else:
            lines = lines[block_start:]
    freezes, verified, resumes, refreshed, gaps, frames, prechecks = 0, 0, [], [], [], [], 0
    menu_boundaries = menu_ready = menu_terminal = post_resume_menu_ready = 0
    failures = []
    reports = []
    messages = []
    if min_resumes < 1 or min_steps < 1 or not math.isfinite(age_limit_ms) or age_limit_ms <= 0:
        raise ValueError('Invalid audit target')
    for i, line in enumerate(lines, 1):
        if line.startswith('[PPO MENU HANDOFF] Boundary MENU confirmed by'):
            menu_boundaries += 1
        if line.startswith('[PPO MENU HANDOFF] MENU reverified after optimizer;'):
            menu_ready += 1
            m = re.search(r'menu_idle_seconds_excluded=([\d.]+)', line)
            if not m or not 0.0 < float(m.group(1)) <= 30.0:
                failures.append(f'line {i}: invalid MENU idle clock correction')
        if line.startswith('[PPO POST RESUME MENU] Confirmed MENU after '):
            post_resume_menu_ready += 1
            m = re.search(
                r'after (\d+) fresh ML\+template captures; '
                r'menu_idle_seconds_excluded=([\d.]+); '
                r'deferred guarded terminal with NO movement\.', line)
            if (not m or not 4 <= int(m.group(1)) <= 8 or
                    not 0.0 < float(m.group(2)) <= 30.0):
                failures.append(f'line {i}: invalid post-resume MENU handoff confirmation')
        if line.startswith('[PPO MENU HANDOFF] Terminal recorded by guarded env.step;'):
            menu_terminal += 1
            if not re.search(r'death_reward=-[\d.]+; NO movement key sent', line):
                failures.append(f'line {i}: missing terminal/death/no-key evidence')
        if '[PPO PAUSE] Freeze verified after' in line:
            freezes += 1
        if '[PPO PAUSE] Verified frozen gameplay before optimizer' in line:
            verified += 1
        if '[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY' in line:
            prechecks += 1
        if '[PPO PAUSE] Resumed after optimizer with pause-aware clocks:' in line:
            value = line.split('clocks:', 1)[1].strip()
            try:
                report = json.loads(value)
                for key in ('corrected_episode_timer', 'corrected_action_timer', 'corrected_monitor_timer'):
                    if report.get(key) is not True:
                        raise ValueError('missing true ' + key)
                paused = float(report['paused_wall_seconds'])
                drift = float(report['active_episode_clock_drift_s'])
                if not math.isfinite(paused) or not 0.0 < paused <= 30:
                    raise ValueError('implausible paused interval')
                if not math.isfinite(drift) or not -0.1 <= drift <= 0.2:
                    raise ValueError('implausible active-episode drift')
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                failures.append(f'line {i}: invalid pause-clock evidence: {exc}')
            else:
                resumes.append(report)
        if '[PPO GAP] {' in line:
            try:
                gaps.append(json.loads(line.split('[PPO GAP] ', 1)[1]))
            except json.JSONDecodeError:
                failures.append(f'line {i}: malformed PPO GAP JSON')
        if '[PPO RESYNC TRACE] frame=' in line:
            m = re.search(r'frame=(\d+)/4\s+since_resume_ms=([\d.]+)\s+state=(\S+)\s+menu_score=([\d.]+)', line)
            if not m:
                failures.append(f'line {i}: incomplete per-frame resume trace')
            else:
                frames.append({'frame': int(m.group(1)), 'ms': float(m.group(2)),
                               'state': m.group(3), 'menu_score': float(m.group(4))})
        if '[PPO RESYNC] refreshed 4-frame stack; cached age now' in line:
            m = re.search(r'cached age now (\d+(?:\.\d+)?)ms', line)
            if not m:
                failures.append(f'line {i}: missing fresh observation age')
            else:
                refreshed.append(float(m.group(1)))
        if '[V612 REPORT] ' in line:
            try:
                reports.append(json.loads(line.split('[V612 REPORT] ', 1)[1]))
            except json.JSONDecodeError:
                failures.append(f'line {i}: malformed V612 report JSON')
        if ('Safe stop' in line or 'ABANDONED' in line or 'Traceback' in line or
                'uncaught exception' in line.lower()):
            failures.append(f'line {i}: interrupted/unsafe training: {line.strip()[:180]}')

    if not reports:
        failures.append('Missing final [V612 REPORT]')
        summary = {}
    else:
        summary = reports[-1]
        if summary.get('completed_requested_steps') is not True:
            failures.append('Final report says requested steps were not completed')
        if not isinstance(summary.get('steps'), int) or summary['steps'] < min_steps:
            failures.append(f"Expected >= {min_steps} steps; reported {summary.get('steps')!r}")
    continuations = len(resumes) + menu_ready
    if continuations < min_resumes:
        failures.append(f'Only {continuations} verified optimizer continuations '
                        f'({len(resumes)} pause resumes, {menu_ready} MENU handoffs); '
                        f'require {min_resumes}')
    if menu_boundaries not in (menu_ready, menu_ready + 1):
        failures.append(f'MENU boundary/continuation imbalance ({menu_boundaries}/{menu_ready})')
    expected_terminals = menu_ready + post_resume_menu_ready
    if menu_terminal != expected_terminals:
        failures.append(f'MENU handoffs missing real terminal rewards '
                        f'({expected_terminals}/{menu_terminal})')
    if post_resume_menu_ready > len(resumes):
        failures.append('More post-resume MENU handoffs than optimizer resumes')
    if menu_boundaries == menu_ready + 1:
        messages.append('Final rollout ended on MENU without another PPO step; final terminal reward is not in the buffer.')
    if freezes != verified:
        failures.append(f'Freeze/verification counts differ ({freezes}/{verified})')
    if verified not in (len(resumes), len(resumes)+1):
        failures.append(f'Freeze/resume imbalance ({verified} freezes vs {len(resumes)} resumes)')
    if len(refreshed) + post_resume_menu_ready != len(resumes):
        failures.append(f'Fresh stacks plus verified terminal handoffs '
                        f'{len(refreshed)}+{post_resume_menu_ready} != resumes {len(resumes)}')
    if len(gaps) != continuations:
        failures.append(f'Optimizer gap events {len(gaps)} != '
                        f'verified continuations {continuations}')
    if any(not math.isfinite(ms) or ms < 0 or ms >= age_limit_ms for ms in refreshed):
        failures.append(f'Some cached post-resume observations are not fresh (<{age_limit_ms}ms)')
    if frames:
        if len(frames) != 4 * len(refreshed):
            failures.append(f'Trace contains {len(frames)} frames, need {4 * len(refreshed)}')
        for j, frame in enumerate(frames):
            if frame['frame'] != (j % 4)+1 or frame['state'] != 'GAMEPLAY':
                failures.append(f'Bad frame in trace index {j}: {frame}')
            if frame['ms'] < 0 or frame['ms'] > 1500:
                failures.append(f'Implausibly late resume frame index {j}: {frame["ms"]}')
    # A verified GAMEPLAY pre-pause screenshot can race with an actual death:
    # pause_for_ppo_optimizer() logged the screenshot, but the control vanished
    # and the independently corroborated MENU handoff replaced the freeze.
    # Each confirmed boundary MENU can explain AT MOST one such extra check.
    # Missing or otherwise unexplained checks still fail closed.
    if prechecks and not (verified <= prechecks <= verified + menu_boundaries):
        failures.append(
            f'Pre-pause classifier checks {prechecks} outside expected '
            f'{verified}..{verified + menu_boundaries} '
            f'(verified freezes plus corroborated boundary MENU races)')
    if not frames:
        messages.append('No per-frame trace: only 4-frame replacement/freshness verified')
    messages.append('Reported pause-clock drift is internal consistency, not an independent Unity timer reading.')
    messages.append('No evidence here that the reward cap or per-step PPO discount is unbiased.')
    result = {
        'status': 'PASS_STABILITY_CANDIDATE' if not failures else 'FAIL_SAFE_AUDIT',
        'steps': summary.get('steps'),
        'completed': summary.get('completed_requested_steps'),
        'freeze_checks': freezes,
        'verified_optimizer_pauses': verified,
        'optimizer_resumes': len(resumes),
        'menu_boundaries': menu_boundaries,
        'menu_handoffs_after_optimizer': menu_ready,
        'post_resume_menu_handoffs': post_resume_menu_ready,
        'menu_terminal_rewards_recorded': menu_terminal,
        'optimizer_continuations': continuations,
        'fresh_stacks': len(refreshed),
        'fresh_frame_age_max_ms': round(max(refreshed), 2) if refreshed else None,
        'pause_duration_total_s': round(sum(float(r['paused_wall_seconds']) for r in resumes), 3),
        'max_reported_abs_active_clock_drift_s':
            round(max(abs(float(r['active_episode_clock_drift_s'])) for r in resumes), 3) if resumes else None,
        'per_frame_gameplay_checks': len(frames),
        'completed_episodes': summary.get('episodes'),
        'mean_survival_seconds': summary.get('mean_survival_seconds'),
        'warnings': messages,
        'failures': failures,
    }
    return result


def audit_reward_steps(payload, *, minimum_steps=512):
    """Replay the actual ProgressRewards implementation on recorded env steps.

    The monitor supplies terminal/truncation evidence independently of the
    logged reward components. This detects excess milestone/death/survival
    credit across rollouts, pauses and episode resets. It does NOT establish
    whether Unity's own clock advanced while paused.
    """
    from rs_v61_rewards import ProgressRewards
    try:
        events = [json.loads(line) for line in _lines(payload) if line.strip()]
    except json.JSONDecodeError as exc:
        return {'status': 'FAIL_REWARD_REPLAY', 'failures': [f'Invalid JSONL: {exc}']}
    calc = ProgressRewards()
    failures, episodes, capped_steps, max_duration_increment = [], 0, 0, 0.0
    boundary_terminals = 0
    prior_duration = 0.0
    prior_done = True
    prev_step = None
    for pos, event in enumerate(events, 1):
        try:
            step = int(event['step'])
            duration = float(event['episode_duration_s'])
            done = event['done']
            truncated = event['truncated']
            observed = {k: float(event[k]) for k in
                        ('reward_survival', 'reward_milestones', 'reward_death', 'reward_v61')}
            observed_env_reward = float(event['environment_reward'])
            if not all(math.isfinite(x) for x in [duration, observed_env_reward, *observed.values()]):
                raise ValueError('non-finite value')
            if not isinstance(done, bool) or not isinstance(truncated, bool):
                raise ValueError('done/truncated must be bool')
            if prev_step is not None and step != prev_step + 1:
                raise ValueError(f'missing timestep: {prev_step} -> {step}')
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f'event {pos}: {exc}')
            break
        if prior_done:
            calc.reset()
            prior_duration = 0.0
        if truncated and not done:
            failures.append(f'event {pos}: truncated flag requires done=true')
        if duration < prior_duration - 0.012:
            failures.append(f'event {pos}: duration decreased without a reset')
        increment = max(0.0, duration - prior_duration)
        max_duration_increment = max(max_duration_increment, increment)
        if increment > calc.max_rewarded_step_seconds + 0.012:
            capped_steps += 1
        # The cap is intentional; the audit must not pretend capped gameplay
        # time was rewarded. Long gaps are reported separately from mismatch.
        expected, pieces = calc.step(duration, done and not truncated)
        for key, expected_value in [('reward_survival', pieces['survival']),
                                    ('reward_milestones', pieces['milestones']),
                                    ('reward_death', pieces['death']),
                                    ('reward_v61', expected)]:
            if abs(observed[key] - round(expected_value, 4)) > 0.00011:
                failures.append(f'event {pos}: {key} observed {observed[key]:.4f} != expected {expected_value:.4f}')
        if abs(observed_env_reward - expected) > 0.001:
            failures.append(f'event {pos}: training env reward differs from reward calculation')
        if event.get('ppo_menu_handoff_terminal', False):
            if not (done and not truncated and observed['reward_death'] < 0):
                failures.append(f'event {pos}: MENU handoff missing real terminal death penalty')
            boundary_terminals += 1
        if done:
            episodes += 1
        prior_done = done
        prior_duration = duration
        prev_step = step
    if len(events) < minimum_steps:
        failures.append(f'Only {len(events)} reward step events; need >= {minimum_steps}')
    return {
        'status': 'PASS_REWARD_REPLAY' if not failures else 'FAIL_REWARD_REPLAY',
        'validated_steps': len(events), 'completed_episodes': episodes,
        'menu_handoff_terminal_events': boundary_terminals,
        'steps_exceeding_survival_reward_cap': capped_steps,
        'maximum_observed_episode_duration_step_s': round(max_duration_increment, 3),
        'note': 'A capped interval does not get survival credit above 0.15s; review large gaps. '
                'Reward replay verifies trainer arithmetic, not Unity internal time or gamma discounts.',
        'failures': failures[:20],
    }


class _Context:
    def __init__(self):
        self.last_action_times = [None]*5


class _Env:
    def __init__(self, origin):
        self.episode_start = origin
        self.last_step = origin
        self.episode_started = origin
        self.suspected_since = None
        self.previous_observation_at = origin
        self._prev_step_start = origin
        self.action_context = _Context()
        self.last_detection = {'captured_at': origin}  # Must remain old/stale.


class _Monitor:
    def __init__(self, env, origin):
        self.env = env
        self.t_start = origin


def synthetic_reward_regression():
    """Compare real V6.1 reward code with and without explicit long pauses.

    Independent synthetic reference uses *active gameplay durations*, not the
    paused implementation's corrected timers. Includes death and new episode.
    """
    from rs_v61_rewards import ProgressRewards
    from tools.ppo_optimizer_pause_clock import compensate_pause_clocks

    episodes = [
        [(0.11, False), (0.37, False), (0.62, False),
         (0.88, False), (1.16, False), (1.44, False), (1.72, True)],
        [(0.10, False), (0.35, False), (0.60, False),
         (0.86, False), (1.11, True)],
    ]
    pauses = {(0, 2): 1.32, (0, 5): 1.187, (1, 1): 2.187}

    def run(with_pauses):
        wall = 1000.0
        output, paused_total, pause_checks, death_count, duration_checks = [], 0.0, 0, 0, []
        reward_calc = ProgressRewards()
        for ei, steps in enumerate(episodes):
            if ei:
                wall += 0.5  # Between-episode restart, not rewarded in next episode.
                reward_calc.reset()
            start = wall
            env = _Env(start)
            monitor = _Monitor(env, start)
            prev_active = 0.0
            for si, (active, died) in enumerate(steps):
                wall += active - prev_active
                prev_active = active
                observed_duration = round(wall - env.episode_start, 2)
                got, parts = reward_calc.step(observed_duration, died)
                if died:
                    death_count += 1
                output.append((ei, si, round(observed_duration, 4),
                               round(got, 5), {k: round(v, 5) for k, v in parts.items()}))
                if with_pauses and (ei, si) in pauses:
                    pause = pauses[(ei, si)]
                    original_elapsed = wall - env.episode_start
                    env.action_context.last_action_times[3] = wall - 0.08
                    action_age = wall - env.action_context.last_action_times[3]
                    captured_at = env.last_detection['captured_at']
                    wall += pause
                    compensate_pause_clocks(env, pause, pause, monitor=monitor, elapsed_wall=pause)
                    pause_checks += 1
                    paused_total += pause
                    if abs((wall - env.episode_start) - original_elapsed) > 1e-9:
                        raise AuditFailure('Survival duration increased during pause')
                    if abs((wall - env.action_context.last_action_times[3]) - action_age) > 1e-9:
                        raise AuditFailure('Action timing feature aged during pause')
                    if abs((wall - monitor.t_start) - original_elapsed) > 1e-9:
                        raise AuditFailure('Monitor episode timer aged during pause')
                    if env.last_detection['captured_at'] != captured_at:
                        raise AuditFailure('Old detector timestamp was backdated or refreshed')
                    duration_checks.append((ei, si, round(wall - env.episode_start, 3)))
        return output, round(paused_total, 3), pause_checks, death_count, duration_checks

    expected, *_ = run(False)
    actual, total_pause, checks, deaths, times = run(True)
    if expected != actual:
        mismatch = next(((a,b) for a,b in zip(expected,actual) if a!=b), None)
        raise AuditFailure(f'Paused vs active-only reward mismatch: {mismatch}')
    if deaths != len(episodes) or checks != len(pauses):
        raise AuditFailure('Incorrect death/pauses count')
    return {
        'status':'PASS_SYNTHETIC_REWARD_CLOCKS',
        'matched_step_rewards':len(actual),
        'verified_pauses':checks,
        'excluded_pause_seconds':total_pause,
        'verified_death_penalties':deaths,
        'verified_episode_resets':len(episodes)-1,
        'timing_samples':times,
        'warning':'This is an offline regression, not independent Unity gameplay time.'
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--self-test', action='store_true')
    src.add_argument('--log', type=Path)
    src.add_argument('--reward-steps', type=Path)
    p.add_argument('--min-resumes', type=int, default=3)
    p.add_argument('--min-steps', type=int, default=512)
    args = p.parse_args()
    try:
        if args.self_test:
            result = synthetic_reward_regression()
        elif args.reward_steps is not None:
            result = audit_reward_steps(args.reward_steps.read_text(encoding='utf-8', errors='replace'),
                                        minimum_steps=args.min_steps)
        else:
            result = audit_training_log(read_training_log(args.log),
                                        min_resumes=args.min_resumes, min_steps=args.min_steps)
    except (AuditFailure, OSError, ValueError) as exc:
        print('[PPO REWARD AUDIT] FAILED:', str(exc))
        raise SystemExit(2)
    print('[PPO REWARD AUDIT] '+json.dumps(result, indent=2))
    if result['status'].startswith('FAIL'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
