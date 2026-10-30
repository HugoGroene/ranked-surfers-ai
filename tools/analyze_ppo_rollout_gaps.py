#!/usr/bin/env python3
"""Analyze non-invasive PPO rollout-resume gap JSONL, without opening the game."""
import argparse
import json
import statistics
from pathlib import Path


def percentile(values, percent):
    if not values:
        return None
    xs = sorted(values)
    index = (len(xs) - 1) * percent / 100
    left, right = int(index), min(len(xs) - 1, int(index) + 1)
    return round(xs[left] + (xs[right] - xs[left]) * (index - left), 2)


def summarize(path):
    records = []
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('event') != 'ppo_rollout_resume_gap':
            raise ValueError(f'unexpected event on line {lineno}')
        records.append(record)
    if not records:
        return {'rollout_boundaries': 0, 'note': 'No completed PPO update boundaries yet.'}
    def dist(key):
        values = [float(x[key]) for x in records if isinstance(x.get(key), (int, float))]
        return {'n':len(values), 'median_ms':round(statistics.median(values), 2) if values else None,
                'p95_ms':percentile(values, 95), 'max_ms':round(max(values), 2) if values else None}
    return {
        'rollout_boundaries': len(records),
        'optimizer_and_overhead': dist('rollout_end_to_next_start_ms'),
        'cached_observation_age': dist('cached_frame_to_next_start_ms'),
        'last_env_step_gap': dist('last_env_step_to_next_start_ms'),
        'potentially_stale_boundaries': sum(1 for x in records if x['potentially_stale']),
        'boundaries_after_terminal': sum(1 for x in records if x['last_env_step_done']),
        'warning': 'Elapsed-age statistics do not prove that a specific action was unsafe. '
                   'Do not interpret as measured survival loss.',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl', type=Path, help='new PPO run folder/ppo_rollout_gaps.jsonl')
    args = p.parse_args()
    print(json.dumps(summarize(args.jsonl), indent=2))


if __name__ == '__main__':
    main()
