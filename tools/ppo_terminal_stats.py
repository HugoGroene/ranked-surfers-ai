"""Format cached survival statistics without capture, inference or file access."""


def format_terminal_stats(summary,elapsed_seconds,transitions,now,benchmark_only=False):
    summary=summary or {}
    def seconds(key):
        value=summary.get(key)
        return f'{value:.2f}s' if value is not None else 'n/a'
    elapsed=max(0.,elapsed_seconds)
    minutes,seconds_elapsed=divmod(int(elapsed),60)
    rate=transitions/elapsed if elapsed else 0.
    updated=summary.get('updated_at')
    age=f'{max(0.,now-updated):.1f}s' if updated is not None else 'n/a'
    label='FROZEN STATS' if benchmark_only else 'PPO STATS'
    metric='processed/s' if benchmark_only else 'optimized/s'
    return (f'[{label}] episodes={summary.get("completed_episodes","n/a")} '
        f'overall_mean={seconds("overall_mean_seconds")} '
        f'recent100_mean={seconds("recent_mean_seconds")} '
        f'recent_n={summary.get("recent_samples",0)} max={seconds("best_seconds")} '
        f'time_limits={summary.get("time_limited_episodes",0)} '
        f'{metric}={rate:.1f} elapsed={minutes:02d}:{seconds_elapsed:02d} stats_age={age}')
