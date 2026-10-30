"""Bounded passive retries for a verified stack that aged during processing."""
import time
from rs_v612_gameplay_guard import GuardUncertainError


def verified_refresh(raw, *, clock=time.perf_counter, attempts=3, max_age_ms=100.):
    reference=getattr(raw,'_ppo_pause_visual_reference',None)
    for attempt in range(attempts):
        # A successful but stale first stack must not discard the rendered-Pause
        # fingerprint for subsequent verification attempts.
        if reference is not None:
            raw._ppo_pause_visual_reference=reference
        observation,reason,captured_at=raw.refresh_after_ppo_optimizer()
        if observation is None:
            return observation,reason,captured_at  # Existing MENU/uncertain handling.
        age_ms=(clock()-captured_at)*1000 if captured_at is not None else float('inf')
        if 0 <= age_ms < max_age_ms:
            return observation,reason,captured_at
        if age_ms < 0:
            break
        if attempt+1 < attempts:
            print(f'[PPO RESYNC FRESHNESS] Verified stack aged {age_ms:.1f}ms; '
                  f'passive retry {attempt+1}/{attempts-1}; NO movement sent.',flush=True)
    raw._guard_hard_stopped=True
    raise GuardUncertainError(f'Refresh itself is stale ({age_ms:.1f}ms) after bounded passive retries. '
                              'Stopping instead of acting on an old observation.')
