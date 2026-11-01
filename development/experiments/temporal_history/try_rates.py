"""Offline simulated capture/decision clocks. No browser, model, sleeps or GPU."""
import argparse
from collections import Counter
import json
import math
import time
import numpy as np
from framebuffer import HistoryConfig, HistoryUnavailable, TimestampedFrameBuffer


def simulate(config, capture_hz, decision_hz, samples=100, shape=(3,288,512)):
    if any(not math.isfinite(r) or r <= 0 for r in (capture_hz, decision_hz)) or samples <= 0:
        raise ValueError('Rates and sample count must be positive and finite')
    b = TimestampedFrameBuffer(config)
    b.reset('synthetic')
    frame = np.zeros(shape, np.uint8)
    captured = 0
    rejected = Counter()
    costs = []
    obs = None
    for i in range(samples):
        now = config.span_seconds + .1 + i/decision_hz
        while captured/capture_hz <= now:
            b.append(frame,captured/capture_hz,segment='synthetic')
            captured += 1
        start = time.perf_counter()
        try:
            obs = b.sample(now)
        except HistoryUnavailable as exc:
            rejected[str(exc)] += 1
        else:
            costs.append((time.perf_counter()-start)*1000)
    return dict(contract=config.contract(), capture_hz=capture_hz, decision_hz=decision_hz,
                accepted=len(costs), requested=samples, rejected=dict(rejected), captures=captured,
                shape=list(obs.image.shape) if obs else None,
                last_ages_ms=(obs.ages_seconds*1000).tolist() if obs else None,
                buffer_mib=b.stored_bytes/2**20,
                dense_128_step_image_mib=float(128*config.frames*np.prod(shape)/2**20),
                sample_cpu_median_ms=float(np.median(costs)) if costs else None,
                sample_cpu_p95_ms=float(np.percentile(costs,95)) if costs else None,
                note='Synthetic timing only; excludes rendering, inference and guards. Not live throughput.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--span-ms',type=float,default=600)
    p.add_argument('--capture-hz',type=float,default=60)
    p.add_argument('--decision-hz',type=float,default=30)
    p.add_argument('--samples',type=int,default=100)
    a = p.parse_args()
    result = simulate(HistoryConfig(frames=a.frames,span_seconds=a.span_ms/1000),
                      a.capture_hz,a.decision_hz,a.samples)
    print(json.dumps(result,indent=2))
    if result['accepted'] != result['requested']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
