"""CPU-only synthetic history selection and allocation benchmark; no gameplay."""
import argparse
import json
import time

import numpy as np

from framebuffer import HistoryConfig, TimestampedFrameBuffer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--span-ms', type=float, default=600)
    p.add_argument('--capture-hz', type=float, default=30)
    p.add_argument('--samples', type=int, default=100)
    a = p.parse_args()
    if not np.isfinite(a.capture_hz) or a.capture_hz <= 0 or a.samples <= 0:
        p.error('capture-hz and samples must be positive')
    config = HistoryConfig(frames=a.frames, span_seconds=a.span_ms/1000)
    buffer = TimestampedFrameBuffer(config)
    buffer.reset('synthetic_episode')
    frame = np.zeros((3, 288, 512), dtype=np.uint8)
    warmup = int(np.ceil((config.span_seconds+0.1)*a.capture_hz))
    costs = []
    for i in range(warmup+a.samples):
        timestamp = 10+i/a.capture_hz
        start = time.perf_counter()
        buffer.append(frame, timestamp, segment='synthetic_episode')
        if i >= warmup:
            observation = buffer.sample(timestamp+0.001)
            costs.append((time.perf_counter()-start)*1000)
    print(json.dumps(dict(contract=config.contract(), shape=list(observation.image.shape),
                         selected_ages_ms=(observation.ages_seconds*1000).tolist(),
                         buffer_mib=buffer.stored_bytes/2**20,
                         observation_mib=observation.image.nbytes/2**20,
                         dense_128_step_image_mib=128*observation.image.nbytes/2**20,
                         append_and_sample_median_ms=float(np.median(costs)),
                         append_and_sample_p95_ms=float(np.percentile(costs,95)),
                         note='Synthetic CPU allocation test, not live reaction latency or training throughput'), indent=2))


if __name__ == '__main__':
    main()
