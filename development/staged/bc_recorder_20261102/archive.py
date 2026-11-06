"""Full-canvas lossless source archive, separate from legacy compatibility frames."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def save_sources(output, samples):
    destination = Path(output)/'source_frames'
    destination.mkdir(exist_ok=False)
    shape = None
    index = []
    for i, sample in enumerate(samples):
        payload = sample['canvas_jpeg']  # Historical field name; staged recorder encodes PNG.
        if not payload.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Native archive requires lossless PNG; JPEG is not silently accepted')
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None or (shape is not None and list(image.shape) != shape):
            raise ValueError('Invalid source frame or changed geometry')
        shape = list(image.shape)
        name = f'{i:06d}.png'
        (destination/name).write_bytes(payload)
        index.append({'file': name, 'sha256': hashlib.sha256(payload).hexdigest(),
                      'captured_at_perf_counter': float(sample['at'])})
    (destination/'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    return {'version': 1, 'directory': 'source_frames', 'index': 'source_frames/index.json',
            'encoding': 'lossless PNG', 'shape_hwc': shape, 'count': len(index),
            'timestamps': 'perf_counter; same frame origin as timestamps.npy',
            'index_sha256': hashlib.sha256((destination/'index.json').read_bytes()).hexdigest()}
