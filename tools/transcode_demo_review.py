"""Create browser-compatible review copies; never modifies original videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2


def main(output):
    queue_path = output/'queue.json'
    queue = json.loads(queue_path.read_text(encoding='utf8'))
    root = Path(queue['source_root'])/'runs'
    encoder = shutil.which('ffmpeg')
    if not encoder:
        raise RuntimeError('ffmpeg is not installed; no dependencies installed automatically')
    media = output/'video'
    media.mkdir(exist_ok=True)

    def inspect(path):
        cap = cv2.VideoCapture(str(path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()

    def convert(row):
        folder = root/row['name']
        manifest_bytes = (folder/'manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != row['manifest_sha256']:
            raise RuntimeError('Replay manifest changed since audit: '+row['name'])
        source = folder/json.loads(manifest_bytes)['preview_video']
        target = media/(row['name']+'.h264.mp4')
        expected = (len(row['times']), row['video_fps'])
        if not target.exists():
            common = [encoder, '-nostdin', '-hide_banner', '-loglevel', 'error',
                      '-n', '-i', str(source), '-an']
            args = common + ['-c:v', 'h264_nvenc', '-preset', 'p3', '-cq', '23',
                             '-b:v', '0', '-pix_fmt', 'yuv420p', '-fps_mode',
                             'passthrough', '-movflags', '+faststart', str(target)]
            result = subprocess.run(args, capture_output=True, text=True, timeout=180)
            if result.returncode:
                # Keep any failed hardware output; use a separate new CPU file.
                target = media/(row['name']+'.software.mp4')
                args = common + ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf',
                                 '23', '-pix_fmt', 'yuv420p', '-fps_mode',
                                 'passthrough', '-movflags', '+faststart', str(target)]
                result = subprocess.run(args, capture_output=True, text=True, timeout=180)
                if result.returncode:
                    raise RuntimeError(f"Transcoding {row['name']} failed: {result.stderr[-400:]}")
        count, fps = inspect(target)
        if count != expected[0] or abs(fps-expected[1]) > .001:
            raise RuntimeError('Review copy changed frame count/rate: '+row['name'])
        return row['name'], target.name, source.stat().st_size, target.stat().st_size

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(convert, row) for row in queue['runs']]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0:
                print(f'[REVIEW VIDEO] Converted/verified {len(results)}/{len(futures)}', flush=True)
    lookup = {name: filename for name, filename, _, _ in results}
    for row in queue['runs']:
        row['original_video'] = row.get('original_video', row['video'])
        row['video'] = 'video/'+lookup[row['name']]
    queue_path.write_text(json.dumps(queue), encoding='utf8')
    template = Path(__file__).with_name('demo_review_template.html').read_text(encoding='utf8')
    (output/'review.html').write_text(template.replace(
        '__REVIEW_DATA__', json.dumps(queue).replace('<', '\\u003c')), encoding='utf8')
    (output/'video_conversion.json').write_text(json.dumps({
        'verified_frame_count_and_fps': len(results), 'originals_modified': False,
        'copies': [{'run': name, 'copy': filename, 'source_bytes': size, 'copy_bytes': converted}
                   for name, filename, size, converted in sorted(results)]}, indent=2), encoding='utf8')
    print(f'Browser-compatible review copies ready: {output / "review.html"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
