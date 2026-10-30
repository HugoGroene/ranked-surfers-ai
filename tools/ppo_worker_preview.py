"""Optional loopback-only grid of existing policy frames; never captures again."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import cv2


def write_preview(folder, image_chw, metadata, *, now=None):
    """Best-effort diagnostic. A browser reading a file must never stop control."""
    folder = Path(folder)
    try:
        rgb = image_chw[-3:].transpose(1, 2, 0)
        small = cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_AREA)
        temporary = folder/'live.partial.jpg'
        if not cv2.imwrite(str(temporary), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 70]):
            return False
        temporary.replace(folder/'live.jpg')
        temp_json = folder/'live.partial.json'
        temp_json.write_text(json.dumps(dict(metadata, updated_at=time.time() if now is None else now)), encoding='utf8')
        temp_json.replace(folder/'live.json')
        return True
    except OSError:
        return False


def start_preview(folder, workers, port):
    """Serve only the new report directory; do not open/activate any browser."""
    folder = Path(folder)
    tiles = ''.join(f'<figure><img id="i{i}" alt="Worker {i}: waiting for frame">'
                    f'<figcaption id="s{i}">Worker {i}: starting</figcaption></figure>' for i in range(workers))
    html = (Path(__file__).with_name('ppo_dashboard.html').read_text(encoding='utf8')
            .replace('__TILES__',tiles).replace('__WORKERS__',str(workers)))
    (folder/'index.html').write_text(html, encoding='utf8')
    class QuietHandler(SimpleHTTPRequestHandler):
        def handle(self):
            try:
                super().handle()
            except (ConnectionAbortedError,ConnectionResetError,BrokenPipeError):
                pass

        def log_message(self, *_):
            pass
    handler = functools.partial(QuietHandler, directory=str(folder.resolve()))
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name='worker-preview')
    thread.start()
    return server


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Serve a saved dashboard without starting training')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8001)
    args = parser.parse_args()
    if not args.report.is_dir():
        parser.error('Report directory must exist')
    workers = len(list(args.report.glob('worker_*')))
    if not workers:
        parser.error('No saved workers')
    server = start_preview(args.report,workers,args.port)
    print(f'Saved dashboard: http://127.0.0.1:{args.port}/',flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
