"""Optional loopback-only grid of existing policy frames; never captures again."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2


def previews_enabled(folder):
    try:
        return json.loads((Path(folder)/'preview_control.json').read_text(encoding='utf8')).get('enabled',True) is not False
    except (OSError,ValueError):
        return True


def write_status(folder,metadata):
    try:
        temporary=Path(folder)/'live.partial.json'
        temporary.write_text(json.dumps(dict(metadata,updated_at=time.time())),encoding='utf8')
        temporary.replace(Path(folder)/'live.json')
        return True
    except OSError:
        return False


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
        def do_POST(self):
            if urlsplit(self.path).path != '/preview-control':
                self.send_error(404)
                return
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                self.send_error(403)
                return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 128:
                    raise ValueError('Invalid length')
                control=json.loads(self.rfile.read(length))
                if type(control.get('enabled')) is not bool:
                    raise ValueError('Expected boolean')
                tmp=folder/'preview_control.partial.json'
                tmp.write_text(json.dumps(dict(enabled=control['enabled'])),encoding='utf8')
                tmp.replace(folder/'preview_control.json')
            except (OSError,ValueError,TypeError):
                self.send_error(400)
                return
            self.send_response(204)
            self.end_headers()

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
