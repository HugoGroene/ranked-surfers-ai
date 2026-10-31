"""Optional loopback-only grid of existing policy frames; never captures again."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np


def previews_enabled(folder):
    return preview_control(folder)['enabled']


def preview_control(folder, default_fps=1):
    try:
        value=json.loads((Path(folder)/'preview_control.json').read_text(encoding='utf8'))
        fps=value.get('fps',default_fps)
        if isinstance(fps,bool) or not isinstance(fps,(int,float)) or not 1<=fps<=30:
            fps=default_fps
        return dict(enabled=value.get('enabled',True) is not False,fps=float(fps))
    except (OSError,ValueError,AttributeError,TypeError):
        return dict(enabled=True,fps=float(default_fps))


class PreviewPublisher:
    """One replaceable pending thumbnail; encoding never blocks game control."""
    def __init__(self,folder,fps=1):
        self.folder=Path(folder)
        self.control=dict(enabled=True,fps=float(fps))
        self.checked=self.last_offer=self.last_status=0.
        self.pending=None
        self.lock=threading.Lock()
        self.wake=threading.Event()
        self.stopped=False
        self.published=self.dropped=0
        self.last_image_at=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='ppo-preview-encoder')
        self.thread.start()

    def offer(self,image,metadata,*,now=None,force=False):
        now=time.perf_counter() if now is None else now
        if now-self.checked>=.25:
            self.control=preview_control(self.folder.parent,self.control['fps'])
            self.checked=now
        if not self.control['enabled']:
            with self.lock:self.pending=None
            if now-self.last_status>=1 or force:
                write_status(self.folder,dict(metadata,images_enabled=False,image_updated_at=self.last_image_at))
                self.last_status=now
            return
        if not force and now-self.last_offer<1/self.control['fps']:
            return
        self.last_offer=now
        # Retain only the latest RGB frame, never the full four-frame stack.
        frame=np.array(image[-3:],copy=True)
        with self.lock:
            self.dropped+=self.pending is not None
            self.pending=(frame,dict(metadata,images_enabled=True),time.time())
        self.wake.set()

    def _run(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            with self.lock:
                item,self.pending=self.pending,None
                stopped=self.stopped
            if item is not None:
                frame,metadata,stamp=item
                if preview_control(self.folder.parent,self.control['fps'])['enabled']:
                    if write_preview(self.folder,frame,dict(metadata,image_updated_at=stamp),now=stamp):
                        self.published+=1
                        self.last_image_at=stamp
            if stopped:return

    def close(self):
        with self.lock:self.stopped=True
        self.wake.set()
        self.thread.join(timeout=3)


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


def start_preview(folder, workers, port, fps=1):
    """Serve only the new report directory; do not open/activate any browser."""
    folder = Path(folder)
    if not (folder/'preview_control.json').exists():
        (folder/'preview_control.json').write_text(json.dumps(dict(enabled=True,fps=fps)),encoding='utf8')
    control_lock=threading.Lock()
    tiles = ''.join(f'<figure><img id="i{i}" alt="Worker {i}: waiting for frame">'
                    f'<figcaption id="s{i}">Worker {i}: starting</figcaption></figure>' for i in range(workers))
    html = (Path(__file__).with_name('ppo_dashboard.html').read_text(encoding='utf8')
            .replace('__TILES__',tiles).replace('__WORKERS__',str(workers)).replace('__PREVIEW_FPS__',str(fps)))
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
                if not isinstance(control,dict):raise ValueError('Expected control object')
                if type(control.get('enabled')) is not bool:
                    raise ValueError('Expected boolean')
                fps=control.get('fps',preview_control(folder)['fps'])
                if isinstance(fps,bool) or not isinstance(fps,(int,float)) or not 1<=fps<=30:
                    raise ValueError('Preview fps must be 1..30')
                with control_lock:
                    tmp=folder/'preview_control.partial.json'
                    tmp.write_text(json.dumps(dict(enabled=control['enabled'],fps=fps)),encoding='utf8')
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
