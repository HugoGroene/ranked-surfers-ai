"""Bounded saving of the three startup policy diagnostics, outside action work."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import cv2
import numpy as np


class PreviewWriter:
    NAMES={'canvas_v1_native.png','canvas_v1_policy.png','road_crop_bounds.png'}

    def __init__(self,folder,write=cv2.imwrite):
        self.folder=Path(folder).resolve()
        self.write=write
        self.jobs={}
        self.closed=False
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='startup-previews')
        # Allocate its thread during environment setup, before the first reset.
        self.executor.submit(lambda:None).result()

    def submit(self,path,image):
        path=Path(path).resolve()
        if self.closed:raise RuntimeError('Preview writer closed')
        if path.parent!=self.folder or path.name not in self.NAMES:
            raise ValueError('Only the three startup policy previews may be deferred')
        if path in self.jobs or path.exists():raise FileExistsError(str(path))
        # Never retain a mutable policy/capture array in the IO thread.
        owned=np.array(image,copy=True,order='C')
        def save():
            began=time.perf_counter()
            if not self.write(str(path),owned):raise OSError('Preview write failed: '+str(path))
            return (time.perf_counter()-began)*1000
        self.jobs[path]=self.executor.submit(save)
        return True # Accepted; completion/failure is checked at shutdown.

    def close(self):
        if self.closed:return
        self.closed=True
        self.executor.shutdown(wait=True)
        saved={};errors={}
        for path,future in self.jobs.items():
            try:saved[path.name]=future.result()
            except Exception as exc:errors[path.name]=repr(exc)
        (self.folder/'startup_previews.json').write_text(json.dumps(dict(
            status='FAILED' if errors else 'COMPLETE',saved_write_ms=saved,errors=errors,
            note='Three owned diagnostic snapshots only; policy pixels and guards unchanged. All writes joined at cleanup.'),indent=2))
        if errors:raise OSError('Startup preview saving failed: '+repr(errors))


def install(trainer):
    if getattr(trainer,'_deferred_policy_previews_installed',False):return
    trainer._deferred_policy_previews_installed=True
    original=trainer.create_live_env
    def create(*args,**kwargs):
        monitored,raw=original(*args,**kwargs)
        writer=PreviewWriter(raw.output_dir)
        raw._policy_preview_writer=writer.submit
        close=raw._close_browser
        async def finish():
            try:return await close()
            finally:writer.close()
        raw._close_browser=finish
        return monitored,raw
    trainer.create_live_env=create
