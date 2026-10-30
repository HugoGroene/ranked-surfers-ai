"""Experimental timestamped DXGI source, opt-in through background-capture dxgi.

Fresh gameplay reads never silently reuse a cached frame. Static UI inspection
must explicitly opt in, and still receives the original presentation timestamp.
Integration must preserve those timestamps through the guard/observation stack.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True)
class NativeFrame:
    image: np.ndarray
    present_at: float
    received_at: float
    serial: int
    fresh: bool


class TimestampedDxgiSource:
    def __init__(self, camera, width, height, *, clock=time.perf_counter, sleep=time.sleep):
        self.camera = camera
        self.width, self.height = width, height
        self.clock, self.sleep = clock, sleep
        self.last = None
        self.serial = 0

    def read(self, *, timeout=.1, max_age=.05, allow_static_ui=False):
        deadline = self.clock()+timeout
        while True:
            image = self.camera.grab(new_frame_only=True, copy=True)
            now = self.clock()
            if image is not None and self.camera._duplicator.accumulated_frames > 0:
                if image.shape != (self.height, self.width, 4) or image.dtype != np.uint8:
                    raise RuntimeError('DXGI image format/geometry changed')
                # Pinned DXcam 0.3.0 exposes the one-shot backend QPC timestamp
                # here; its public latest_frame_time applies to buffered mode.
                present = float(self.camera._duplicator.latest_frame_time)
                if not np.isfinite(present) or present <= 0 or present > now+.02:
                    raise RuntimeError('Invalid DXGI presentation clock')
                if self.last is not None and present < self.last.present_at:
                    raise RuntimeError('DXGI presentation clock regressed')
                distinct = self.last is None or present > self.last.present_at
                if distinct:
                    self.serial += 1
                    self.last = NativeFrame(image, present, now, self.serial, now-present <= max_age)
                    if self.last.fresh:
                        return self.last
            if allow_static_ui and self.last is not None:
                return replace(self.last, fresh=False)
            if now >= deadline:
                return None
            self.sleep(min(.002, max(0.,deadline-now)))

    def close(self):
        self.camera.release()


def open_dxgi(display):
    dependencies = Path(__file__).resolve().parents[1]/'development/local/dxcam_030'
    if not (dependencies/'dxcam').is_dir():
        raise RuntimeError('Install the isolated optional capture dependencies first')
    sys.path.insert(0, str(dependencies))
    import dxcam
    import importlib.metadata
    if importlib.metadata.version('dxcam') != '0.3.0':
        raise RuntimeError('This adapter requires pinned DXcam 0.3.0')
    expected = tuple(display[k] for k in ('left','top','width','height'))
    matches = []
    for device_index, outputs in enumerate(dxcam.__factory.outputs):
        for output_index, output in enumerate(outputs):
            rect = output.desc.DesktopCoordinates
            if ((rect.left,rect.top,rect.right-rect.left,rect.bottom-rect.top) == expected
                    and output.attached_to_desktop):
                matches.append((device_index,output_index))
    if len(matches) != 1:
        raise RuntimeError('No unique DXGI output for the selected physical rectangle')
    device, output = matches[0]
    camera = dxcam.create(device_idx=device, output_idx=output, output_color='BGRA', backend='dxgi')
    return TimestampedDxgiSource(camera, display['width'], display['height'])


class DxgiDisplayGrabber:
    """MSS-shaped interface; GDI is used for monitor enumeration only."""
    def __init__(self, display_index):
        import mss
        from tools.ppo_background_display import rectangle, select_display
        self.inventory = mss.MSS()
        self.monitors = self.inventory.monitors
        self.display = rectangle(select_display(self.monitors, display_index))
        self.source = open_dxgi(self.display)
        self.last_sample = None
        self.require_new = False
        self.cached_ui_reads = 0

    def grab(self, region):
        from tools.ppo_background_display import rectangle
        if rectangle(region) != self.display:
            raise RuntimeError('DXGI grab requested a different display/rectangle')
        require_new, self.require_new = self.require_new, False
        sample = self.source.read(timeout=.25, allow_static_ui=not require_new)
        if sample is None:
            raise RuntimeError('No DXGI frame available; no cached fallback to another backend')
        self.last_sample = sample
        self.cached_ui_reads += not sample.fresh
        return sample.image

    def close(self):
        try:
            self.source.close()
        finally:
            self.inventory.close()
