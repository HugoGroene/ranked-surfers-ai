"""Capture-only capability/timing probe: no browser, game, inputs or image files.

DXcam is isolated under development/local/dxcam_030, not installed into the
original environment. Private output enumeration is pinned to DXcam 0.3.0 and
matched by physical rectangle; an MSS index is never treated as a DXGI index.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--display', type=int, required=True)
    parser.add_argument('--backend', choices=['mss','dxgi'], required=True)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 10:
        parser.error('Capture-only probe must be at most ten seconds')
    if args.report.exists():
        parser.error('Report exists; choose a new file')
    import mss
    from tools.ppo_background_display import select_display
    inventory = mss.MSS()
    selected = select_display(inventory.monitors, args.display)
    expected = (selected['left'], selected['top'], selected['width'], selected['height'])
    camera = None
    result = dict(backend=args.backend, display=args.display, rectangle=expected,
                  image_files_saved=False, no_game_or_inputs=True)
    try:
        if args.backend == 'dxgi':
            dependencies = Path(__file__).resolve().parents[1]/'development/local/dxcam_030'
            if not (dependencies/'dxcam').is_dir():
                raise RuntimeError('Isolated DXcam 0.3.0 dependency is not installed')
            sys.path.insert(0, str(dependencies))
            import dxcam
            import importlib.metadata
            version = importlib.metadata.version('dxcam')
            if version != '0.3.0':
                raise RuntimeError(f'Expected pinned DXcam 0.3.0; got {version}')
            result.update(version=version, devices=dxcam.device_info(), outputs=dxcam.output_info())
            matches = []
            for device_index, outputs in enumerate(dxcam.__factory.outputs):
                for output_index, output in enumerate(outputs):
                    rect = output.desc.DesktopCoordinates
                    actual = (rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top)
                    if actual == expected and output.attached_to_desktop:
                        matches.append((device_index, output_index))
            if len(matches) != 1:
                raise RuntimeError(f'Expected exactly one DXGI output matching the virtual display; found {matches}')
            device_index, output_index = matches[0]
            camera = dxcam.create(device_idx=device_index, output_idx=output_index,
                                  output_color='BGRA', backend='dxgi')
            grab = lambda: camera.grab(new_frame_only=True)
        else:
            grab = lambda: inventory.grab(selected)
        timings = []
        frame_timings = []
        available = empty = 0
        began = time.perf_counter()
        while time.perf_counter()-began < args.seconds:
            start = time.perf_counter()
            image = grab()
            timings.append((time.perf_counter()-start)*1000.)
            if image is None:
                empty += 1
            else:
                available += 1
                frame_timings.append(timings[-1])
            time.sleep(.02)  # no busy loop or attempt to drive DWM at maximum rate
        result.update(status='CAPTURE_PROBE_COMPLETE', duration=time.perf_counter()-began,
                      available_frames=available, no_new_frame=empty,
                      median_call_ms=statistics.median(timings), max_call_ms=max(timings),
                      median_available_frame_call_ms=statistics.median(frame_timings) if frame_timings else None,
                      calls_over_100ms=sum(t > 100 for t in timings),
                      freshness='new-present-only' if camera is not None else 'desktop-read; present freshness unverified',
                      note='A static desktop may have no new DXGI presents. This is not a gameplay throughput benchmark.')
    except BaseException as exc:
        result.update(status='CAPTURE_PROBE_FAILED', error=repr(exc))
        raise
    finally:
        if camera is not None:
            camera.release()
        inventory.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
