"""Fixed-policy 1/N-display throughput diagnostic. Never updates/saves a model.

Workers own separate browsers and inference copies. This is not shared PPO.
Use distinct extended displays; parent measurement excludes process/model load.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import uuid


async def identify_window(page, native):
    """Wait for Chromium's native title to catch up; never activate it."""
    import asyncio
    original_title = await page.title()
    marker = 'RS-background-' + uuid.uuid4().hex
    try:
        await page.evaluate('(v) => document.title=v', marker)
        for attempt in range(8):
            await asyncio.sleep(.1)
            try:
                return native.find(marker)
            except RuntimeError:
                if attempt == 7:
                    raise
    finally:
        await page.evaluate('(v) => document.title=v', original_title)


def combine(results, seconds):
    steps = sum(r.get('steps', 0) for r in results)
    stopped = any(r.get('error') or r.get('guard_hard_stopped') for r in results)
    return dict(status='STOPPED' if stopped else 'COMPLETE_FIXED_POLICY_DIAGNOSTIC',
                workers=len(results), measured_seconds=seconds, valid_decisions=steps,
                aggregate_decisions_per_second=steps/seconds if seconds > 0 and not stopped else None,
                partial_decisions_per_second=steps/seconds if seconds > 0 else None,
                model_updated=False, shared_learner=False, input_response_verified=False,
                note='Transport/motion evidence needs review; no claim of learning speedup or Unity input causality')


def worker(index, display, options, ready, start, deadline, stopped, folder):
    output = Path(folder)/f'worker_{index}'
    output.mkdir()
    result = dict(display=display, steps=0, error=None, guard_hard_stopped=False)
    raw = monitored = None
    handle = None
    import contextlib
    with (output/'worker.log').open('w', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                import cv2
                import numpy as np
                import torch
                from stable_baselines3 import PPO
                import train_rs_v612_ppo as trainer
                args = trainer.parse_args([
                    '--mode', 'eval', '--run-name', output.name, '--resume', options['checkpoint'],
                    '--eval-policy', 'deterministic', '--resolution', '512x288',
                    '--fps', '30', '--rate-mode', 'uncapped', '--press-delay-ms', '5',
                    '--reuse-observation-guard', '--state-model', options['state_model'],
                    '--detector', options['detector'], '--background-display', str(display),
                    '--background-capture', options.get('background_capture', 'mss'),
                    '--episode-max-seconds', '300',
                    *(['--canvas-policy-input'] if options['canvas_policy_input'] else [])])
                monitored, raw = trainer.create_live_env(args, output)
                model = PPO.load(options['checkpoint'], env=monitored,
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
                # Listener belongs to the game canvas document, not the user's browser.
                awaitable = raw.canvas.evaluate('''el => {
                    const w=el.ownerDocument.defaultView;
                    w.__rsBackgroundKeys={};
                    w.addEventListener('keydown', e => {
                        w.__rsBackgroundKeys[e.key]=(w.__rsBackgroundKeys[e.key]||0)+1;
                    }, true);
                    w.__rsBackgroundRaf={count:0,last:0};
                    const tick=t=>{w.__rsBackgroundRaf.count++;w.__rsBackgroundRaf.last=t;w.requestAnimationFrame(tick)};
                    w.requestAnimationFrame(tick);
                }''')
                raw._call(awaitable)
                native = None
                if __import__('sys').platform == 'win32':
                    owned = getattr(raw, '_background_native_window', None)
                    if owned is not None:
                        native, handle = owned.foreground_api, owned.handle
                    else:
                        from tools.ppo_browser_focus import WindowsForeground
                        native = WindowsForeground()
                        handle = raw._call(identify_window(raw.page, native))
                ready.set()
                if not start.wait(options['startup_timeout']):
                    raise RuntimeError('Parent never started measurement')
                if stopped.is_set():
                    return
                began = time.perf_counter()
                obs, _ = monitored.reset()
                renderer = raw._call(raw.canvas.evaluate('''el => {
                    const gl=el.getContext('webgl2')||el.getContext('webgl');
                    if(!gl)return null; const ext=gl.getExtension('WEBGL_debug_renderer_info');
                    return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
                }'''))
                print('[BACKGROUND WEBGL]', renderer, flush=True)
                result['webgl_renderer'] = renderer
                steps = episodes = polls = background_polls = moving_polls = 0
                previous = None
                last_poll = -1.
                with (output/'trace.jsonl').open('w', encoding='utf8') as trace:
                    while time.perf_counter() < deadline.value and not stopped.is_set():
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = monitored.step(int(action))
                        if time.perf_counter() >= deadline.value:
                            break  # Do not count a late completion outside the common interval.
                        steps += 1
                        result['steps'] = steps
                        now = time.perf_counter()
                        if now-last_poll >= .5:
                            last_poll = now
                            view = raw._motion_view(raw._v2_last_bgra, raw.crop)
                            delta = None if previous is None else float(np.abs(view.astype(float)-previous).mean())
                            previous = view.astype(float)
                            moving_polls += delta is not None and delta > 1.
                            background = None if native is None else not native.is_foreground(handle)
                            background_polls += background is True
                            polls += 1
                            keys = raw._call(raw.canvas.evaluate('el => el.ownerDocument.defaultView.__rsBackgroundKeys'))
                            raf = raw._call(raw.canvas.evaluate('''el => ({
                                ...el.ownerDocument.defaultView.__rsBackgroundRaf,
                                visibility:el.ownerDocument.visibilityState,
                                focused:el.ownerDocument.hasFocus()})'''))
                            sample = getattr(raw.grabber, 'last_sample', None)
                            native_frame_info = (dict(present_age_ms=(now-sample.present_at)*1000,
                                           transfer_age_ms=(now-sample.received_at)*1000,
                                           delivery_ms=(sample.received_at-sample.present_at)*1000,
                                           serial=sample.serial, fresh=sample.fresh) if sample is not None else None)
                            trace.write(json.dumps(dict(seconds=now-began, steps=steps,
                                action=int(action), reward=float(reward), background=background,
                                motion_delta=delta, canvas_keydowns=keys, browser_raf=raf,
                                native_frame=native_frame_info))+'\n')
                            if options.get('preview_port') and polls % 2 == 0:
                                from tools.ppo_worker_preview import write_preview
                                write_preview(output, obs['image'], dict(phase='fixed policy', steps=steps))
                            if polls <= 4 or (polls % 20 == 0):
                                cv2.imwrite(str(output/f'capture_{polls:04d}.jpg'), raw._v2_last_bgra)
                        if done or truncated:
                            episodes += 1
                            if time.perf_counter() < deadline.value and not stopped.is_set():
                                obs, _ = monitored.reset()
                result.update(episodes=episodes, wall_seconds=time.perf_counter()-began,
                              background_polls=background_polls, motion_polls=moving_polls, polls=polls,
                              canvas_keydowns=keys if polls else {},
                              guard=raw.hazard_report()['v612_live_input_guard'])
                result['guard_hard_stopped'] = raw._guard_hard_stopped
                result['capture_backend'] = options.get('background_capture', 'mss')
                if hasattr(raw.grabber, 'source'):
                    result.update(native_presentations=raw.grabber.source.serial,
                                  cached_ui_reads=raw.grabber.cached_ui_reads)
            except BaseException as exc:
                result['error'] = repr(exc)
                result['guard_hard_stopped'] = bool(getattr(raw, '_guard_hard_stopped', False))
                traceback.print_exc()
                stopped.set()
            finally:
                ready.set()
                if raw is not None:
                    import rs_v5_segmentation_live_benchmark as bench
                    result['stage_timings'] = {name: bench.as_report(getattr(raw, name, [])) for name in
                        ('capture_grab_ms','capture_alignment_ms','capture_menu_ms','pack_ms','game_crop_ms')}
                if monitored is not None:
                    monitored.close()
                elif raw is not None:
                    raw.close()
                (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--displays', nargs='+', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=90.)
    parser.add_argument('--startup-timeout', type=float, default=180.)
    parser.add_argument('--state-model', type=Path, default=Path('checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt'))
    parser.add_argument('--detector', type=Path, default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    parser.add_argument('--canvas-policy-input', action='store_true')
    parser.add_argument('--background-capture', choices=['mss','dxgi'], default='mss')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--preview-port', type=int, default=None,
                        help='Optional loopback thumbnail grid, e.g. 8001; one update/second')
    parser.add_argument('--telemetry', action='store_true', help='One-second CPU per-core/GPU samples outside the control loop')
    args = parser.parse_args(argv)
    if not 10 <= args.seconds <= 900 or not 10 <= args.startup_timeout <= 600:
        parser.error('Measurement must be 10..900 seconds; startup timeout 10..600 seconds')
    if len(args.displays) != len(set(args.displays)):
        parser.error('Workers must use distinct displays')
    if args.preview_port is not None and not 1024 <= args.preview_port <= 65535:
        parser.error('Preview port must be 1024..65535')
    for path in (args.checkpoint, args.state_model, args.detector):
        if not path.is_file():
            parser.error(f'Missing local asset: {path}')
    if args.report.exists():
        parser.error('Report exists; choose a new path')
    import mss
    from tools.ppo_background_display import select_display
    with mss.MSS() as capture:
        for index in args.displays:
            select_display(capture.monitors, index)
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.report.mkdir(parents=True)
    (args.report/'config.json').write_text(json.dumps(vars(args), indent=2, default=str), encoding='utf8')
    ctx = mp.get_context('spawn')
    start, stopped = ctx.Event(), ctx.Event()
    deadline = ctx.Value('d', 0.)
    processes, ready = [], []
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    results = []
    status = {'status': 'STARTING', 'source_sha256': source_hash}
    state_path = args.report/'benchmark.json'
    state_path.write_text(json.dumps(status, indent=2), encoding='utf8')
    began = None
    preview = None
    sampler = None
    try:
        if args.preview_port is not None:
            from tools.ppo_worker_preview import start_preview
            preview = start_preview(args.report, len(args.displays), args.preview_port)
            print(f'Worker grid: http://127.0.0.1:{args.preview_port}/', flush=True)
        for index, display in enumerate(args.displays):
            event = ctx.Event()
            process = ctx.Process(target=worker, args=(index, display, options, event, start,
                                                       deadline, stopped, str(args.report)))
            process.start()
            processes.append(process)
            ready.append(event)
        if args.telemetry:
            from tools.ppo_resource_sampler import ResourceSampler
            sampler = ResourceSampler(args.report/'system_usage.jsonl', [p.pid for p in processes])
            sampler.start()
        startup_end = time.perf_counter()+args.startup_timeout
        for event in ready:
            if not event.wait(max(0., startup_end-time.perf_counter())) or stopped.is_set():
                raise RuntimeError('Worker startup failed/timed out; inspect worker.log')
        began = time.perf_counter()
        deadline.value = began+args.seconds
        start.set()
        state_path.write_text(json.dumps(dict(status, status='MEASURING'), indent=2), encoding='utf8')
        print(f'Measuring {len(processes)} workers for {args.seconds:g}s. Use your normal browser now.', flush=True)
        for process in processes:
            process.join(timeout=max(0., deadline.value-time.perf_counter())+60.)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError('Worker failed or failed to stop; inspect worker.log')
        results = [json.loads((args.report/f'worker_{i}'/'result.json').read_text()) for i in range(len(processes))]
        status.update(combine(results, args.seconds), results=results)
    except BaseException as exc:
        status.update(status='STOPPED', error=repr(exc))
        raise
    finally:
        stopped.set()
        start.set()
        for process in processes:
            process.join(timeout=5.)
            if process.is_alive():
                # Last resort after graceful signalling; own diagnostic worker only.
                process.terminate()
                process.join(timeout=5.)
        status['source_unchanged'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == source_hash
        state_path.write_text(json.dumps(status, indent=2), encoding='utf8')
        if preview is not None:
            preview.shutdown()
            preview.server_close()
        if sampler is not None:
            sampler.close()
    print(json.dumps(status, indent=2))
    if status['status'] == 'STOPPED' or not status['source_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
