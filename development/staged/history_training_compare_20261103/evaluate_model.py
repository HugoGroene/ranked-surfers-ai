"""Matched single-worker evaluation of the fixed4/fixed5 comparison checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from parallel_adapter import ROOT, LAYOUTS, install, configuration, sources


def install_visual_preview(trainer, layout):
    """Display existing captured RGB frames; never request extra game captures."""
    import time
    import cv2
    original = trainer.create_live_env
    def create(*args, **kwargs):
        monitored, raw = original(*args, **kwargs)
        title = f'{layout} evaluation - full game preview (ESC stops)'
        last = [0.0]
        opened = [False]
        def show():
            now = time.perf_counter()
            if now - last[0] < 1 / 15:
                return
            frame = getattr(raw, 'last_source_native', None)
            if frame is None:
                return
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (960, max(1, round(height * 960 / width))),
                               interpolation=cv2.INTER_AREA)
            cv2.imshow(title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            opened[0] = True
            last[0] = now
            if cv2.waitKey(1) & 0xff == 27:
                raise KeyboardInterrupt
        step, reset, close = monitored.step, monitored.reset, monitored.close
        def visual_step(*args, **kwargs):
            result = step(*args, **kwargs)
            show()
            return result
        def visual_reset(*args, **kwargs):
            result = reset(*args, **kwargs)
            show()
            return result
        def visual_close():
            try:
                return close()
            finally:
                if opened[0]:
                    cv2.destroyWindow(title)
        monitored.step, monitored.reset, monitored.close = visual_step, visual_reset, visual_close
        return monitored, raw
    trainer.create_live_env = create


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', choices=LAYOUTS, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--minutes', type=float, default=180)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--state-model',type=Path,default=ROOT/'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt')
    view = p.add_mutually_exclusive_group()
    view.add_argument('--visible-browser', action='store_true',
                   help='Open the actual game in visible Chromium; foreground desktop capture')
    view.add_argument('--visual', action='store_true',
                   help='Show captured gameplay at up to 15 FPS; adds display overhead')
    opt = p.parse_args()
    opt.checkpoint = opt.checkpoint.resolve()
    opt.output = opt.output.resolve()
    if not opt.checkpoint.is_file():
        p.error('Checkpoint does not exist')
    if not opt.output.is_relative_to(ROOT / 'reports') or opt.output.exists():
        p.error('Choose a new output directory under reports')
    if opt.episodes < 1 or not 0 < opt.minutes <= 2880:
        p.error('Invalid episode count or time limit')
    os.chdir(ROOT)
    os.environ['RS_COMPARISON_LAYOUT'] = opt.layout
    trainer = install()
    if opt.visible_browser:
        from visible_native import install as install_visible_browser
        install_visible_browser()
        from visible_startup import install as install_ready_start
        install_ready_start(trainer)
        from visible_episode_trace import install as install_episode_trace
        install_episode_trace(trainer)
    from run_trial import trainer_args
    from tools.ppo_observation_contract import requested_contract, validate_checkpoint
    from tools.ppo_run_integrity import atomic_json
    options = SimpleNamespace(mode='eval', output=opt.output, checkpoint=opt.checkpoint,
        episodes=opt.episodes, minutes=opt.minutes, seed=123, decision_hz=30,
        state_model=opt.state_model.resolve())
    a = trainer_args(trainer, options)
    contract = requested_contract(a)
    validate_checkpoint(opt.checkpoint, contract)
    if opt.check_only:
        from stable_baselines3 import PPO
        model = PPO.load(str(opt.checkpoint), device='cpu')
        assert list(model.observation_space['image'].shape) == contract['image_shape']
        assert model.observation_space['aux'].shape == (contract['aux_dim'],)
        print(f'PREFLIGHT PASS: {opt.layout}; image={contract["image_shape"]}; '
              f'aux={contract["aux_dim"]}; saved_steps={model.num_timesteps}', flush=True)
        return
    if opt.visual:
        install_visual_preview(trainer, opt.layout)
    before = hashlib.sha256(opt.checkpoint.read_bytes()).hexdigest()
    manifest = sources()
    import train_rs_v2_verify as v2
    v2.RUN_ROOT = opt.output.parent
    opt.output.mkdir(parents=True)
    # evaluate() requires an empty destination and writes its own configuration.
    try:
        success = trainer.evaluate(a)
    finally:
        unchanged = hashlib.sha256(opt.checkpoint.read_bytes()).hexdigest() == before
        atomic_json(opt.output / 'comparison_eval.json', dict(layout=opt.layout,
            configuration=configuration(), contract=contract,
            checkpoint=str(opt.checkpoint), checkpoint_sha256=before,
            checkpoint_unchanged=unchanged, source_unchanged=sources() == manifest,
            requested_episodes=opt.episodes, workers=1, deterministic=True,
            dashboard=False, detector_mode='off', visual_preview=opt.visual,
            visible_browser=opt.visible_browser,
            latency_pipeline=dict(reuse_pump_crop=os.environ.get('RS_REUSE_PUMP_CROP')=='1',
                                  defer_startup_previews=os.environ.get('RS_DEFER_STARTUP_PREVIEWS')=='1',
                                  threaded_preprocess=os.environ.get('RS_PUMP_PREPROCESS_THREAD')=='1'),
            capture_backend=('visible_desktop_mss' if os.environ.get('RS_VISIBLE_CAPTURE','desktop')=='desktop'
                             else 'visible_native_bitmap_shared_memory') if opt.visible_browser else 'gpu_100',
            preview_note='Visible modes are qualitative checks; timing can differ from headless'))
    if not unchanged or sources() != manifest:
        raise RuntimeError('Checkpoint or sources changed during evaluation')
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
