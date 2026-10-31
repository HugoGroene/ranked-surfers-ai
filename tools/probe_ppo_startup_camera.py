"""Record one guarded startup and two passive seconds; no movement or policy."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import time


def main():
    import cv2
    import train_rs_v612_ppo as trainer
    from tools.ppo_headless_env import install
    folder=Path('reports')/('startup_camera_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True,exist_ok=False)
    args=trainer.parse_args(['--mode','probe','--background-display','1',
        '--canvas-policy-input','--framing-profile','canvas_v1','--detector-mode','off',
        '--state-model','checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt'])
    install(trainer,'electron',electron_transport='shared_memory')
    env=raw=None;frames=[];reset_at=None;error=None
    try:
        env,raw=trainer.create_live_env(args,folder)
        capture=raw._capture
        async def record():
            result=await capture()
            at=time.perf_counter()
            if len(frames)<600:
                pixels=cv2.resize(raw._v2_last_bgra,(960,540))[:,:,:3].copy()
                frames.append((at,pixels))
            return result
        raw._capture=record
        env.reset()
        reset_at=time.perf_counter()
        async def observe():
            deadline=time.perf_counter()+2
            while time.perf_counter()<deadline:
                await raw._capture()
                await asyncio.sleep(.02)
        raw._call(observe(),timeout=10)
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        try:
            if env is not None:env.close()
        finally:
            records=[]
            for i,(at,pixels) in enumerate(frames):
                filename=f'frame_{i:04d}.jpg'
                cv2.imwrite(str(folder/filename),pixels)
                records.append(dict(file=filename,at=at,since_reset_ms=(at-reset_at)*1000 if reset_at else None))
            (folder/'frames.json').write_text(json.dumps(dict(frames=records,error=error,
                note='Capture-completion timestamps; reset completion is not exact text disappearance. No movement inputs.'),indent=2))
            print(f'RECORDING: {folder}',flush=True)


if __name__=='__main__':
    main()
