"""Offline framing comparison; never edits models, recordings or runtime defaults."""
import json
from pathlib import Path
import cv2
import numpy as np
import rs_v5_segmentation_live_benchmark as bench


def main():
    output = Path('reports/framing_audit_20261030_02')
    output.mkdir(exist_ok=False)
    root = Path('reports/headless_six_repeat_20261030_01/worker_0')
    desktop = cv2.imread(str(root/'background_display.png'))
    canvas = cv2.imread(str(root/'background_canvas.png'))
    saved = cv2.imread(str(root/'background_policy.png'))
    reconstructed = bench.fit_rgb_to_target(cv2.cvtColor(bench.crop_game(canvas,bench.DEFAULT_CROP),cv2.COLOR_BGR2RGB),
                                           (512,288),cv2.INTER_LINEAR)
    actual_saved_headless_equal = bool(np.array_equal(cv2.cvtColor(reconstructed,cv2.COLOR_RGB2BGR),saved))
    # Use one gameplay image for the comparison; keep independent exact-source
    # validation of the saved worker policy image above.
    canvas = cv2.imread('reports/headless_prototype_20261030_two_01/worker_0/frame_0020.jpg')
    desktop = np.zeros((1080,1920,3),dtype=np.uint8)
    desktop[87:1024,8:1912] = canvas
    sources = [('Legacy fractions on padded desktop (simulation)', desktop, bench.DEFAULT_CROP),
               ('Current headless: fractions inside canvas', canvas, bench.DEFAULT_CROP),
               ('Candidate: whole canvas, letterboxed', canvas, (0,0,1,1))]
    rows, facts = [], []
    for index, (title, source, crop) in enumerate(sources):
        rgb = cv2.cvtColor(bench.crop_game(source,crop),cv2.COLOR_BGR2RGB)
        policy = cv2.cvtColor(bench.fit_rgb_to_target(rgb,(512,288),cv2.INTER_LINEAR),cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output/f'{index}_policy.png'),policy)
        panel = np.zeros((325,512,3),np.uint8)
        panel[37:] = policy
        cv2.putText(panel,title,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        rows.append(panel)
        facts.append(dict(title=title,source_shape=list(source.shape),crop_rect=bench.crop_rect(source.shape,crop),
                          native_shape=list(rgb.shape)))
    cv2.imwrite(str(output/'same_frame_comparison.png'),np.concatenate(rows,axis=0))
    legacy_path = Path('ai_data/rl_v2_runs/v612_largebc_defaults_3h_20261029_204226/matched_detection_previews/00_source.jpg')
    legacy = cv2.imread(str(legacy_path))
    legacy_policy = bench.fit_rgb_to_target(cv2.cvtColor(legacy,cv2.COLOR_BGR2RGB),(512,288),cv2.INTER_LINEAR)
    cv2.imwrite(str(output/'historical_detector_source.jpg'),legacy)
    cv2.imwrite(str(output/'historical_policy_reconstructed.png'),cv2.cvtColor(legacy_policy,cv2.COLOR_RGB2BGR))
    manifests = []
    for folder in Path('ai_data/demonstrations_v13/runs').glob('demo_*'):
        path = folder/'manifest.json'
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding='utf8'))
        manifests.append(dict(run=folder.name,status=m.get('status'),
            region=m.get('capture',{}).get('region'),
            preview_shape=m.get('preview_canvas_original_shape'),
            preview_exists=(folder/'preview.mp4').exists(),
            frames_exists=(folder/'frames.npz').exists()))
    folder = Path('ai_data/demonstrations_v13/runs/demo_20261028_145438_471297')
    with np.load(folder/'frames.npz',allow_pickle=False) as archive:
        frame = archive['frames'][100].transpose(1,2,0)
    cv2.imwrite(str(output/'bc_stored_frame_100.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    video = cv2.VideoCapture(str(folder/'preview.mp4'))
    video_info = dict(frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),fps=video.get(cv2.CAP_PROP_FPS),
                      width=video.get(cv2.CAP_PROP_FRAME_WIDTH),height=video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.set(cv2.CAP_PROP_POS_FRAMES,100)
    ok, image = video.read()
    video.release()
    if ok:
        cv2.imwrite(str(output/'bc_preview_video_frame_100.jpg'),image)
    from ultralytics import YOLO
    detector = YOLO('ai_data/v5_hazards/runs/segment_round04/weights/best.pt')
    report = dict(comparisons=facts,actual_saved_headless_equal=actual_saved_headless_equal,historical_native_shape=list(legacy.shape),
                  bc_manifests=manifests,bc_example_video=video_info,
                  detector_train_args=detector.ckpt.get('train_args',{}),
                  caveats=['Same-frame desktop panel uses black padding, not historical Chrome/taskbar pixels.',
                           'Historical policy reconstructed from JPEG source, not original tensor.',
                           'Video frame index does not establish exact alignment to BC sample index.',
                           'No classifier thresholds, weights or default capture paths changed.'])
    (output/'audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
    print(json.dumps(dict(output=str(output),comparisons=facts,manifests=len(manifests),
                         historical_native_shape=list(legacy.shape),bc_video=video_info,
                         detector_data=report['detector_train_args'].get('data')),indent=2))


if __name__ == '__main__':
    main()
