"""Create an isolated recorder snapshot; never rewrite production files."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def build():
    source = ROOT / 'rs_v612_demo.py'
    text = source.read_text(encoding='utf-8')
    replacements = [
        ("'.jpg', full_canvas_bgr,\n                            [cv2.IMWRITE_JPEG_QUALITY, 89]",
         "'.png', full_canvas_bgr,\n                            [cv2.IMWRITE_PNG_COMPRESSION, 1]"),
        ("ROOT = Path('ai_data/demonstrations_v1')", "ROOT = Path('ai_data/demonstrations_native_v1')"),
        ("np.savez_compressed(output/'frames.npz', frames=frames)",
         "from archive import save_sources\n    source_archive = save_sources(output, samples)\n    np.savez_compressed(output/'frames.npz', frames=frames)"),
        ("'schema': 1, 'status': status, 'feature_version': FEATURE_VERSION, 'preview_video': video,",
         "'source_archive': source_archive,\n        'schema': 1, 'status': status, 'feature_version': FEATURE_VERSION, 'preview_video': video,"),
        ("if status == 'approved' and not segments:",
         "if status != 'candidate':\n        raise ValueError('Use the staged review CLI for explicit visual approval')\n    if status == 'approved' and not segments:"),
        ("    file = folder/'manifest.json'\n    data = json.loads(file.read_text(encoding='utf-8'))",
         "    if status == 'approved':\n        raise ValueError('Use dataset.py review --reviewed --segments after inspecting the preview')\n    file = folder/'manifest.json'\n    data = json.loads(file.read_text(encoding='utf-8'))"),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f'Snapshot patch anchor not unique: {old[:70]}')
        text = text.replace(old, new)
    destination = HERE / 'recorder_snapshot.py'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(text, encoding='utf-8')
    (HERE/'snapshot_provenance.json').write_text(json.dumps({
        'source': 'rs_v612_demo.py', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'snapshot_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'changes': ['lossless native PNG archive', 'separate output root', 'explicit offline visual approval'],
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build()
