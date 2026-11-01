"""Isolated road-vision experiments. No changes to production or temporal sources."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TEMPORAL=HERE.parent/'temporal_history'
for folder in (ROOT,TEMPORAL):
    if str(folder) not in sys.path:sys.path.append(str(folder))

from experiment_config import configuration as temporal_configuration,source_hashes as temporal_hashes

CROPS={'full':(0.,0.,1.,1.),'road_wide':(.15,0.,.85,1.),'road_center':(.25,0.,.75,1.)}
PRESETS={
    'full512_tiny':('full',(512,288),'tiny'),
    'full128_tiny':('full',(128,128),'tiny'),
    'road128_tiny':('road_wide',(128,128),'tiny'),
    'road192_tiny':('road_wide',(192,192),'tiny'),
    'road128_spatial':('road_wide',(128,128),'spatial'),
    'road192_spatial':('road_wide',(192,192),'spatial'),
    'road256_spatial':('road_wide',(256,256),'spatial'),
    'road384_spatial':('road_wide',(384,384),'spatial'),
    'road96_spatial':('road_wide',(96,96),'spatial'),
    'center128_spatial':('road_center',(128,128),'spatial')}
DEFAULT_ARMS=list(PRESETS)[:8]

def configuration(arm,frames=4,span_ms=500,capture_hz=60,decision_hz=30,spacing='uniform'):
    crop,size,encoder=PRESETS[arm]
    from history_spacing import offsets
    if spacing!='uniform':offsets(spacing,frames,span_ms/1000)
    result=dict(version=1,arm=arm,crop=crop,roi=list(CROPS[crop]),size=list(size),encoder=encoder,
                temporal=temporal_configuration(frames,span_ms,capture_hz,decision_hz),
                resize='area_letterbox',detector_mode='off')
    if spacing!='uniform':result['temporal']['spacing']=spacing
    return result

def source_hashes():
    result=temporal_hashes()
    for p in sorted(HERE.glob('*.py')):
        result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    capture=HERE.parent/'capture_pipeline'
    for p in [*capture.glob('*.py'),*capture.glob('*.cjs'),
              ROOT/'rankedsurfers-downloaded/game/releases/ae44e7782a4c8bbe5b10/arena.html']:
        result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    from tools.ppo_run_integrity import atomic_json
    atomic_json(Path(path),value)
