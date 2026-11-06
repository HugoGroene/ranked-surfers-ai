"""Isolated BC tooling. Imports are read-only; never modify an active trainer."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for folder in (ROOT,ROOT/'development/experiments/road_vision',
               ROOT/'development/staged/history_training_compare_20261103',
               ROOT/'development/experiments/capture_pipeline',
               ROOT/'development/staged/map_import_20261102'):
    if str(folder) not in sys.path:sys.path.append(str(folder))
DATA=ROOT/'ai_data/multimap_bc_v1'
MAPS=json.loads((ROOT/'development/experiments/multimap_classifier/maps.json').read_text())['maps']
from development.experiments.multimap_classifier.classifier_config import classifier_path
CLASSIFIER=classifier_path()
ACTIONS=['NONE','LEFT','RIGHT','JUMP','ROLL']
KEYS={'ArrowLeft':1,'ArrowRight':2,'ArrowUp':3,'ArrowDown':4,
      'KeyA':1,'KeyD':2,'KeyW':3,'KeyS':4}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def contract(checkpoint):
    with zipfile.ZipFile(checkpoint) as z:c=json.loads(z.read('data'))['observation_contract']
    if (c.get('aux_dim')!=789 or c.get('image_shape',[0])[0]!=12 or c.get('detector_mode')!='off'
            or c.get('command_history',{}).get('slots')!=32):
        raise ValueError('Requires current fixed4 + 32-command-history detector-off checkpoint')
    t=c['road_vision']['temporal']
    if t.get('recent_frames')!=1 or t.get('context_offsets_seconds')!=[.05,.25,.75]:
        raise ValueError('Expected fixed ages [0, .05, .25, .75]')
    return c


def require_idle(*,allow_classifier_training=False):
    """Recorder/BC CLI must not contend with the current training or evaluation."""
    import psutil
    current=psutil.Process()
    own={current.pid}
    # Windows venv python.exe can remain alive as the base interpreter's
    # launcher with identical arguments. Exclude only that exact ancestor
    # chain; a second recorder with identical arguments must still block.
    command=current.cmdline()[1:]
    parent=current.parent()
    while parent is not None:
        try:
            if not parent.name().lower().startswith('python') or parent.cmdline()[1:]!=command:break
            own.add(parent.pid)
            parent=parent.parent()
        except psutil.Error:
            break
    active=[]
    names={'train_ppo_parallel.py','stability_case.py','run_training.py','evaluate.py',
           'evaluate_model.py','latency_probe.py','stability.py','record.py','train.py'}
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid in own or 'python' not in (p.info['name'] or '').lower():continue
        args=p.info['cmdline'] or []
        if allow_classifier_training and any(
                Path(arg).suffix=='.py' and Path(arg).resolve()==ROOT/'development/experiments/multimap_classifier/train.py'
                for arg in args[1:]):continue
        if any(Path(arg).name in names for arg in args[1:]):active.append(p.pid)
    if active:raise RuntimeError(f'Stop training/evaluation/recording first. Active Python PIDs: {active}')
