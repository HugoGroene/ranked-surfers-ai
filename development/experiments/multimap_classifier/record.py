"""Manual local recording, retaining map-specific full guard images for the old labeller."""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
from classifier_config import classifier_path

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'ai_data/multimap_screen_states'
REGISTRY=json.loads(Path(__file__).with_name('maps.json').read_text())
MAPS={name:ROOT/spec['root'] for name,spec in REGISTRY['maps'].items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',choices=MAPS,required=True)
    p.add_argument('--seconds',type=int,default=120)
    p.add_argument('--split',choices=('train','validation'),required=True)
    a=p.parse_args()
    if not 10<=a.seconds<=600:p.error('seconds must be 10..600')
    if not (MAPS[a.map]/'index.html').exists():p.error('Local map index missing')
    sys.path.insert(0,str(ROOT/'development/staged/map_import_20261102'))
    import record_validation as recorder
    recorder.MAP=MAPS[a.map]
    session=DATA/'sessions'/f'{a.map}_{a.split}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    try:
        asyncio.run(recorder.record(a.seconds,classifier_session=session,map_id=a.map,model_path=classifier_path()))
    finally:
        manifest=session/'session.json'
        if manifest.exists():
            data=json.loads(manifest.read_text())
            data['split']=a.split
            data['map_release']=REGISTRY['maps'][a.map]['release']
            manifest.write_text(json.dumps(data,indent=2))
            print('SESSION:',session,flush=True)
            print('LABEL COMMAND:',flush=True)
            print(f'& $py -B -s rs_v4_screen_states.py label --root "{DATA}" --session "{session}"',flush=True)

if __name__=='__main__':main()
