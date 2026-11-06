"""Manual live candidate check on both local maps; no automatic inputs or promotion."""
import argparse
import asyncio
import hashlib
from pathlib import Path
import sys
from record import ROOT,MAPS
from classifier_config import classifier_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=int,default=90)
    p.add_argument('--model',type=Path,default=classifier_path())
    a=p.parse_args()
    if not 10<=a.seconds<=600:p.error('seconds must be 10..600')
    if not a.model.is_file():p.error('Candidate not found')
    digest=hashlib.sha256(a.model.read_bytes()).hexdigest()
    sys.path.insert(0,str(ROOT/'development/staged/map_import_20261102'))
    import record_validation as recorder
    for name,folder in MAPS.items():
        input(f'Next map: {name}. Press Enter when ready. Play manually; show menus, deaths, pause/resume. ')
        recorder.MAP=folder
        asyncio.run(recorder.record(a.seconds,map_id=name,model_path=a.model,candidate_check=True))
    assert hashlib.sha256(a.model.read_bytes()).hexdigest()==digest
    print('Candidate observations saved for review. No classifier promoted; automated stability remains separate.')

if __name__=='__main__':main()
