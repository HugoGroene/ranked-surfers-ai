"""Review all BC-derived classifier sessions, retaining source train/validation splits."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from common import ROOT,read,digest,require_idle
import json
import rs_v4_screen_states as states


def sessions(root):
    result=[]
    for file in (root/'sessions').glob('*/session.json'):
        meta=read(file)
        if not meta.get('bc_source_session'):continue
        session=file.parent
        if not (session/'import_manifest.json').is_file():continue
        progress=session/'label_queue_progress.json'
        old=read(progress) if progress.exists() else {}
        done=(old.get('complete',False) and (session/'labels.csv').is_file()
            and old.get('labels_sha256')==digest(session/'labels.csv')
            and old.get('import_sha256')==digest(session/'import_manifest.json'))
        result.append((session,meta,done))
    return sorted(result,key=lambda row:(row[1]['split'],row[1]['bc_source_session'],row[0].name))


def run_queue(root,list_only=False):
    queue=sessions(root)
    for session,meta,done in queue:
        print(f'{"DONE" if done else "PENDING"}: {session.name} | {meta["split"]} | {meta["frames"]} images',flush=True)
    if list_only:return
    for session,meta,done in queue:
        if done:continue
        print('LABEL SESSION:',session,flush=True)
        complete=states.cmd_label(SimpleNamespace(root=root,session=session,resume=True))
        (session/'label_queue_progress.json').write_text(json.dumps(dict(complete=bool(complete),
            labels_sha256=digest(session/'labels.csv'),import_sha256=digest(session/'import_manifest.json')),indent=2))
        if not complete:
            print('Saved and stopped. Run the same command to continue.',flush=True);return
    print('ALL CLASSIFIER REVIEW SESSIONS COMPLETE. No classifier trained or replaced.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'ai_data/multimap_screen_states')
    p.add_argument('--list',action='store_true')
    a=p.parse_args()
    if not a.list:require_idle()
    run_queue(a.root.resolve(),a.list)


if __name__=='__main__':main()
