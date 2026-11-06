"""Prepare unlabelled classifier-review sessions from BC warning frames."""
import argparse
from bisect import bisect_left,bisect_right
from pathlib import Path
import cv2
from common import ROOT,read,write_new,digest
from tools.ppo_headless_env import compose_guard_frame


def warning_indices(rows,spacing=.25,all_warnings=False):
    """Sparse warning samples plus preceding/following context, without relabelling."""
    selected=set();last=float('-inf');times=[r['at'] for r in rows]
    for i,r in enumerate(rows):
        if r['state'] not in (('UNCERTAIN','OTHER','LOADING','MENU') if all_warnings else ('UNCERTAIN','OTHER','LOADING')):continue
        boundary=(i==0 or rows[i-1]['state']!=r['state'] or i==len(rows)-1 or rows[i+1]['state']!=r['state'])
        if all_warnings or boundary or r['at']-last>=spacing:
            selected.add(i);last=r['at']
            if boundary:
                selected.add(max(0,bisect_right(times,r['at']-.25)-1))
                selected.add(min(len(rows)-1,bisect_left(times,r['at']+.25)))
    return sorted(selected)


def prepare(session,root,all_warnings=False):
    session=Path(session).resolve();meta=read(session/'session.json')
    if meta.get('split') not in ('train','validation'):raise ValueError('Missing source split')
    root=Path(root)
    # Never put derived material from the same recording into another split.
    imported=set()
    for file in (root/'sessions').glob('*/session.json'):
        existing=read(file)
        if existing.get('bc_source_session')==str(session):
            if existing.get('split')!=meta['split']:raise ValueError('Source recording split conflict')
            if not all_warnings:raise ValueError('Already imported this source: '+str(file.parent))
            for frame in read(file.parent/'import_manifest.json')['frames']:
                imported.add(str(Path(frame['source']).resolve()))
    sources=[]
    for file in sorted(session.glob('run_*/frames.json')):
        rows=read(file)
        for i in warning_indices(rows,all_warnings=all_warnings):
            r=rows[i];path=file.parent/'frames'/r['file']
            if not path.resolve().is_relative_to(session):raise ValueError('Frame outside source session')
            if str(path.resolve()) not in imported:
                sources.append((path,r,file.parent.name,r['at']-rows[0]['at']))
    if all_warnings and (session/'guard_observations.jsonl').exists():
        seen={str(p.resolve()) for p,_,_,_ in sources}|imported
        import json
        with (session/'guard_observations.jsonl').open(encoding='utf8') as f:
            for line in f:
                row=json.loads(line)
                if not row.get('warning') or not row.get('frame'):continue
                path=(session/row['frame']).resolve()
                if not path.is_relative_to(session):raise ValueError('Evidence outside source session')
                if str(path) in seen or not path.is_file():continue
                seen.add(str(path));row['sha256']=digest(path)
                sources.append((path,row,row.get('run') or 'outside',row.get('run_age_seconds') or 0.))
    if not sources:return None
    dest=root/'sessions'/(('bc_full_' if all_warnings else 'bc_')+session.name)
    dest.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,(path,row,run,age) in enumerate(sources):
        if digest(path)!=row['sha256']:raise ValueError('Source frame checksum mismatch')
        native=cv2.imread(str(path))
        if native is None or native.shape!=(945,1920,3):raise ValueError('Wrong native canvas shape')
        guard=compose_guard_frame(native,0,87)[:,:,:3]
        name=f'{i:04d}_{run}_{round(age*1000):07d}ms.png'
        target=dest/name
        if not cv2.imwrite(str(target),guard,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise OSError('Cannot save guard image')
        records.append(dict(file=name,source=str(path),source_sha256=row['sha256'],
            run=run,age_seconds=age,recorded_state=row['state'],probabilities=row.get('probabilities'),
            sha256=digest(target),label=None))
    write_new(dest/'import_manifest.json',dict(frames=records,
        note='Warnings and nearby context, not approved labels. Review in existing screen-state label UI.'))
    write_new(dest/'session.json',dict(map_id=meta['map'],split=meta['split'],
        map_release=meta.get('map_release'),bc_source_session=str(session),
        format='canvas_v1_guard_1920x1080_offset_0_87',
        source_session_sha256=digest(session/'session.json'),frames=len(records),
        labels='unlabelled; human review required',sampling=(
            'all saved non-gameplay candidate frames and diagnostic evidence plus context; previously imported sources omitted'
            if all_warnings else 'warning bursts sampled at 250ms plus context')))
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions',type=Path,nargs='+',required=True)
    p.add_argument('--root',type=Path,default=ROOT/'ai_data/multimap_screen_states')
    p.add_argument('--all-warnings',action='store_true',help='Every saved warning plus menu transitions; supplement earlier imports without changing them')
    a=p.parse_args();cv2.setNumThreads(1)
    for session in a.sessions:
        dest=prepare(session,a.root,all_warnings=a.all_warnings)
        print('REVIEW SESSION:',dest or 'No saved warning frames')
        if dest:print(f'& $py -B -s rs_v4_screen_states.py label --root "{a.root}" --session "{dest}"')


if __name__=='__main__':main()
