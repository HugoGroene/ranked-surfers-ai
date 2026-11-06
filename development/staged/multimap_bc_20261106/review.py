"""Duration-ranked visual review; explicit approvals with start:end trimming."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
from common import DATA,read,write_new,digest


def runs(root):
    return sorted(Path(root).glob('sessions/*/run_*/manifest.json'),key=lambda p:read(p)['duration'],reverse=True)


def latest(run):
    files=sorted((run/'reviews').glob('*.json'))
    return (read(files[-1]),files[-1]) if files else ({'status':'candidate'},None)


def approve(run,segments,reviewed,status='approved',*,duration_rule=None):
    run=Path(run);manifest=read(run/'manifest.json');intervals=[]
    for part in (segments or '').split(','):
        if not part:continue
        lo,hi=map(float,part.split(':'))
        if not 0<=lo<hi<=manifest['duration'] or (intervals and lo<=intervals[-1][1]):
            raise ValueError('Use ordered disjoint start:end intervals inside the recording')
        intervals.append([lo,hi])
    if duration_rule is not None:
        minimum,trim=duration_rule
        if (not 0<trim<minimum or manifest['duration']<minimum or
                intervals!=[[0.,manifest['duration']-trim]]):
            raise ValueError('Duration selection must retain exactly the requested interval')
    if status=='approved' and ((not reviewed and duration_rule is None) or not intervals):
        raise ValueError('Approval requires visual review and explicit trimmed intervals')
    directory=run/'reviews';directory.mkdir(exist_ok=True)
    revision=len(list(directory.glob('*.json')))
    path=directory/f'{revision:06d}_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    write_new(path,dict(status=status,reviewed=reviewed,intervals=intervals,
        approval_basis='user_directed_duration_trim' if duration_rule is not None else 'visual_review',
        duration_rule=None if duration_rule is None else dict(minimum_seconds=minimum,trim_end_seconds=trim),
        manifest_sha256=digest(run/'manifest.json'),index_sha256=digest(run/'frames.json'),
        events_sha256=digest(run/'events.json')))
    return path


def authorized(review,manifest):
    if review.get('reviewed'):return True
    if review.get('approval_basis')!='user_directed_duration_trim':return False
    rule=review.get('duration_rule') or {}
    minimum,trim=rule.get('minimum_seconds',0),rule.get('trim_end_seconds',0)
    return (0<trim<minimum and manifest['duration']>=minimum and
            review.get('intervals')==[[0.,manifest['duration']-trim]])


def catalogue(root,min_seconds=0):
    items=[]
    for file in runs(root):
        m=read(file);run=file.parent
        if m['duration']<min_seconds:continue
        rows=read(run/'frames.json');origin=rows[0]['at'] if rows else 0
        frames=[dict(src=(run/'frames'/r['file']).resolve().as_uri(),t=round(r['at']-origin,3),state=r['state']) for r in rows]
        events=[dict(t=round(e['at']-origin,3),code=e['code'],type=e['type']) for e in read(run/'events.json') if e['at']>=origin]
        items.append(dict(title=f"{m['map']} / {m['split']} / {m['duration']:.2f}s / {latest(run)[0]['status']}",
                          run=str(run.resolve()),frames=frames,events=events,
                          warnings=sum(r['state']!='GAMEPLAY' for r in rows)))
    output=Path(root)/f'review_{datetime.now():%Y%m%d_%H%M%S_%f}.html'
    payload=json.dumps(items).replace('<','\\u003c')
    document='''<!doctype html><meta charset="utf-8"><title>BC candidates</title>
<style>body{font:16px sans-serif;background:#17191d;color:#ddd;max-width:1100px;margin:auto}img{width:100%}section{margin:30px 0;padding:15px;background:#25282e}input{width:85%}pre{white-space:pre-wrap}button{padding:8px}</style>
<h1>Native BC recordings — longest first</h1><p>Candidate only. Inspect inputs and gameplay, then approve explicit safe intervals using review.py. Trim before the mistake, not merely before death. Use separate recording sessions for validation.</p><main></main><script>
const runs=DATA;
for(const r of runs){const s=document.createElement('section');const h=document.createElement('h2');h.textContent=r.title;s.append(h);
const p=document.createElement('pre');p.textContent=r.run;s.append(p);const img=document.createElement('img');s.append(img);
const warning=document.createElement('p');warning.textContent=r.warnings+' non-GAMEPLAY frames: inspect before approval; these can be real deaths/pauses or classifier mistakes.';s.append(warning);
const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=r.frames.length-1;slider.value=0;s.append(slider);
const button=document.createElement('button');button.textContent='Play / pause';s.append(button);const label=document.createElement('pre');s.append(label);
const flagged=document.createElement('button');flagged.textContent='Next state warning';s.append(flagged);
let timer=null;function draw(){const f=r.frames[+slider.value];if(!f)return;img.src=f.src;label.textContent=f.t+' s | '+f.state+' | nearby inputs: '+r.events.filter(e=>Math.abs(e.t-f.t)<.15).map(e=>e.code+' '+e.type+' @ '+e.t).join(', ');}
function next(){draw();const i=+slider.value;if(i>=r.frames.length-1){timer=null;return;}timer=setTimeout(()=>{slider.value=i+1;next()},Math.max(1,(r.frames[i+1].t-r.frames[i].t)*1000));}
flagged.onclick=()=>{if(timer){clearTimeout(timer);timer=null}const i=r.frames.findIndex((f,i)=>i>+slider.value&&f.state!=='GAMEPLAY');if(i>=0){slider.value=i;draw()}};
slider.oninput=draw;button.onclick=()=>{if(timer){clearTimeout(timer);timer=null}else next()};draw();document.querySelector('main').append(s);}
</script>'''.replace('const runs=DATA;',f'const runs={payload};')
    with output.open('x',encoding='utf8') as f:f.write(document)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    listing=subs.add_parser('list');listing.add_argument('--root',type=Path,default=DATA)
    listing.add_argument('--min-seconds',type=float,default=0)
    choose=subs.add_parser('approve');choose.add_argument('run',type=Path)
    choose.add_argument('--segments',required=True);choose.add_argument('--reviewed',action='store_true')
    reject=subs.add_parser('reject');reject.add_argument('run',type=Path)
    a=p.parse_args()
    if a.command=='list':print(catalogue(a.root,a.min_seconds))
    elif a.command=='approve':print(approve(a.run,a.segments,a.reviewed))
    else:print(approve(a.run,None,True,'rejected'))


if __name__=='__main__':main()
