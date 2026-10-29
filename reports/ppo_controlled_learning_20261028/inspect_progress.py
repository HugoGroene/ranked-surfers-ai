import json,re
from pathlib import Path
p=Path(__file__).resolve().parent
s=json.loads((p/'batch/batch.json').read_text())
i=len(s['completed'])
r={'status':s['status'],'completed_stages':i,'total_stages':len(s['manifest']['stages'])}
if i<len(s['manifest']['stages']):
 a=s['manifest']['stages'][i]['args']; name=a[a.index('--run-name')+1]; mode=a[a.index('--mode')+1]
 r.update(stage=name,mode=mode)
 log=p/'batch'/(name+'.log')
 if log.exists():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-160000)); tail=f.read().decode('utf8',errors='replace')
  if mode=='train':
   records=[x.split('[V612 PROGRESS] ',1)[1] for x in tail.splitlines() if '[V612 PROGRESS] ' in x]
   if records:
    try:
     x=json.loads(records[-1]); r.update(new_steps=x['steps']-1662427,seconds=x['seconds'],episodes=x['episodes'],recent_mean=x['mean_recent_25_seconds'])
    except ValueError: pass
  else:
   matches=re.findall(r'\[V61 EVAL\] episode (\d+)/',tail)
   r['episodes_completed']=int(matches[-1]) if matches else 0
print(json.dumps(r))
