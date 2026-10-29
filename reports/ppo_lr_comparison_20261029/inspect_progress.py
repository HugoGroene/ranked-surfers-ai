"""Small read-only progress snapshot; batch.json remains authoritative."""
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def tail(path):
    if not path.exists():return ''
    with path.open('rb') as handle:
        handle.seek(0,2)
        handle.seek(max(0,handle.tell()-65536))
        return handle.read().decode('utf8',errors='replace')


batch=json.loads((HERE/'batch/batch.json').read_text())
completed=len(batch['completed'])
stages=batch['manifest']['stages']
if completed==len(stages):
    print(json.dumps(dict(status=batch['status'],completed=completed)))
else:
    args=stages[completed]['args']
    name=args[args.index('--run-name')+1]
    mode=args[args.index('--mode')+1]
    result=dict(status=batch['status'],stage=name)
    if mode=='train':
        for line in reversed(tail(ROOT/'ai_data/rl_v2_runs'/name/'learning_transitions.jsonl').splitlines()):
            try:row=json.loads(line)
            except ValueError:continue
            result['transitions']=row.get('transition',row.get('through_transition',1662427))-1662427
            break
    else:
        matches=re.findall(r'\[V61 EVAL\] episode (\d+)/100',tail(HERE/'batch'/(name+'.log')))
        result['episodes']=int(matches[-1]) if matches else 0
    if batch.get('error'):result['error']=batch['error']
    print(json.dumps(result))
