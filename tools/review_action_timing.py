"""Offline timestamped action filmstrips; no automatic movement-duration claims."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root=args.report.resolve()
    frames=[json.loads(line) for line in (root/'sequences/frames.jsonl').read_text().splitlines()]
    steps=[json.loads(line) for line in (root/'step_trace.jsonl').read_text().splitlines()]
    trials=json.loads((root/'v61_input_probe.json').read_text())['trials']
    rows=[];sections=[]
    for episode,trial in enumerate(trials,1):
        actions=[s for s in steps if s['episode']==episode and s.get('phase')=='action' and s['action']!=0]
        if not actions:continue
        first=actions[0]
        origin=first['elapsed_seconds']-first['step_wall_ms']/1000
        selected={}
        for f in frames:
            if f['episode']==episode and ('_action_' in f['picture']):
                selected[f['step']]=f
        ordered=sorted(selected.values(),key=lambda f:f['elapsed_seconds'])
        row=dict(episode=episode,pattern=trial['pattern'],trial=trial['trial'],
                 time_origin='start of first action env.step; not key dispatch or Unity tick',
                 action_step_starts_ms=[round(1000*(s['elapsed_seconds']-s['step_wall_ms']/1000-origin),2) for s in actions],
                 frames=[dict(file=f['picture'],after_step_start_ms=round(1000*(f['elapsed_seconds']-origin),2)) for f in ordered],
                 movement_onset_ms=None,movement_completion_ms=None,
                 note='Manual review required; sparse frames and occlusion may prevent duration measurement.')
        rows.append(row)
        cards=''.join('<figure><img loading="lazy" src="sequences/'+html.escape(f['file'],quote=True)+'"><figcaption>'+str(f['after_step_start_ms'])+' ms</figcaption></figure>' for f in row['frames'])
        sections.append(f'<h2>Episode {episode}: {html.escape(trial["pattern"])} trial {trial["trial"]}</h2><div class="frames">{cards}</div>')
    target=root/'action_timing_review.html'
    data=root/'action_timing_review.json'
    if target.exists() or data.exists():raise FileExistsError('Review already exists; preserving it')
    data.write_text(json.dumps(rows,indent=2),encoding='utf8')
    target.write_text('<!doctype html><meta charset="utf-8"><title>Action timing review</title><style>body{font-family:sans-serif;background:#151515;color:#eee}.frames{display:flex;overflow-x:auto}figure{margin:6px}img{width:512px}figcaption{text-align:center}</style><h1>Recorded action sequences</h1><p>Times are observation-return times relative to action-step start, not exact input or simulation timestamps. Identify onset and completion visually; do not infer movement from camera motion alone.</p>'+''.join(sections),encoding='utf8')
    print(target)


if __name__=='__main__':main()
