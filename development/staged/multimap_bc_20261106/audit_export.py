"""Validate a completed export and write a compact report and exact-input gallery."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import numpy as np
from common import read,digest
from train import Dataset,split_check


def audit(folder):
    folder=Path(folder);meta=read(folder/'dataset.json')
    training=Dataset(folder,'train')
    try:
        validation=Dataset(folder,'validation')
        try:split_check(training,validation)
        finally:validation.close()
    finally:training.close()
    if digest(meta['checkpoint'])!=meta['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed')
    totals=defaultdict(lambda:dict(runs=0,samples=0,stored_frames=0,source_frames=0,
                                  actions=[0]*5,sessions=set(),reviewed_state_changes={}))
    gallery=['<!doctype html><meta charset="utf-8"><title>BC policy inputs</title>',
             '<style>body{background:#171717;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}section{margin-bottom:32px}</style>',
             '<h1>Exact BC policy inputs</h1><p>Each strip contains four 384×384 RGB images, oldest to newest. '
             'Requested ages: 0.75, 0.25, 0.05, 0 seconds. Missing history is black with validity flags in the auxiliary vector. '
             'These previews are unchanged model pixels. Human-reviewed state corrections are recorded in dataset.json.</p>']
    chosen=set()
    for run in meta['runs']:
        key=run['map']+' '+run['split'];entry=totals[key]
        entry['runs']+=1;entry['samples']+=run['count'];entry['stored_frames']+=len(run['source_frame_indices'])
        entry['source_frames']+=run['source_frames'];entry['sessions'].add(run['session'])
        entry['actions']=[a+b for a,b in zip(entry['actions'],run['action_counts'])]
        for change,count in run['reviewed_state_changes'].items():
            entry['reviewed_state_changes'][change]=entry['reviewed_state_changes'].get(change,0)+count
        # One example per session plus runs where labels recovered gameplay.
        if run['session'] not in chosen or (run['reviewed_state_changes'] and len(chosen)<25):
            chosen.add(run['session'])
            previews=sorted((folder/run['path']).glob('input_*.png'))
            preview=previews[len(previews)//2]
            gallery.append('<section><h2>'+html.escape(key+' / '+Path(run['source']).name)+'</h2><p>'+html.escape(run['session'])+'</p><img loading="lazy" src="'+preview.relative_to(folder).as_posix()+'"></section>')
    for entry in totals.values():
        entry['sessions']=sorted(entry['sessions'])
        if not all(entry['actions']):raise ValueError('Missing action class in a map/split')
    result=dict(status='PASS',checks='All exported array checksums, shapes, ranges, split isolation and source checkpoint hash',
                groups=dict(totals),actions_order=['NONE','LEFT','RIGHT','JUMP','ROLL'],
                bytes=sum(p.stat().st_size for p in folder.glob('run_*/*') if p.is_file()),
                note='Duration-selected demonstrations, final three seconds trimmed. No claim of perfect actions or gameplay improvement.')
    (folder/'preparation_report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (folder/'policy_input_gallery.html').write_text('\n'.join(gallery),encoding='utf8')
    print(json.dumps(result,indent=2));print('GALLERY:',folder/'policy_input_gallery.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    audit(parser.parse_args().folder)
