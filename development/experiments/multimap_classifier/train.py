"""Train a candidate using explicit per-map session splits; never promote automatically."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import rs_v4_screen_states as states


def audit_baseline(examples,metadata,baseline,repo=ROOT):
    """An expansion must retain every labelled example/split used by the baseline."""
    old=json.loads(Path(baseline).read_text())
    current={(str(path.resolve()),label,s) for path,label,s in examples}
    for row in old['examples']:
        path=(repo/row['path']).resolve()
        if (str(path),row['label'],row['session']) not in current:
            raise ValueError('Baseline labelled example missing or relabelled: '+str(path))
        if metadata[row['session']]['split']!=row['split']:
            raise ValueError('Baseline session split changed: '+row['session'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Baseline image changed: '+str(path))
    return dict(manifest=str(Path(baseline).resolve()),
        sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
        retained_examples=len(old['examples']),status='ALL_BASELINE_EXAMPLES_RETAINED')

def partition(root):
    examples=states.load_dataset(root)
    metadata={}
    for _,_,session in examples:
        if session not in metadata:
            m=json.loads((root/'sessions'/session/'session.json').read_text())
            if m.get('map_id')=='original':m['map_id']='halloween'
            if m.get('split') not in ('train','validation'):raise ValueError('Missing explicit split: '+session)
            if m.get('format')!='canvas_v1_guard_1920x1080_offset_0_87':raise ValueError('Wrong framing: '+session)
            metadata[session]=m
    if not examples:raise ValueError('No labelled examples')
    for name in {m['map_id'] for m in metadata.values()}:
        for split in ('train','validation'):
            counts=Counter(label for _,label,s in examples if metadata[s]['map_id']==name and metadata[s]['split']==split)
            if set(states.CLASSES)-set(counts):raise ValueError(f'{name} {split} missing classes: {counts}')
    return examples,metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--base-manifest',type=Path,default=ROOT/'ai_data/multimap_screen_states/candidates/multimap_20261104_181224_862793/dataset_manifest.json')
    a=p.parse_args()
    if not 1<=a.epochs<=200:p.error('epochs must be 1..200')
    root=ROOT/'ai_data/multimap_screen_states'
    examples,meta=partition(root)
    baseline=audit_baseline(examples,meta,a.base_manifest)
    print('Baseline retention:',json.dumps(baseline))
    manifest=[];split_hashes={'train':set(),'validation':set()}
    for path,label,s in examples:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        split=meta[s]['split'];split_hashes[split].add(digest)
        manifest.append(dict(path=str(path.relative_to(ROOT)),label=label,session=s,
                             map=meta[s]['map_id'],split=split,sha256=digest))
    overlap=split_hashes['train'] & split_hashes['validation']
    print('Session coverage:',json.dumps({s:dict(Counter(l for _,l,t in examples if t==s)) for s in meta},indent=2))
    print('Identical image hashes across splits:',len(overlap), '(may include static loading/menu screens)')
    if a.check_only:return
    output=root/'candidates'/datetime.now().strftime('multimap_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'dataset_manifest.json').write_text(json.dumps(dict(examples=manifest,sessions=meta,
        baseline_retention=baseline,exact_hash_overlap=len(overlap),epochs=a.epochs,seed=123),indent=2))
    print('CANDIDATE:',output,flush=True)
    torch,_=states.get_torch()
    torch.manual_seed(123);states.np.random.seed(123)
    # Reuse the old architecture/trainer. Pool ONLY explicitly held-out sessions;
    # original map/session identities remain in the manifest and per-map report.
    original=states.load_dataset
    states.load_dataset=lambda _: [(path,label,'heldout' if meta[s]['split']=='validation' else s)
                                   for path,label,s in examples]
    try:
        states.cmd_train(SimpleNamespace(root=root,threads=2,min_per_class=1,
            val_session='heldout',crop=states.CROP,batch_size=32,lr=.001,
            epochs=a.epochs,output=output/'candidate.pt'))
    finally:states.load_dataset=original
    model=states.ScreenStateClassifier(output/'candidate.pt')
    report=dict(status='CANDIDATE_REQUIRES_LIVE_VALIDATION',maps={},errors=[],
                note='Held-out sessions select the best epoch; these are validation, not an independent final test.')
    for path,label,s in examples:
        if meta[s]['split']!='validation':continue
        name=meta[s]['map_id']
        metrics=report['maps'].setdefault(name,dict(classes=list(model.classes),
            confusion=[[0]*len(model.classes) for _ in model.classes],samples=0,correct=0,uncertain_at_090=0))
        prediction,probs=model.state(states.cv2.imread(str(path)),confidence=.9)
        guess=max(probs,key=probs.get)
        metrics['confusion'][model.classes.index(label)][model.classes.index(guess)]+=1
        metrics['samples']+=1;metrics['correct']+=int(guess==label)
        metrics['uncertain_at_090']+=int(prediction=='UNCERTAIN')
        if guess!=label or prediction=='UNCERTAIN':report['errors'].append(dict(image=str(path),truth=label,prediction=prediction,probabilities=probs))
    (output/'per_map_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['maps'],indent=2))
    print('REPORT:',output/'per_map_validation.json')

if __name__=='__main__':main()
