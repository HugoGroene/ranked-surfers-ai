"""Read-only model/result checks; write NEW primary or combined analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat',action='store_true')
    args=parser.parse_args()
    output=HERE/('combined_results.json' if args.repeat else 'primary_results.json')
    if output.exists():raise SystemExit('Analysis exists; preserve original result')
    plan=read(HERE/'plan.json')
    hashes=read(HERE/'code_sha256.json')
    changed=[p for p,h in hashes.items() if not (ROOT/p).is_file()
             or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    source=ROOT/plan['source_checkpoint']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_sha256']
    assert read(HERE/'batch/batch.json')['status']=='COMPLETE'
    labels={'baseline':['ppolr_20261029_baseline'],
            'low':['ppolr_20261029_low_eval'],'high':['ppolr_20261029_high_eval']}
    if args.repeat:
        assert read(HERE/'repeat_batch/batch.json')['status']=='COMPLETE'
        labels['low'].append('ppolr_20261029_low_repeat')
        labels['high'].append('ppolr_20261029_high_repeat')
    evaluations={}
    samples={}
    for label,names in labels.items():
        sessions=[]
        pooled=[]
        for name in names:
            folder=ROOT/'ai_data/rl_v2_runs'/name
            report=read(folder/'v61_eval_report.json')
            durations=np.asarray(report['durations_seconds'])
            assert report['eval_episodes']==len(durations)==100
            guard=report['detector']['v612_live_input_guard']
            assert not guard['guard_hard_stopped'] and guard['uncertain_state_abort']==0
            sessions.append(dict(run=name,mean=float(durations.mean()),median=float(np.median(durations)),
                best=float(durations.max()),reached_5s=int((durations>=5).sum()),
                reached_10s=int((durations>=10).sum()),action_counts=report['action_counts'],
                action_probabilities=report['mean_action_probabilities'],
                timing=report['gameplay_only_intervals_ms'],guard=guard))
            pooled.extend(durations.tolist())
        samples[label]=np.asarray(pooled)
        evaluations[label]=dict(episodes=len(pooled),mean=float(np.mean(pooled)),
            median=float(np.median(pooled)),best=max(pooled),sessions=sessions)
    rng=np.random.default_rng(plan['bootstrap_seed'])
    contrasts={}
    for a,b in [('low','high'),('low','baseline'),('high','baseline')]:
        da,db=samples[a],samples[b]
        draws=rng.choice(da,(plan['bootstrap_resamples'],len(da))).mean(axis=1)
        draws-=rng.choice(db,(plan['bootstrap_resamples'],len(db))).mean(axis=1)
        contrasts[a+'_minus_'+b]=dict(mean_seconds=float(da.mean()-db.mean()),
            bootstrap_95_ci=np.quantile(draws,[.025,.975]).tolist())
    training={}
    saved_settings=[]
    for label,rate in [('low',.0001),('high',.0003)]:
        name='ppolr_20261029_'+label+'_train'
        folder=ROOT/'ai_data/rl_v2_runs'/name
        report=read(folder/'v61_report.json')
        effective=read(folder/'v61_effective_ppo_settings.json')['effective_settings']
        protocol=read(folder/'learning_protocol.json')
        audit=read(HERE/'batch'/(name+'_audits.json'))
        assert report['completed_requested_steps'] and report['steps']==32768
        assert audit['status']=='PASS_SETTLED_LEARNING'
        assert effective['n_steps']==512 and effective['learning_rate_at_start']==rate
        assert effective['target_kl'] is None
        assert protocol['reference_seconds']==.05
        with zipfile.ZipFile(folder/'latest.zip') as archive:
            assert archive.testzip() is None
            saved= json.loads(archive.read('data'))
        assert saved['num_timesteps']==plan['source_timesteps']+32768
        assert saved['learning_rate']==rate and saved['n_steps']==512
        saved_settings.append({k:v for k,v in effective.items() if k not in ('learning_rate','learning_rate_at_start')})
        training[label]=dict(run=name,effective_settings=effective,
            wall_seconds=report['wall_seconds'],steps_per_second=report['avg_steps_per_sec_including_resets'],
            transition_accounting=report['learning_transitions'],audit=audit,
            checkpoint_sha256=hashlib.sha256((folder/'latest.zip').read_bytes()).hexdigest(),
            saved_timesteps=saved['num_timesteps'],saved_n_updates=saved['_n_updates'],
            guard=report['detector']['v612_live_input_guard'])
    assert saved_settings[0]==saved_settings[1]
    delta=contrasts['low_minus_high']
    lo,hi=delta['bootstrap_95_ci']
    close=abs(delta['mean_seconds'])<.5 or lo<=0<=hi
    result=dict(status='COMPLETE_COMBINED' if args.repeat else 'COMPLETE_PRIMARY',
        evaluations=evaluations,contrasts=contrasts,training=training,
        primary_close_rule_triggered=close if not args.repeat else None,
        repeats_included=args.repeat,source_unchanged=True,runtime_code_changes=changed,
        limitations=plan['limitations'],
        uncertainty_note='Independent episode bootstrap conditional on these checkpoints/sessions; pooled repeats are not independent training seeds. Adaptive repetition follows the predeclared close rule.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    for label,r in evaluations.items():print(label,r['episodes'],round(r['mean'],4),round(r['median'],4))
    print('contrasts',json.dumps(contrasts))
    print('REPEAT_REQUIRED',bool(close and not args.repeat))


if __name__=='__main__':main()
