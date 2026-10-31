"""Join buffered timings to actual command transitions; do not sum overlapping stages."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

ADDITIVE=('source_delivery_ms','post_reply_capture_ms','post_capture_to_policy_ms',
          'policy_wall_ms','pre_command_guard_handoff_ms')


def overlap_ms(events,kind,start,end):
    spans=sorted((max(start,r['start']),min(end,r['end'])) for r in events
                 if r['kind']==kind and r['start']<end and r['end']>start)
    total=0.;right=start
    for lo,hi in spans:
        total+=max(0.,hi-max(lo,right));right=max(right,hi)
    return total*1000


def stats(values):
    values=[float(x) for x in values if x is not None and math.isfinite(x)]
    if not values:return None
    return dict(count=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
                p95=float(np.percentile(values,95)),maximum=float(max(values)))


def decisions(ledger,events):
    policy={r['transition']:r for r in events if r['kind']=='policy'}
    result=[]
    for row in ledger:
        if row.get('event')!='collected' or row.get('dispatch_at') is None:continue
        inf=policy.get(row['transition'])
        if inf is None or inf['passive']:continue
        source=row.get('source_capture') or {}
        if source!=inf['source']:raise ValueError('Policy profiling/ledger source lineage differs')
        if any(source.get(k) is None for k in ('request_started_at','image_received_at','captured_at')):continue
        request,reply,captured=(source[k] for k in ('request_started_at','image_received_at','captured_at'))
        dispatch=row['dispatch_at']
        if not request<=reply<=captured<=inf['start']<=inf['end']<=dispatch:
            raise ValueError('Profile timestamps do not form a causal decision timeline')
        value=dict(transition=row['transition'],sampled_action=row['sampled_action'],
            source_serial=source.get('serial'),electron_serial=source.get('electron_serial'),
            request_to_command_ms=(dispatch-request)*1000,
            source_delivery_ms=(reply-request)*1000,
            post_reply_capture_ms=(captured-reply)*1000,
            post_capture_to_policy_ms=(inf['start']-captured)*1000,
            policy_wall_ms=(inf['end']-inf['start'])*1000,
            tensor_host_ms=(inf['tensor_ready']-inf['start'])*1000,
            forward_host_enqueue_ms=(inf['forward_returned']-inf['tensor_ready'])*1000,
            action_ready_host_ms=(inf['end']-inf['forward_returned'])*1000,
            gpu_stream_ms=inf.get('gpu_stream_ms'),
            pre_command_guard_handoff_ms=(dispatch-inf['end'])*1000)
        if abs(sum(value[k] for k in ADDITIVE)-value['request_to_command_ms'])>.001:
            raise ValueError('Additive timing partition does not reconcile')
        # Child metrics overlap the partition above; report them separately.
        for kind,label in [('_pack','pack_ms'),('_native_frames','native_preprocess_ms'),
                           ('_read_screen_state','state_classifier_before_policy_ms')]:
            value[label]=overlap_ms(events,kind,captured,inf['start'])
        for kind,label in [('_capture','guard_capture_ms'),
                           ('_read_screen_state','guard_classifier_ms')]:
            value[label]=overlap_ms(events,kind,inf['end'],dispatch)
        value['source_page_screenshot_ms']=overlap_ms(events,'page_screenshot',request,reply)
        value['source_bounding_box_ms']=overlap_ms(events,'canvas_bounding_box',request,reply)
        candidates=[r for r in events if r['kind']=='_step_browser' and r['start']<=dispatch<=r['end']]
        guard=min(candidates,key=lambda r:r['end']-r['start']).get('guard_source',{}) if candidates else {}
        value['guard_source_serial']=guard.get('electron_serial',guard.get('captured_at'))
        value['guard_source_changed']=(guard.get('request_started_at') is not None and
            guard['request_started_at']!=request)
        value['source_capture_details']=source.get('capture_profile',{})
        value['guard_capture_details']=guard.get('capture_profile',{})
        result.append(value)
    return result


def summarize(rows):
    if not rows:return dict(command_samples=0)
    fields=[k for k in rows[0] if k.endswith('_ms')]
    total=[r['request_to_command_ms'] for r in rows]
    low,high,tail=np.percentile(total,[25,75,95])
    groups=dict(all=rows,central_50_percent=[r for r in rows if low<=r['request_to_command_ms']<=high],
                slowest_5_percent=[r for r in rows if r['request_to_command_ms']>=tail])
    return dict(command_samples=len(rows),
        metrics={k:stats(r.get(k) for r in rows) for k in fields},
        additive_groups={name:dict(count=len(group),total_mean_ms=float(np.mean([r['request_to_command_ms'] for r in group])),
            means_ms={k:float(np.mean([r[k] for r in group])) for k in ADDITIVE})
            for name,group in groups.items() if group},
        guard_source_changed=sum(r['guard_source_changed'] for r in rows),
        source_details={key:stats(r['source_capture_details'].get(key) for r in rows)
            for key in {k for r in rows for k in r['source_capture_details']}},
        guard_details={key:stats(r['guard_capture_details'].get(key) for r in rows)
            for key in {k for r in rows for k in r['guard_capture_details']}})


def read_rows(path):
    with path.open(encoding='utf8') as handle:return [json.loads(s) for s in handle if s.strip()]


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(run=str(args.report),workers={},note=(
        'Additive groups partition request-to-command wall time. Child timings overlap and must not be summed. '
        'Node request-to-paint combines render/readback/frame-wait/scheduling. GPU readback alone is unmeasured. '
        'Transport residual includes endpoint scheduling/copies, not pure network time. '
        'CUDA stream intervals include contention/idle time, not exclusive kernel execution. '
        'Command issue is not Unity input consumption.'))
    all_rows=[]
    for worker in sorted(args.report.glob('worker_*')):
        meta=json.loads((worker/'pipeline_profile_metadata.json').read_text())
        if meta['dropped_events']:raise ValueError('Profile event buffer overflowed')
        rows=decisions(read_rows(worker/'learning_transitions.jsonl'),read_rows(worker/'pipeline_profile.jsonl'))
        report['workers'][worker.name]=summarize(rows)
        all_rows.extend(dict(worker=worker.name,**row) for row in rows)
    report['pooled']=summarize(all_rows)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    with (args.output/'decisions.jsonl').open('w',encoding='utf8') as handle:
        for row in all_rows:handle.write(json.dumps(row)+'\n')
    print(json.dumps(report['pooled']['additive_groups'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
