"""Read-only comparison of bounded native-readback training probes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/experiments/multimap_classifier'))
from summarize_probes import summarize,metrics


def analyze(folder):
    flags=json.loads((folder/'result.json').read_text())
    result=summarize(folder)
    result.update(bulk=flags.get('bulk_readback',False),direct=flags.get('direct_readback',False),
        probe_status=flags['status'],source_unchanged=flags['source_unchanged'],
        source_sha256=flags['source_sha256'],remaining_pids=flags['remaining_owned_pids'])
    samples={}
    for path in (folder/'case/train').glob('worker_*/learning_transitions.jsonl'):
        with path.open() as f:
            for line in f:
                row=json.loads(line)
                if row.get('event')!='collected':continue
                profile=(row.get('source_capture') or {}).get('capture_profile') or {}
                for key in ('mapped_copy_ms','shared_copy_ms','gpu_bridge_ms','gpu_map_wait_ms',
                            'node_request_to_paint_ms','mapped_copy_calls','mapped_row_pitch','direct_shared_readback'):
                    if isinstance(profile.get(key),(float,int)):
                        samples.setdefault(key,[]).append(float(profile[key]))
                if all(k in profile for k in ('mapped_copy_ms','shared_copy_ms')):
                    samples.setdefault('two_copy_total_ms',[]).append(profile['mapped_copy_ms']+profile['shared_copy_ms'])
    result['readback']={key:metrics(values) for key,values in samples.items()}
    report=json.loads((folder/'case/train/parallel_training.json').read_text())
    result['audits']=[a['status'] for a in report['audits']]
    result['finalization_errors']=report.get('finalization_errors')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reports',nargs='+',type=Path)
    a=p.parse_args();runs=[analyze(f) for f in a.reports]
    output=ROOT/'reports'/f'readback_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    output.write_text(json.dumps(dict(runs=runs,note='Short diagnostic training, not learning comparison. '
        'Per-command latency excludes NONE; phases separated. Copy timings do not establish end-to-end improvement. '
        'Original checkpoint retained. No change to image size, pixels, guards or PPO parameters.'),indent=2))
    print('REPORT:',output)
    for run in runs:
        print(json.dumps(dict(report=run['report'],bulk=run['bulk'],direct=run['direct'],
            ordinary=run['latency']['ordinary'],copy_total=run['readback']['two_copy_total_ms'],
            optimized_per_second=run['optimized_per_second'],status=run['probe_status'])))


if __name__=='__main__':main()
