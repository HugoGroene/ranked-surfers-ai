"""Headless continuation with the explicit extended input contract."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from command_adapter import ROOT, HERE, install, sources, configuration, contract

# Windows spawn must install the same process-local environment before loading
# the production worker target and the adapted policy checkpoint.
if __name__=='__main__':os.environ['RS_COMMAND_HISTORY_ACTIVE']='1'
if os.environ.get('RS_COMMAND_HISTORY_ACTIVE')=='1':install()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=10)
    p.add_argument('--workers',type=int,choices=range(1,6),default=4)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not .25<=a.minutes<=2880:p.error('Minutes must be .25..2880')
    a.checkpoint=a.checkpoint.resolve()
    from tools.ppo_observation_contract import validate_checkpoint
    validate_checkpoint(a.checkpoint,contract())
    root=(a.output or ROOT/'reports'/datetime.now().strftime('command_history_train_%Y%m%d_%H%M%S_%f')).resolve()
    if root.exists() or not root.is_relative_to(ROOT/'reports'):p.error('Choose a new reports folder')
    root.mkdir(parents=True)
    manifest=root/'sources.json';manifest.write_text(json.dumps(sources(),indent=2))
    (root/'plan.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),workers=a.workers,
        minutes=a.minutes,configuration=configuration(),contract=contract()),indent=2))
    print('COMMAND HISTORY RUN:',root,flush=True)
    import run_workers
    run_workers.sources=sources
    run_workers.configuration=configuration
    sys.argv=['run_workers','--workers',str(a.workers),'--minutes',str(a.minutes),
        '--checkpoint',str(a.checkpoint),'--output',str(root/'train'),
        '--expected-sources',str(manifest)]
    run_workers.main()


if __name__=='__main__':main()
