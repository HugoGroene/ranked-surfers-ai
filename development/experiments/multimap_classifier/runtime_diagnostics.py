"""Read-only memory/process snapshots at stage boundaries; never kill processes."""
import json
from pathlib import Path
import time
import psutil


def snapshot(root):
    root=str(Path(root).resolve()).lower()
    memory=psutil.virtual_memory()
    processes=[]
    for process in psutil.process_iter(['pid','ppid','name','cmdline','memory_info','create_time']):
        try:
            row=process.info
            name=(row['name'] or '').lower()
            matched=root in ' '.join(row['cmdline'] or []).lower()
            if 'electron' in name or (name.startswith('python') and matched):
                processes.append(dict(pid=row['pid'],ppid=row['ppid'],name=name,
                    created=row['create_time'],rss_mib=row['memory_info'].rss/1024**2,
                    project_command_match=matched))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return dict(wall_time=time.time(),memory_percent=memory.percent,
                available_gib=memory.available/1024**3,processes=processes,
                note='All visible Electron processes plus project-matching Python processes. '
                     'RSS includes shared pages; do not sum it as private RAM. No processes terminated.')


def append_snapshot(path,stage,root):
    try:
        value=dict(stage=stage,**snapshot(root))
    except Exception as exc:
        value=dict(stage=stage,error=str(exc))
    with Path(path).open('a',encoding='utf8') as handle:
        handle.write(json.dumps(value)+'\n')


if __name__=='__main__':
    print(json.dumps(snapshot(Path(__file__).resolve().parents[3]),indent=2))
