"""Low-rate diagnostics in a separate thread; never blocks the action loop."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


class ResourceSampler:
    def __init__(self, path, process_ids=()):
        self.path = Path(path)
        self.process_ids = list(process_ids)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='ppo-resource-sampler')

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3.)

    def _run(self):
        with self.path.open('w', encoding='utf8', buffering=1) as handle:
            try:
                import psutil
                processes = [psutil.Process(pid) for pid in self.process_ids]
                psutil.cpu_percent(percpu=True)
                for process in processes:
                    process.cpu_percent()
                executable = shutil.which('nvidia-smi')
                child_processes = {}
                while not self.stop_event.wait(1.):
                    cpu = psutil.cpu_percent(percpu=True)
                    row = dict(wall_time=time.time(), perf_time=time.perf_counter(),
                               cpu_per_logical_core=cpu, cpu_busiest_core_percent=max(cpu),
                               cpu_mean_percent=sum(cpu)/len(cpu), memory_percent=psutil.virtual_memory().percent,
                               memory_available_gib=psutil.virtual_memory().available/(1024**3))
                    row['worker_cpu_percent_unscaled'] = {}
                    row['owned_browser_process_cpu'] = []
                    for process in processes:
                        try:
                            row['worker_cpu_percent_unscaled'][process.pid] = process.cpu_percent()
                            for child in process.children(recursive=True):
                                if child.pid not in child_processes:
                                    child.cpu_percent()
                                    child_processes[child.pid] = child
                        except psutil.NoSuchProcess:
                            pass
                    for pid, child in list(child_processes.items()):
                        try:
                            row['owned_browser_process_cpu'].append(dict(pid=pid, name=child.name(),
                                cpu_percent_unscaled=child.cpu_percent()))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            child_processes.pop(pid, None)
                    if executable:
                        try:
                            result = subprocess.run([executable,
                                '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=2.,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                            row['nvidia_csv'] = result.stdout.strip()
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            row['gpu_query_error'] = str(exc)
                    handle.write(json.dumps(row)+'\n')
                    try:
                        temporary=self.path.parent/'telemetry.partial.json'
                        temporary.write_text(json.dumps(row),encoding='utf8')
                        temporary.replace(self.path.parent/'telemetry.json')
                    except OSError:
                        pass
            except Exception as exc:
                handle.write(json.dumps(dict(telemetry_error=repr(exc)))+'\n')
