"""Opt-in precise wait for the existing decision cap; never relax freshness.

Only the synchronous pre-step rate limiter uses this timer. Guard retry sleeps,
game clocks, optimizer pauses and capture scheduling are unchanged.
"""
import ast
from collections import deque
import ctypes
import inspect
import json
import math
import os
from pathlib import Path
import textwrap
import time


class DeadlineWait:
    def __init__(self, precise=False):
        self.handle=None
        self.rows=deque(maxlen=20000)
        self.calls=0
        self.mode='sleep'
        if precise and os.name=='nt':
            from ctypes import wintypes as w
            self.api=ctypes.WinDLL('kernel32',use_last_error=True)
            self.api.CreateWaitableTimerExW.argtypes=[ctypes.c_void_p,w.LPCWSTR,w.DWORD,w.DWORD]
            self.api.CreateWaitableTimerExW.restype=w.HANDLE
            self.api.SetWaitableTimer.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_longlong),w.LONG,ctypes.c_void_p,ctypes.c_void_p,w.BOOL]
            self.api.SetWaitableTimer.restype=w.BOOL
            self.api.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD]
            self.api.WaitForSingleObject.restype=w.DWORD
            self.api.CloseHandle.argtypes=[w.HANDLE];self.api.CloseHandle.restype=w.BOOL
            # Unnamed, non-inheritable synchronization timer; no global timer setting.
            self.handle=self.api.CreateWaitableTimerExW(None,None,2,0x00100002)
            if self.handle:self.mode='high_resolution_timer'

    def wait(self,seconds):
        if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid pacing duration')
        start=time.perf_counter();deadline=start+seconds
        while (remaining:=deadline-time.perf_counter())>0:
            if self.handle:
                due=ctypes.c_longlong(-max(1,math.ceil(remaining*10_000_000)))
                if not self.api.SetWaitableTimer(self.handle,ctypes.byref(due),0,None,None,False):
                    raise ctypes.WinError(ctypes.get_last_error())
                status=self.api.WaitForSingleObject(self.handle,max(1000,math.ceil(remaining*1000)+100))
                if status!=0:raise RuntimeError(f'Pacing wait failed: {status}')
            else:time.sleep(remaining)
        self.calls+=1
        self.rows.append(dict(requested_ms=seconds*1000,actual_ms=(time.perf_counter()-start)*1000))

    def close(self):
        if self.handle:
            handle,self.handle=self.handle,None
            self.api.CloseHandle(handle)


def replace_wait(function,owner):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n)=='time.sleep(remaining)']
    if len(matches)!=1:raise RuntimeError('Rate limiter changed; revalidate pacing')
    matches[0].func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_latency_rate_wait',ctx=ast.Load())
    # Recompiled nested method must preserve the original zero-argument super owner.
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='super' and not node.args:
            node.args=[ast.Name(id='_original_owner',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())]
    ast.fix_missing_locations(tree)
    scope=dict(function.__globals__,_original_owner=owner)
    scope.update(inspect.getclosurevars(function).nonlocals)
    exec(compile(tree,'<diagnostic precise pacing>','exec'),scope)
    return scope[function.__name__]


def install(trainer,mode):
    if mode not in ('sleep','precise'):raise ValueError('Unknown pacing mode')
    if getattr(trainer,'_latency_pacing_installed',False):return
    original=trainer.build_env_class
    def build():
        base=original()
        owners=[c for c in base.__mro__ if c.__name__=='NativeCroppedV4Env' and 'step' in c.__dict__]
        if len(owners)!=1:raise RuntimeError('Unexpected pacing owner')
        owner=owners[0];owner.step=replace_wait(owner.step,owner)
        class PacedEnv(base):
            def __init__(self,*args,**kwargs):
                self._latency_waiter=DeadlineWait(mode=='precise')
                try:super().__init__(*args,**kwargs)
                except BaseException:
                    self._latency_waiter.close();raise
            def _latency_rate_wait(self,seconds):self._latency_waiter.wait(seconds)
            def close(self):
                try:return super().close()
                finally:
                    self._latency_waiter.close()
                    (Path(self.output_dir)/'decision_pacing.json').write_text(json.dumps(dict(
                        requested_mode=mode,mode=self._latency_waiter.mode,calls=self._latency_waiter.calls,
                        samples=list(self._latency_waiter.rows)),indent=2))
        return PacedEnv
    trainer.build_env_class=build
    trainer._latency_pacing_installed=True
