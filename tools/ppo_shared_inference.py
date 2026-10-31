"""Versioned shared inference, with per-worker slots and bounded policy waits."""
import hashlib
import struct
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from multiprocessing.connection import wait
import numpy as np
from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_shared_arrays import SharedArrays


def validate_reply(reply, serial, version, digest, deadline):
    lateness=time.perf_counter()-deadline
    if lateness>0:
        raise GuardUncertainError(f'Shared inference reply missed its deadline by {lateness*1000:.1f}ms; no action dispatched')
    if (reply.get('serial')!=serial or reply.get('version')!=version or reply.get('policy_hash')!=digest):
        raise GuardUncertainError('Shared inference returned a mismatched policy/request')
    if reply.get('error'):
        raise GuardUncertainError('Shared inference: '+reply['error'])


def forward_with_probabilities(policy,observation,deterministic=False):
    """Read the distribution from this forward, without a second pass or draw."""
    original=policy._get_action_dist_from_latent
    had_override='_get_action_dist_from_latent' in policy.__dict__
    captured=[]
    def observe(*args,**kwargs):
        distribution=original(*args,**kwargs)
        captured.append(distribution.distribution.probs)
        return distribution
    policy._get_action_dist_from_latent=observe
    try:result=policy(observation,deterministic=deterministic)
    finally:
        if had_override:policy._get_action_dist_from_latent=original
        else:delattr(policy,'_get_action_dist_from_latent')
    return result,captured[-1]


class InferenceClient:
    def __init__(self,resource,timeout_ms):
        self.policy_connection=resource['policy']
        self.detector_connection=resource['detector']
        self.policy_slot=SharedArrays(descriptor=resource['policy_slot'])
        self.detector_slot=SharedArrays(descriptor=resource['detector_slot'])
        self.timeout=timeout_ms/1000
        self.version=-1
        self.digest=None
        self.serial=0
        self.detector_serial=0
        self.policy_requests=self.detector_requests=0
        self.last_probabilities=None

    def predict(self,observation,kind='sample',deterministic=False):
        deadline=time.perf_counter()+self.timeout
        self.serial+=1
        for key,value in observation.items():
            source=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
            target=self.policy_slot.arrays[key]
            if source.shape!=target.shape or source.dtype!=target.dtype:
                raise GuardUncertainError('Shared policy observation schema mismatch')
            target[:]=source
        struct.pack_into('<Q',self.policy_slot.memory.buf,0,self.serial)
        self.policy_connection.send(dict(serial=self.serial,version=self.version,policy_hash=self.digest,
            deadline=deadline,kind=kind,deterministic=bool(deterministic)))
        if not self.policy_connection.poll(max(0.,deadline-time.perf_counter())):
            raise GuardUncertainError('Shared policy inference deadline exceeded; no action dispatched')
        reply=self.policy_connection.recv()
        validate_reply(reply,self.serial,self.version,self.digest,deadline)
        self.policy_requests+=1
        values=np.asarray(reply['values'],np.float32)
        if values.shape!=(1,1) or not np.isfinite(values).all():
            raise GuardUncertainError('Invalid shared critic value')
        if kind=='value':return values
        action=np.asarray(reply['actions'])
        log_prob=np.asarray(reply['log_probs'],np.float32)
        if (action.shape!=(1,) or action.dtype.kind not in 'iu' or not ((0<=action)&(action<5)).all()
                or log_prob.shape!=(1,) or not np.isfinite(log_prob).all()):
            raise GuardUncertainError('Invalid shared policy action/log probability')
        probabilities=np.asarray(reply['probabilities'],np.float32)
        if (probabilities.shape!=(1,5) or not np.isfinite(probabilities).all() or
                (probabilities<0).any() or not np.allclose(probabilities.sum(axis=1),1.,atol=1e-5)):
            raise GuardUncertainError('Invalid shared policy probabilities')
        self.last_probabilities=(self.version,self.serial,time.perf_counter(),probabilities[0].copy())
        return action.astype(np.int64),values,log_prob

    def close(self):
        self.policy_connection.close()
        self.detector_connection.close()
        self.policy_slot.close()
        self.detector_slot.close()


class RemotePolicy:
    """Collector-compatible CPU proxy. Only the shared service holds its weights."""
    def __init__(self,client,observation_space):
        import torch
        self.client=client
        self.observation_space=observation_space
        self.device=torch.device('cpu')

    def bind_version(self,digest):
        if digest!=self.client.digest or self.client.version<0:
            raise GuardUncertainError('Collector/shared policy version mismatch')

    def set_training_mode(self,mode):
        if mode:raise RuntimeError('Remote inference policy cannot optimize')

    def diagnostic_snapshot(self):
        cached=self.client.last_probabilities
        if cached is None or cached[0]!=self.client.version:return None
        version,serial,at,probs=cached
        from rs_v61_features import ACTION_NAMES
        return dict(probabilities=[round(float(v),5) for v in probs],
            max_action=ACTION_NAMES[int(np.argmax(probs))],max_probability=round(float(probs.max()),5),
            entropy=round(float(-(probs*np.log(np.maximum(probs,1e-9))).sum()),5),
            source='last completed policy decision; not a prediction on the following observation',
            policy_version=version,request_serial=serial,age_ms=(time.perf_counter()-at)*1000)

    def __call__(self,observation,deterministic=False):
        import torch
        return tuple(torch.from_numpy(v) for v in self.client.predict(observation,deterministic=deterministic))

    def predict_values(self,observation):
        import torch
        return torch.from_numpy(self.client.predict(observation,kind='value'))

    def obs_to_tensor(self,observation):
        import torch
        output={}
        for key,value in observation.items():
            value=np.asarray(value)
            shape=self.observation_space[key].shape
            output[key]=torch.as_tensor(value.reshape((-1,*shape)))
        return output,True


class RemoteDetector:
    task='segment'
    def __init__(self,client):
        from rs_v5_shadow_segmentation import V5_CLASSES
        self.names=dict(enumerate(V5_CLASSES))
        self.client=client

    def detect_frame(self,frame,captured_at,episode,request_id,imgsz,conf):
        client=self.client
        client.detector_serial+=1
        serial=client.detector_serial
        target=client.detector_slot.arrays['pixels'].reshape(-1)
        if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or frame.size>target.size:
            raise GuardUncertainError('Shared detector source geometry mismatch')
        target[:frame.size]=frame.reshape(-1)
        struct.pack_into('<Q',client.detector_slot.memory.buf,0,serial)
        client.detector_connection.send(dict(serial=serial,shape=frame.shape,captured_at=captured_at,
            episode=episode,request_id=request_id,imgsz=imgsz,conf=conf))
        if not client.detector_connection.poll(5):
            raise GuardUncertainError('Shared detector service timed out')
        result=client.detector_connection.recv()
        if result.get('serial')!=serial or result.get('error'):
            raise GuardUncertainError('Shared detector failed or returned another source')
        result.pop('serial')
        if result['episode']!=episode or result['request_id']!=request_id or result['captured_at']!=captured_at:
            raise GuardUncertainError('Shared detector source lineage mismatch')
        result['source_frame']=frame
        client.detector_requests+=1
        return result


def serve(control,resources,options):
    slots=[]
    detector_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='shared-yolo')
    detector_job=None
    detector_queue=deque()
    try:
        import torch
        from stable_baselines3 import PPO
        from tools.ppo_parallel_learning import install_policy,policy_arrays,policy_hash
        from tools.ppo_reaction_metrics import warm_policy
        from rs_v5_shadow_segmentation import detect_v5,check_model_names
        torch.set_num_threads(1)
        model=PPO.load(options['checkpoint'],device=options.get('inference_device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
        model.policy.optimizer.state.clear()
        model.rollout_buffer=None
        model.policy.set_training_mode(False)
        warm_policy(model.policy,model.observation_space)
        # New batch shapes must not incur cold GPU setup inside an action deadline.
        with torch.no_grad():
            for size in range(2,len(resources)+1):
                batch={key:torch.zeros((size,*space.shape),dtype=torch.from_numpy(np.empty((),space.dtype)).dtype,
                                       device=model.device) for key,space in model.observation_space.spaces.items()}
                model.policy(batch,deterministic=True)
                model.policy.predict_values(batch)
        if model.device.type=='cuda':torch.cuda.synchronize(model.device)
        detector=None
        if options['detector_mode']=='on':
            from ultralytics import YOLO
            detector=YOLO(options['detector'])
            check_model_names(detector)
            height,width=(945,1920) if options.get('framing_profile')=='canvas_v1' else (937,1904)
            detect_v5(detector,np.zeros((height,width,3),np.uint8),time.perf_counter(),0,0,960,.20)
        policy_channels={};detector_channels={}
        for resource in resources:
            policy_slot=SharedArrays(descriptor=resource['policy_slot'])
            detector_slot=SharedArrays(descriptor=resource['detector_slot'])
            slots.extend([policy_slot,detector_slot])
            policy_channels[resource['policy']]=policy_slot
            detector_channels[resource['detector']]=detector_slot
        version=-1;digest=None
        control.send(dict(op='ready'))
        while True:
            available=wait([control,*policy_channels,*detector_channels],timeout=.002 if detector_job is not None else None)
            if control in available:
                message=control.recv()
                if message['op']=='stop':break
                if message['op']!='install' or message['version']!=version+1:
                    raise ValueError('Invalid shared inference version advance')
                install_policy(model.policy,message['weights'],message['policy_hash'])
                version,digest=message['version'],message['policy_hash']
                control.send(dict(op='installed',version=version,policy_hash=digest))
            ready=[c for c in available if c in policy_channels]
            # Batch only already-ready requests; optional wait is tightly bounded.
            end=time.perf_counter()+options.get('inference_batch_wait_ms',0)/1000
            while ready and time.perf_counter()<end:
                more=wait([c for c in policy_channels if c not in ready],timeout=max(0.,end-time.perf_counter()))
                if not more:break
                ready.extend(more)
            requests=[]
            for connection in ready:
                try:request=connection.recv()
                except (EOFError,OSError):
                    policy_channels.pop(connection)
                    connection.close()
                    continue
                if (request['version']!=version or request['policy_hash']!=digest or
                        request['deadline']<time.perf_counter()):
                    connection.send(dict(serial=request['serial'],version=version,policy_hash=digest,error='stale request'))
                    continue
                slot=policy_channels[connection]
                if struct.unpack_from('<Q',slot.memory.buf)[0]!=request['serial']:
                    raise ValueError('Shared policy slot publication mismatch')
                observation={key:value.copy() for key,value in slot.arrays.items()}
                if struct.unpack_from('<Q',slot.memory.buf)[0]!=request['serial']:
                    raise ValueError('Shared policy slot changed while reading')
                requests.append((connection,request,observation))
            for kind in ('sample','value'):
                for deterministic in (False,True):
                    group=[r for r in requests if r[1]['kind']==kind and r[1]['deterministic']==deterministic]
                    if not group:continue
                    batch={key:torch.as_tensor(np.concatenate([r[2][key] for r in group]),device=model.device)
                           for key in group[0][2]}
                    with torch.no_grad():
                        if kind=='value':values=model.policy.predict_values(batch).cpu().numpy()
                        else:
                            (actions,values,log_probs),probabilities=forward_with_probabilities(model.policy,batch,deterministic)
                            actions,values,log_probs=actions.cpu().numpy(),values.cpu().numpy(),log_probs.cpu().numpy()
                            probabilities=probabilities.cpu().numpy()
                    for i,(connection,request,_) in enumerate(group):
                        reply=dict(serial=request['serial'],version=version,policy_hash=digest,values=values[i:i+1])
                        if kind=='sample':reply.update(actions=actions[i:i+1],log_probs=log_probs[i:i+1],probabilities=probabilities[i:i+1])
                        connection.send(reply)
            # Prioritize newly arrived policy requests before starting YOLO.
            pending_policy=False
            for c in list(policy_channels):
                try:pending_policy=pending_policy or c.poll()
                except (OSError,EOFError):
                    policy_channels.pop(c)
                    c.close()
            if detector_job is not None and detector_job[2].done():
                connection,request,future=detector_job
                result=future.result()
                result.pop('source_frame')
                try:connection.send(dict(serial=request['serial'],**result))
                except (OSError,EOFError):pass
                detector_job=None
            for connection in available:
                if connection not in detector_channels:continue
                try:request=connection.recv()
                except (EOFError,OSError):
                    detector_channels.pop(connection)
                    connection.close()
                    continue
                if detector is None:raise ValueError('Detector disabled in shared service')
                slot=detector_channels[connection]
                if struct.unpack_from('<Q',slot.memory.buf)[0]!=request['serial']:
                    raise ValueError('Shared detector publication mismatch')
                shape=tuple(request['shape'])
                if len(shape)!=3 or shape[-1]!=3 or any(n<=0 for n in shape) or np.prod(shape)>slot.arrays['pixels'].size:
                    raise ValueError('Invalid shared detector dimensions')
                if any(c is connection for c,_ in detector_queue):raise ValueError('More than one outstanding detector request')
                detector_queue.append((connection,request))
            if detector_job is None and detector_queue and not pending_policy:
                connection,request=detector_queue.popleft()
                slot=detector_channels[connection]
                shape=tuple(request['shape'])
                frame=slot.arrays['pixels'].reshape(-1)[:np.prod(shape)].reshape(shape).copy()
                if struct.unpack_from('<Q',slot.memory.buf)[0]!=request['serial']:
                    raise ValueError('Shared detector slot changed while reading')
                detector_job=(connection,request,detector_pool.submit(detect_v5,detector,frame,
                    request['captured_at'],request['episode'],request['request_id'],request['imgsz'],request['conf']))
    except BaseException as exc:
        try:control.send(dict(op='error',error=repr(exc),traceback=traceback.format_exc()))
        except (OSError,EOFError):pass
        raise
    finally:
        detector_pool.shutdown(wait=True,cancel_futures=True)
        for slot in slots:slot.close()


class InferenceHost:
    def __init__(self,context,observation_space,count,options):
        self.slots=[];self.clients=[];resources=[]
        self.control,child=context.Pipe()
        for _ in range(count):
            policy_slot=SharedArrays({key:((1,*space.shape),space.dtype) for key,space in observation_space.spaces.items()})
            detector_slot=SharedArrays({'pixels':((945,1920,3),np.uint8)})
            self.slots.extend([policy_slot,detector_slot])
            pclient,pserver=context.Pipe();dclient,dserver=context.Pipe()
            descriptions=dict(policy_slot=policy_slot.descriptor,detector_slot=detector_slot.descriptor)
            self.clients.append(dict(policy=pclient,detector=dclient,**descriptions))
            resources.append(dict(policy=pserver,detector=dserver,**descriptions))
        self.process=context.Process(target=serve,args=(child,resources,options),name='ppo-shared-inference')
        self.process.start()
        child.close()
        for r in resources:r['policy'].close();r['detector'].close()

    def expect(self,op,timeout):
        if not self.control.poll(timeout):raise TimeoutError('Shared inference coordinator timeout')
        result=self.control.recv()
        if result.get('op')!=op:raise RuntimeError('Shared inference failed: '+str(result))
        return result

    def install(self,version,weights,digest,timeout):
        self.control.send(dict(op='install',version=version,weights=weights,policy_hash=digest))
        response=self.expect('installed',timeout)
        if response['version']!=version or response['policy_hash']!=digest:
            raise ValueError('Shared policy installation acknowledgement mismatch')

    def close(self):
        try:self.control.send(dict(op='stop'))
        except (OSError,EOFError,BrokenPipeError):pass
        self.process.join(timeout=5)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        self.control.close()
        for client in self.clients:
            client['policy'].close();client['detector'].close()
        for slot in self.slots:slot.close()
