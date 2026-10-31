"""Experimental LOCAL Practice workers with one shared, pause-aware optimizer.

Each worker collects a whole trajectory independently, then stays paused until
the central optimizer commits the cohort. No lockstep waits during gameplay.
"""
import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import time
import traceback


def worker_main(index, display, options, connection, abort, resources=None):
    output = Path(options['report'])/f'worker_{index}'
    output.mkdir()
    raw = monitored = callback = collector = None
    error = None
    rollout_slot=inference_client=publisher=None
    resources=resources or {}
    with (output/'worker.log').open('w', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                import numpy as np
                import torch
                from stable_baselines3 import PPO
                import train_rs_v612_ppo as trainer
                if options.get('headless'):
                    from tools.ppo_headless_env import install
                    install(trainer,options.get('headless_capture','playwright'),
                            profile=options.get('profile_reaction',False),
                            electron_transport=options.get('electron_transport','socket'),
                            electron_window=resources.get('electron_window'))
                    time.sleep(index*2.)  # Avoid simultaneous Unity/server startup bursts.
                from tools.ppo_parallel_learning import install_policy, rollout_packet
                from tools.ppo_settled_learning import SettledPPO
                steps = options['worker_steps']
                args = trainer.parse_args([
                    '--mode', 'train', '--run-name', output.name, '--resume', options['checkpoint'],
                    '--steps', str(options['steps']//options['workers']), '--n-steps', str(steps),
                    '--max-minutes', '480', '--pause-ppo-optimizer', '--resync-rollouts',
                    '--audit-reward-timing', '--settled-learning', '--reuse-observation-guard',
                    '--resolution', '512x288', '--fps', '30', '--rate-mode', 'uncapped',
                    '--press-delay-ms', '5', '--state-model', options['state_model'],
                    '--detector', options['detector'], '--background-display', str(display),
                    '--detector-mode',options.get('detector_mode','on'),
                    *(['--cache-detector-features'] if options.get('cache_detector_features') else []),
                    '--background-capture', options.get('background_capture', 'mss'),
                    '--episode-max-seconds', '300',
                    '--framing-profile', options.get('framing_profile','legacy'),
                    *(['--canvas-policy-input'] if options['canvas_policy_input'] else [])])
                if options.get('opencv_threads') is not None:
                    import cv2
                    cv2.setNumThreads(options['opencv_threads'])
                detector_service=None
                if resources.get('inference'):
                    from tools.ppo_shared_inference import InferenceClient,RemoteDetector
                    inference_client=InferenceClient(resources['inference'],options['inference_timeout_ms'])
                    if options.get('detector_mode','on')=='on':detector_service=RemoteDetector(inference_client)
                startup_lock=resources.get('startup_lock')
                if startup_lock is None:
                    monitored, raw = trainer.create_live_env(args, output,detector_service=detector_service)
                else:
                    # One host must not synchronously read back frames while
                    # several Unity instances are compiling/loading together.
                    with startup_lock:
                        if abort.is_set():raise RuntimeError('Cohort aborted during browser initialization')
                        monitored, raw = trainer.create_live_env(args, output,detector_service=detector_service)
                if options.get('reuse_frame_state'):
                    from tools.ppo_frame_state_cache import install_frame_state_cache
                    install_frame_state_cache(raw)
                collector = PPO.load(options['checkpoint'], env=monitored,
                                     device='cpu' if inference_client is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
                if collector.n_steps != steps:
                    raise ValueError('Worker/source rollout settings disagree')
                collector.__class__ = SettledPPO
                from tools.ppo_settled_learning import TimedDictBuffer
                buffer_class=TimedDictBuffer
                if options.get('compact_observations'):
                    from tools.ppo_compact_buffer import CompactTimedDictBuffer
                    buffer_class=CompactTimedDictBuffer
                collector.configure_settled(raw, output, options['reference_seconds'],buffer_class=buffer_class)
                # Only the coordinator optimizes. Replica momentum is never used.
                collector.policy.optimizer.state.clear()
                from tools.ppo_reaction_metrics import warm_policy
                if inference_client is None:warm_policy(collector.policy,collector.observation_space)
                else:
                    from tools.ppo_shared_inference import RemotePolicy
                    collector.policy=RemotePolicy(inference_client,collector.observation_space)
                if resources.get('rollout_slot'):
                    from tools.ppo_rollout_transport import RolloutSlot
                    rollout_slot=RolloutSlot(descriptor=resources['rollout_slot'])
                if options.get('preview_port'):
                    from tools.ppo_worker_preview import PreviewPublisher
                    publisher=PreviewPublisher(output,options.get('preview_fps',1))
                if options.get('profile_reaction'):
                    from tools.ppo_pipeline_profile import PipelineProfiler
                    PipelineProfiler(raw,output,collector.policy.device)
                connection.send(dict(op='ready', worker=index))
                initialized = False
                version_expected = 0
                while True:
                    command = connection.recv()
                    if command['op'] == 'stop':
                        break
                    if command['op'] != 'collect' or command['version'] != version_expected or abort.is_set():
                        raise RuntimeError('Invalid collection command or aborted cohort')
                    if inference_client is not None:
                        inference_client.version=command['version']
                        inference_client.digest=command['policy_hash']
                    install_policy(collector.policy, command['weights'], command['policy_hash'])
                    if not initialized:
                        callback = trainer.build_callback(args, output, raw=raw)
                        original_step = callback._on_step
                        episode_count = [0]
                        reaction_status=[{}]
                        reaction_updated=[0.]
                        def preview_status(phase,steps):
                            recent=callback.episode_durations[-25:]
                            if time.perf_counter()-reaction_updated[0] >= 5:
                                from tools.ppo_reaction_metrics import latency_summary
                                reaction_status[0]=latency_summary(list(getattr(raw,'_reaction_samples',[])))
                                reaction_updated[0]=time.perf_counter()
                            return dict(phase=phase,steps=steps,episodes=len(callback.episode_durations),
                                recent_mean_seconds=sum(recent)/len(recent) if recent else None,
                                reaction_latency=reaction_status[0],
                                sampled_action_counts=dict(callback.action_total))
                        def publish_preview(image,phase,steps):
                            status=preview_status(phase,steps)
                            publisher.offer(image,status)
                        def step_callback():
                            allowed = bool(original_step())
                            now = time.perf_counter()
                            if len(callback.episode_durations) > episode_count[0]:
                                try:
                                    with (output/'survival_episodes.jsonl').open('a',encoding='utf8') as episodes:
                                        for duration in callback.episode_durations[episode_count[0]:]:
                                            info = next((i for i in callback.locals.get('infos',[]) if 'episode' in i),{})
                                            episodes.write(json.dumps(dict(at=now,duration=duration,
                                                reason=info.get('termination_reason'),worker=index))+'\n')
                                except OSError as exc:
                                    print(f'[DASHBOARD] Episode metric unavailable: {exc}',flush=True)
                                episode_count[0] = len(callback.episode_durations)
                            if publisher is not None:
                                observation = callback.locals.get('new_obs', collector._last_obs)
                                publish_preview(observation['image'][0],'collecting',
                                    collector._settled_stats['collected_transitions']+1)
                            return allowed and not abort.is_set()
                        callback._on_step = step_callback
                        _, callback = collector._setup_learn(total_timesteps=args.steps,
                            callback=callback, reset_num_timesteps=False)
                        callback.on_training_start({}, {})
                        initialized = True
                    if not collector.collect_rollouts(collector.env, callback, collector.rollout_buffer, steps):
                        raise RuntimeError('Worker collection stopped before its rollout completed')
                    if (getattr(raw, '_ppo_pause_pending', None) is None
                            and not getattr(raw, '_ppo_boundary_menu_pending', False)):
                        raise RuntimeError('Worker did not reach a verified frozen/menu barrier')
                    if options.get('preview_port'):
                        publish_preview(collector._last_obs['image'][0],'paused for shared update',
                                        collector._settled_stats['collected_transitions'])
                    if rollout_slot is not None:
                        packet=rollout_slot.publish(collector.rollout_buffer,index,command['version'],command['policy_hash'])
                    else:
                        packet = rollout_packet(collector.rollout_buffer, index,
                                                command['version'], command['policy_hash'])
                    connection.send(dict(op='rollout', **packet))
                    del packet
                    commit = connection.recv()
                    if commit['op'] != 'commit' or commit['version'] != command['version']:
                        raise RuntimeError('Collected experience was not acknowledged by its optimizer')
                    used = commit['used_indices']
                    if len(set(used)) != len(used) or any(i < 0 or i >= steps for i in used):
                        raise RuntimeError('Invalid optimizer coverage acknowledgement')
                    if rollout_slot is not None:rollout_slot.acknowledge(command['version'])
                    collector._settled_trained_this_run += steps
                    stats = collector._settled_stats
                    stats['completed_rollout_transitions'] += steps
                    stats['optimized_transitions'] += len(used)
                    stats['optimizer_sample_presentations'] += commit['sample_presentations']
                    collector._record(dict(event='benchmark_observed' if options.get('benchmark_only') else 'optimized', through_transition=collector.num_timesteps,
                        optimized_transitions=stats['optimized_transitions'], used_indices=used,
                        epochs=commit['epochs'], policy_version=command['version']))
                    version_expected += 1
                    if commit['stop']:
                        break
                connection.send(dict(op='finished', worker=index,
                    transitions=collector._settled_stats,
                    guard_hard_stopped=raw._guard_hard_stopped))
            except BaseException as exc:
                error = repr(exc)
                traceback.print_exc()
                abort.set()
                try:
                    connection.send(dict(op='error', worker=index, error=error))
                except (OSError, EOFError):
                    pass
            finally:
                if callback is not None and callback.log_handle is not None:
                    callback.log_handle.close()
                if raw is not None:
                    if collector is not None:
                        stats = collector._settled_stats
                        stats['discarded_partial_transitions'] = max(0,
                            stats['collected_transitions']-stats['completed_rollout_transitions'])
                        stats['completed_interactions_not_ledgered'] = max(0,
                            collector.num_timesteps-collector._settled_initial_timesteps-stats['collected_transitions'])
                    summary = dict(error=error, guard_hard_stopped=raw._guard_hard_stopped,
                        learning_transitions=getattr(collector, '_settled_stats', {}),
                        policy_replica_only=True, optimizer_updates_in_worker=0)
                    summary['detector_mode']=options.get('detector_mode','on')
                    images=getattr(getattr(collector,'rollout_buffer',None),'observations',{}).get('image')
                    if hasattr(images,'stored_bytes'):
                        summary['compact_storage']=dict(stored_bytes=images.stored_bytes,
                            dense_equivalent_bytes=int(np.prod(images.shape)),unique_rgb_frames=images.count)
                    geometry_cache=getattr(raw,'_detector_feature_cache',None)
                    if geometry_cache is not None:
                        summary['detector_feature_cache']=dict(hits=geometry_cache.hits,misses=geometry_cache.misses)
                    if inference_client is not None:
                        summary['shared_inference']=dict(policy_requests=inference_client.policy_requests,
                            detector_requests=inference_client.detector_requests)
                    if publisher is not None:
                        publisher.close()
                        summary['preview_encoder']=dict(published=publisher.published,dropped=publisher.dropped)
                    from tools.ppo_reaction_metrics import latency_summary
                    summary['reaction_latency']=latency_summary(list(getattr(raw,'_reaction_samples',[])))
                    source=getattr(raw,'_headless_source',None)
                    summary['capture_backend']=getattr(raw,'_headless_backend',None)
                    summary['frame_state_cache']=dict(hits=getattr(raw,'frame_state_cache_hits',0),
                        misses=getattr(raw,'frame_state_cache_misses',0),enabled=bool(options.get('reuse_frame_state')))
                    if hasattr(source,'ready'):
                        summary['electron_runtime']={key:value for key,value in source.ready.items() if key!='token'}
                        summary['raw_frame_reads']=source.reads
                        summary['stale_frame_retries']=source.stale_frame_retries
                    if hasattr(source,'stream_reads'):
                        summary['stream_reads']=source.stream_reads
                        summary['surface_fallbacks']=source.fallback_reads
                    if getattr(raw,'_pipeline_profiler',None) is not None:
                        try:
                            raw._pipeline_profiler.save()
                        except Exception as profile_exc:
                            summary['profile_error']=repr(profile_exc)
                    (output/'result.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
                if monitored is not None:
                    monitored.close()
                elif raw is not None:
                    raw.close()
                if inference_client is not None:inference_client.close()
                if rollout_slot is not None:rollout_slot.close()
                connection.close()


def receive_all(connections, expected, timeout, abort):
    pending = dict(enumerate(connections))
    results = {}
    deadline = time.perf_counter()+timeout
    while pending:
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f'Worker timeout waiting for {expected}')
        available = wait(list(pending.values()), timeout=min(remaining, 1.))
        if abort.is_set() and not available:
            raise RuntimeError('A worker aborted; inspect worker logs')
        for connection in available:
            index = next(i for i, c in pending.items() if c is connection)
            message = connection.recv()
            if message.get('op') != expected or message.get('worker') != index:
                raise RuntimeError(f'Worker {index}: expected {expected}, received {message.get("error",message.get("op"))}')
            results[index] = message
            del pending[index]
    return [results[i] for i in range(len(connections))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--displays', type=int, nargs='+', required=True)
    parser.add_argument('--headless', action='store_true', help='Experimental browser capture; displays are worker IDs, no monitors required')
    parser.add_argument('--profile-reaction',action='store_true',
                        help='Buffer per-decision capture/preprocessing/inference/guard timings for a short diagnostic')
    parser.add_argument('--electron-transport',choices=['socket','shared_memory'],default='socket')
    parser.add_argument('--reuse-frame-state',action='store_true',help='Reuse deterministic classification only for the identical capture/model/confidence')
    parser.add_argument('--rollout-transport',choices=['pipe','shared_memory'],default='pipe')
    parser.add_argument('--compact-observations',action='store_true',help='Store exact unique RGB frames and reconstruct policy minibatches')
    parser.add_argument('--cache-detector-features',action='store_true')
    parser.add_argument('--opencv-threads',type=int,default=None,help='Optional per-worker OpenCV thread cap; 1 avoids nested thread oversubscription')
    parser.add_argument('--shared-inference',action='store_true',help='One versioned policy/detector service for all workers')
    parser.add_argument('--inference-timeout-ms',type=float,default=50.)
    parser.add_argument('--inference-batch-wait-ms',type=float,default=0.)
    parser.add_argument('--electron-host',choices=['per_worker','shared'],default='per_worker')
    parser.add_argument('--benchmark-only',action='store_true',help='Frozen-policy pipeline diagnostic; no optimizer updates or saved checkpoints')
    parser.add_argument('--headless-capture',choices=['playwright','cdp','stream','electron'],default='playwright',
                        help='Headless browser image backend; cdp uses a validated direct surface clip')
    parser.add_argument('--framing-profile', choices=['legacy','canvas_v1'], default='legacy')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=65536, help='Total transitions across workers; complete cohorts only')
    parser.add_argument('--max-minutes', type=float, default=180.)
    parser.add_argument('--save-every-minutes', type=float, default=10.)
    parser.add_argument('--worker-timeout', type=float, default=180.)
    parser.add_argument('--state-model', type=Path, default=Path('checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt'))
    parser.add_argument('--detector', type=Path, default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    parser.add_argument('--detector-mode',choices=['on','off'],default='on')
    parser.add_argument('--canvas-policy-input', action='store_true')
    parser.add_argument('--background-capture', choices=['mss','dxgi'], default='mss')
    parser.add_argument('--preview-port', type=int, default=None)
    parser.add_argument('--preview-fps',type=float,default=1.,help='Preview cap 1..30; newest collected images only')
    parser.add_argument('--end-plot',choices=['auto','show','save','off'],default='auto',
                        help='Post-training survival plot: auto opens without dashboard; show forces a window; save writes PNG only; off disables. Auto skips frozen benchmarks.')
    parser.add_argument('--telemetry', action='store_true', help='Record CPU, RAM and NVIDIA usage once per second')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    count = len(args.displays)
    if args.compact_observations and args.rollout_transport!='shared_memory':
        parser.error('Compact observations require shared-memory rollout transport')
    if not 1<=args.preview_fps<=30 or not 1<=args.inference_timeout_ms<=100 or not 0<=args.inference_batch_wait_ms<=2:
        parser.error('Invalid preview/inference limits')
    if args.opencv_threads is not None and not 1<=args.opencv_threads<=16:parser.error('OpenCV threads must be 1..16')
    if args.electron_host=='shared' and (not args.headless or args.headless_capture!='electron' or args.profile_reaction):
        parser.error('Shared Electron host requires headless Electron without detailed profiling')
    if args.headless_capture=='electron' and not args.headless:
        parser.error('Experimental Electron capture requires --headless')
    if args.electron_transport=='shared_memory' and (not args.headless or args.headless_capture!='electron'):
        parser.error('Shared-memory transport requires --headless --headless-capture electron')
    if not 1 <= count <= (6 if args.headless else 2) or len(set(args.displays)) != count:
        parser.error('Supports up to six headless workers or two dedicated displays')
    if not 0 < args.max_minutes <= 480 or args.save_every_minutes <= 0 or args.worker_timeout < 15:
        parser.error('Invalid time limits')
    if args.preview_port is not None and not 1024 <= args.preview_port <= 65535:
        parser.error('Invalid preview port')
    for path in (args.checkpoint, args.state_model,*([args.detector] if args.detector_mode=='on' else [])):
        if not path.is_file():
            parser.error(f'Missing local asset: {path}')
    if args.report.exists():
        parser.error('Report directory exists; choose a new name')
    import mss
    from tools.ppo_background_display import select_display
    if not args.headless:
        with mss.MSS() as capture:
            for index in args.displays:
                select_display(capture.monitors, index)
    import torch
    import train_rs_v3_ppo  # checkpoint feature class
    from stable_baselines3 import PPO
    from tools.ppo_parallel_learning import configure_learner, merge_rollouts, policy_arrays, policy_hash
    from tools.ppo_settled_learning import reference_seconds_for, audit_settled_ledger
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    model = PPO.load(str(args.checkpoint), device='cuda' if torch.cuda.is_available() else 'cpu')
    worker_steps = model.n_steps
    cohort = worker_steps*count
    if args.steps < cohort or args.steps % cohort:
        parser.error(f'--steps must be divisible by {cohort} ({count} workers x {worker_steps})')
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    initial_policy_digest=policy_hash(policy_arrays(model.policy))
    args.report.mkdir(parents=True)
    reference = reference_seconds_for(model, None)
    from tools.ppo_settled_learning import TimedDictBuffer
    buffer_class=TimedDictBuffer
    if args.compact_observations:
        from tools.ppo_compact_buffer import CompactTimedDictBuffer
        buffer_class=CompactTimedDictBuffer
    configure_learner(model, args.report, count, reference,buffer_class=buffer_class)
    model.framing_profile = args.framing_profile
    model.detector_mode=args.detector_mode
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    options.update(worker_steps=worker_steps, workers=count, reference_seconds=reference)
    config = dict(options, aggregate_transitions_per_update=cohort, learning_rate=model.lr_schedule(1.),
                  batch_size=model.batch_size, n_epochs=model.n_epochs, gamma=model.gamma,
                  gae_lambda=model.gae_lambda, source_sha256=source_hash,
                  optimizer='disabled frozen-policy diagnostic' if args.benchmark_only else 'one shared masked PPO',
                  policy_inference='shared versioned service' if args.shared_inference else 'per-worker synchronized replicas')
    (args.report/'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    report = dict(status='STARTING', rounds=0, optimized_transitions=0,processed_transitions=0,source_sha256=source_hash,
                  benchmark_only=args.benchmark_only,model_updated=False)
    def save_report():
        (args.report/'parallel_training.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    save_report()
    ctx = mp.get_context('spawn')
    abort = ctx.Event()
    connections, processes = [], []
    preview = None
    sampler = None
    survival_metrics = None
    started = None
    slots=[]
    inference_host=electron_group=None
    resources=[{} for _ in range(count)]
    try:
        if args.rollout_transport=='shared_memory':
            from tools.ppo_rollout_transport import RolloutSlot
            template=buffer_class(worker_steps,model.observation_space,model.action_space,device='cpu',n_envs=1)
            for i in range(count):
                slot=RolloutSlot(template);slots.append(slot);resources[i]['rollout_slot']=slot.descriptor
            del template
        if args.shared_inference:
            from tools.ppo_shared_inference import InferenceHost
            inference_host=InferenceHost(ctx,model.observation_space,count,options)
            inference_host.expect('ready',args.worker_timeout)
            for i in range(count):resources[i]['inference']=inference_host.clients[i]
        if args.electron_host=='shared':
            from tools.ppo_electron_capture import ElectronGroup
            width,height=(1920,945) if args.framing_profile=='canvas_v1' else (1904,937)
            electron_group=ElectronGroup(args.report/'electron_group',count,width,height,args.electron_transport)
            startup_lock=ctx.Lock()
            for i in range(count):
                resources[i]['electron_window']=electron_group.windows[i]
                resources[i]['startup_lock']=startup_lock
        if args.preview_port is not None:
            from tools.ppo_worker_preview import start_preview
            preview = start_preview(args.report, count, args.preview_port,args.preview_fps)
            print(f'Worker grid: http://127.0.0.1:{args.preview_port}/', flush=True)
        for index, display in enumerate(args.displays):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker_main, args=(index, display, options, child, abort,resources[index]))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        if args.telemetry:
            import os
            from tools.ppo_resource_sampler import ResourceSampler
            sampler = ResourceSampler(args.report/'system_usage.jsonl',
                                      [os.getpid(), *[p.pid for p in processes]])
            sampler.start()
        receive_all(connections, 'ready', args.worker_timeout, abort)
        started = last_save = time.perf_counter()
        started_wall=time.time()
        runtime=dict(rounds=0,optimized_transitions=0,elapsed_seconds=0,optimized_per_second=0,
            started_wall=started_wall,requested_seconds=args.max_minutes*60,finished=False,
            workers=count,worker_rollout=worker_steps,aggregate_rollout=cohort,
            learning_rate=model.lr_schedule(1.),framing_profile=args.framing_profile,
            benchmark_only=args.benchmark_only,detector_mode=args.detector_mode,
            rollout_transport=args.rollout_transport,compact_observations=args.compact_observations,
            shared_inference=args.shared_inference,electron_host=args.electron_host)
        (args.report/'runtime.json').write_text(json.dumps(runtime),encoding='utf8')
        from tools.ppo_survival_metrics import SurvivalMetrics
        from tools.ppo_terminal_stats import format_terminal_stats
        survival_metrics = SurvivalMetrics(args.report,count,started,args.max_minutes*60)
        survival_metrics.start()
        version = 0
        report['status'] = 'RUNNING'
        while report['processed_transitions'] < args.steps:
            collection_started=time.perf_counter()
            weights = policy_arrays(model.policy)
            digest = policy_hash(weights)
            if inference_host is not None:inference_host.install(version,weights,digest,args.worker_timeout)
            for connection in connections:
                connection.send(dict(op='collect', version=version, weights=None if args.shared_inference else weights, policy_hash=digest))
            packets = receive_all(connections, 'rollout', args.worker_timeout, abort)
            if slots:packets=[slot.packet(packet,version,digest) for slot,packet in zip(slots,packets)]
            collection_seconds=time.perf_counter()-collection_started
            merge_rollouts(model.rollout_buffer, packets, worker_steps, version, digest)
            model.num_timesteps += cohort
            masks = model.rollout_buffer.actor_masks
            model._settled_stats['collected_transitions'] += cohort
            model._settled_stats['actor_transitions'] += int(masks.sum())
            model._settled_stats['passive_transitions'] += int(masks.size-masks.sum())
            before_updates = model._n_updates
            before_samples = model._settled_stats['optimizer_sample_presentations']
            update_started=time.perf_counter()
            if not args.benchmark_only:model.train()
            else:
                model._settled_stats['completed_rollout_transitions']+=cohort
                model.parallel_used_indices=[]
            update_seconds=time.perf_counter()-update_started
            training_stats={key:float(value) if math.isfinite(float(value)) else None for key,value in model.logger.name_to_value.items()
                            if key.startswith('train/')}
            if not args.benchmark_only:model.logger.dump(step=model.num_timesteps)
            report.update(rounds=version+1, optimized_transitions=model._settled_stats['optimized_transitions'],
                          processed_transitions=model._settled_stats['completed_rollout_transitions'],
                          active_wall_seconds=time.perf_counter()-started)
            report['model_updated']=not args.benchmark_only
            runtime.update(dict(
                rounds=version+1, optimized_transitions=report['optimized_transitions'],
                elapsed_seconds=report['active_wall_seconds'],
                processed_transitions=report['processed_transitions'],
                processed_per_second=report['processed_transitions']/report['active_wall_seconds'],
                optimized_per_second=report['optimized_transitions']/report['active_wall_seconds'],
                collection_seconds=collection_seconds,update_seconds=update_seconds,
                training_stats=training_stats,total_timesteps=model.num_timesteps))
            done = (report['processed_transitions'] >= args.steps
                    or time.perf_counter()-started >= args.max_minutes*60)
            if not args.benchmark_only and (done or time.perf_counter()-last_save >= args.save_every_minutes*60):
                save_atomic_checkpoint(model, args.report/'latest.zip')
                import shutil
                with (args.report/f'checkpoint_{report["processed_transitions"]:09d}.zip').open('xb') as snapshot:
                    with (args.report/'latest.zip').open('rb') as source:
                        shutil.copyfileobj(source,snapshot)
                last_save = time.perf_counter()
                runtime['last_checkpoint_transitions']=report['processed_transitions']
            temporary=args.report/'runtime.partial.json'
            temporary.write_text(json.dumps(runtime),encoding='utf8')
            temporary.replace(args.report/'runtime.json')
            for index, connection in enumerate(connections):
                used = [i-index*worker_steps for i in model.parallel_used_indices
                        if index*worker_steps <= i < (index+1)*worker_steps]
                # All source runs currently use no KL stopping, so every sample
                # is presented once per completed epoch; declare partial coverage.
                connection.send(dict(op='commit', version=version, used_indices=used,
                    sample_presentations=len(used)*(model._n_updates-before_updates),
                    epochs=model._n_updates-before_updates, stop=done))
            save_report()
            label='FROZEN POLICY' if args.benchmark_only else 'SHARED PPO'
            print(f'[{label}] round={version+1} processed={report["processed_transitions"]} '
                  f'optimized_unique={report["optimized_transitions"]}', flush=True)
            print(format_terminal_stats(survival_metrics.snapshot(),report['active_wall_seconds'],
                report['processed_transitions'] if args.benchmark_only else report['optimized_transitions'],
                time.perf_counter(),args.benchmark_only),flush=True)
            del packets
            if done:
                report['stop_reason'] = 'steps' if report['processed_transitions'] >= args.steps else 'wall_time_after_shared_update'
                break
            version += 1
        report['workers'] = receive_all(connections, 'finished', args.worker_timeout, abort)
        report['audits'] = [] if args.benchmark_only else [audit_settled_ledger(args.report/f'worker_{i}'/'learning_transitions.jsonl', minimum=worker_steps)
                           for i in range(count)]
        if any(a['status'] != 'PASS_SETTLED_LEARNING' for a in report['audits']):
            raise RuntimeError('A worker transition audit failed')
        report['status'] = 'COMPLETE_FROZEN_POLICY_DIAGNOSTIC' if args.benchmark_only else 'COMPLETE'
    except BaseException as exc:
        report.update(status='STOPPED', error=repr(exc))
        raise
    finally:
        abort.set()
        if sampler is not None:
            sampler.close()
        for connection in connections:
            try:
                connection.send(dict(op='stop'))
            except (OSError, EOFError):
                pass
        for process in processes:
            process.join(timeout=30.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.)
        for connection in connections:
            connection.close()
        if inference_host is not None:inference_host.close()
        if electron_group is not None:electron_group.close()
        # Drop all borrowed views before unmapping coordinator-owned slots.
        if slots:
            packets=None
            model.rollout_buffer.reset()
            for slot in slots:slot.close()
        if preview is not None:
            preview.shutdown()
            preview.server_close()
        if survival_metrics is not None:
            try:
                survival_metrics.close()
            except OSError as exc:
                report['dashboard_error'] = repr(exc)
        if report['optimized_transitions']:
            save_atomic_checkpoint(model, args.report/'latest.zip')
        report['source_unchanged'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == source_hash
        if args.benchmark_only:
            report['policy_unchanged']=policy_hash(policy_arrays(model.policy))==initial_policy_digest
        if started is not None:
            report['active_wall_seconds'] = time.perf_counter()-started
            report['optimized_transitions_per_second'] = report['optimized_transitions']/report['active_wall_seconds']
            report['processed_transitions_per_second']=report['processed_transitions']/report['active_wall_seconds']
            runtime.update(finished=True,status=report['status'],elapsed_seconds=report['active_wall_seconds'],
                           optimized_per_second=report['optimized_transitions_per_second'])
            (args.report/'runtime.json').write_text(json.dumps(runtime),encoding='utf8')
        save_report()
        if survival_metrics is not None and started is not None:
            print(format_terminal_stats(survival_metrics.snapshot(),report['active_wall_seconds'],
                report['processed_transitions'] if args.benchmark_only else report['optimized_transitions'],
                time.perf_counter(),args.benchmark_only),flush=True)
    print(json.dumps(report, indent=2))
    from tools.plot_ppo_survival import finish_plot
    finish_plot(args.report,args.end_plot,args.preview_port,args.benchmark_only)


if __name__ == '__main__':
    main()
