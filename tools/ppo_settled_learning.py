"""Episode-aware PPO collection with passive transitions and game-time GAE.

Local Practice only. No architecture/action-mapping change. Passive waits train
the critic, never a fabricated policy action. Historical PPO remains available.
"""
import json
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, explained_variance


def reference_seconds_for(model, requested):
    return requested if requested is not None else getattr(model,'settled_reference_seconds',.05)


class VerifiedProgressRewards:
    """Same coefficients/milestones/death, on verified active time only.

    No per-decision survival cap: the caller must supply a monotone clock which
    advances only on confident gameplay captures and excludes UI pause time.
    Terminal confirmation does not advance this clock.
    """
    def __init__(self, per_second=.8, milestone_bonus=.15,
                 milestone_seconds=.5, death_penalty=1.):
        self.per_second, self.milestone_bonus = per_second, milestone_bonus
        self.milestone_seconds, self.death_penalty = milestone_seconds, death_penalty
        self.reset()

    def reset(self):
        self.last_duration = 0.
        self.last_milestone = 0

    def step(self, duration, died):
        duration = float(duration)
        if not np.isfinite(duration) or duration < self.last_duration-1e-8:
            raise ValueError('Invalid verified active reward clock')
        duration = max(duration, self.last_duration)
        increment = duration-self.last_duration
        milestone = int((duration+1e-9)//self.milestone_seconds)
        pieces = dict(survival=self.per_second*increment,
            milestones=0. if died else self.milestone_bonus*max(0,milestone-self.last_milestone),
            death=-self.death_penalty if died else 0.)
        self.last_duration, self.last_milestone = duration, milestone
        return sum(pieces.values()), pieces


class TimedDictBuffer(DictRolloutBuffer):
    def reset(self):
        super().reset()
        self.active_seconds = np.zeros((self.buffer_size,self.n_envs),np.float64)
        self.actor_masks = np.ones((self.buffer_size,self.n_envs),np.float32)

    def compute_returns_and_advantage(self, last_values, dones):
        values = last_values.detach().cpu().numpy().flatten()
        carry = np.zeros(self.n_envs)
        for i in reversed(range(self.buffer_size)):
            alive = 1.-(dones if i==self.buffer_size-1 else self.episode_starts[i+1])
            next_value = values if i==self.buffer_size-1 else self.values[i+1]
            exponent = self.active_seconds[i]/self.reference_seconds
            discount = self.gamma**exponent
            trace = self.gae_lambda**exponent
            delta = self.rewards[i]+discount*next_value*alive-self.values[i]
            carry = delta+discount*trace*alive*carry
            self.advantages[i] = carry
        self.returns = self.advantages+self.values

    def _get_samples(self, batch_inds, env=None):
        data = super()._get_samples(batch_inds,env)
        return SimpleNamespace(**data._asdict(),
            sample_indices=batch_inds,
            actor_mask=self.to_torch(self.actor_masks.reshape(-1)[batch_inds]))


class SettledPPO(PPO):
    """Single DummyVecEnv + Monitor; explicit corrected collection protocol."""
    def configure_settled(self, raw, output, reference_seconds=.05, max_minutes=None):
        if self.n_envs!=1 or self.use_sde:
            raise ValueError('Settled collector requires one discrete local environment')
        if not np.isfinite(reference_seconds) or reference_seconds<=0:
            raise ValueError('Invalid time-discount reference')
        self.rollout_buffer = TimedDictBuffer(self.n_steps,self.observation_space,
            self.action_space,device=self.device,gamma=self.gamma,
            gae_lambda=self.gae_lambda,n_envs=1)
        self.rollout_buffer.reference_seconds = reference_seconds
        self.settled_reference_seconds = reference_seconds
        self._settled_raw = raw
        self._settled_output = output
        self._settled_deadline = time.perf_counter()+max_minutes*60 if max_minutes else float('inf')
        self._settled_stats = dict(collected_transitions=0,optimized_transitions=0,
            actor_transitions=0,passive_transitions=0,boundary_settlements=0,
            completed_rollout_transitions=0,optimizer_sample_presentations=0,
            discarded_partial_transitions=0,stop_reason=None)
        self._settled_ledger = output/'learning_transitions.jsonl'
        if self._settled_ledger.exists():
            raise ValueError('Transition ledger exists')
        self._settled_trained_this_run = 0
        self._settled_checkpoint_due = False
        self._settled_last_endpoint_capture = None
        self._settled_initial_timesteps = int(self.num_timesteps)
        reward=raw.survival_rewards
        root=Path(__file__).resolve().parents[1]
        protocol=dict(version=1,gamma=self.gamma,gae_lambda=self.gae_lambda,
            reference_seconds=reference_seconds,
            reward_coefficients={name:getattr(reward,name) for name in
                ('per_second','milestone_bonus','milestone_seconds','death_penalty')},
            source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('train_rs_v612_ppo.py','tools/ppo_settled_learning.py')})
        (output/'learning_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

    def _record(self, row):
        with self._settled_ledger.open('a',encoding='utf8') as handle:
            handle.write(json.dumps(row)+'\n')

    def _passive_step(self, env, kind):
        self._settled_raw._learning_passive_kind = kind
        return env.step(np.array([0]))

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if self._settled_stats['stop_reason']=='wall_time_at_optimized_rollout_boundary':
            return False
        raw, buffer = self._settled_raw, rollout_buffer
        self.policy.set_training_mode(False)
        buffer.reset()
        before_resume = self._last_obs
        prior_capture = (getattr(self,'_settled_last_endpoint_capture',None)
            or getattr(raw,'_learning_observation_capture',None))
        had_boundary = callback.rollout_gap_tracker.last_rollout_end_at is not None
        callback.on_rollout_start()
        passive = had_boundary
        # Restore the preceding endpoint; resync's new endpoint is returned by
        # an explicit passive env transition, not silently substituted.
        if passive:
            self._last_obs = before_resume
        for i in range(n_rollout_steps):
            old_obs = self._last_obs
            reward_before = raw.survival_rewards.last_duration
            episode_before = raw.episode_number
            with th.no_grad():
                tensor = obs_as_tensor(old_obs,self.device)
                if passive:
                    values = self.policy.predict_values(tensor)
                    actions = np.array([0])
                    log_probs = th.zeros(1,device=self.device)
                else:
                    sampled,values,log_probs = self.policy(tensor)
                    actions = sampled.cpu().numpy()
            new_obs,rewards,dones,infos = (
                self._passive_step(env,'resync') if passive else env.step(actions))
            info = infos[0]
            dispatched = info.get('learning_dispatched_key')
            dispatch_at = info.get('learning_dispatch_at')
            parts = [dict(duration=info['reward_duration_s'],
                died=bool(dones[0] and not info.get('TimeLimit.truncated',False)),
                reward=float(rewards[0]), pieces={k:info['reward_'+k] for k in ('survival','milestones','death')})]
            duration = info['reward_duration_s']-reward_before
            actor = not passive and not info.get('learning_action_suppressed',False)
            self.num_timesteps += 1
            boundary = i==n_rollout_steps-1
            if boundary:
                # Pause/menu evidence now precedes final insertion and GAE.
                callback.update_locals(dict(actions=actions,new_obs=new_obs,rewards=rewards,
                    dones=dones,infos=infos))
                callback.on_rollout_end()
                if not dones[0]:
                    tail_obs,tail_reward,tail_done,tail_infos = self._passive_step(env,'boundary')
                    tail = tail_infos[0]
                    duration += max(0.,tail['reward_duration_s']-info['reward_duration_s'])
                    parts.append(dict(duration=tail['reward_duration_s'],
                        died=bool(tail_done[0] and not tail.get('TimeLimit.truncated',False)),
                        reward=float(tail_reward[0]),
                        pieces={k:tail['reward_'+k] for k in ('survival','milestones','death')}))
                    rewards = rewards+tail_reward
                    new_obs,dones,infos = tail_obs,tail_done,tail_infos
                    for key in ('reward_survival','reward_milestones','reward_death','reward_v61'):
                        infos[0][key] = float(info[key])+float(tail[key])
                    self._settled_stats['boundary_settlements'] += 1
                    if dones[0]:
                        # DummyVecEnv reset a NEW episode. Freeze it too; the
                        # already-settled old terminal must not become a later
                        # fake action. Failure here is a declared safe stop.
                        raw.pause_for_ppo_optimizer()
                        new_obs={key:value[np.newaxis].copy() for key,value in raw._learning_boundary_observation.items()}
                        print('[SETTLED] Terminal settled; next episode frozen before optimization.',flush=True)
                elif (getattr(raw,'_ppo_pause_pending',None) is not None
                      and hasattr(raw,'_learning_boundary_observation')):
                    new_obs={key:value[np.newaxis].copy() for key,value in raw._learning_boundary_observation.items()}
            callback.update_locals(dict(actions=actions,new_obs=new_obs,rewards=rewards,
                dones=dones,infos=infos))
            # The existing callback logs statistics but corrected wall-time
            # stops happen only AFTER a full rollout has been optimized.
            if not callback.on_step():
                self._settled_stats['discarded_partial_transitions'] += i
                self._settled_stats['stop_reason'] = 'callback_stop_before_buffer_add'
                return False
            self._update_info_buffer(infos,dones)
            raw_reward = float(rewards[0])
            dt = max(0.,float(duration))
            if dones[0] and infos[0].get('TimeLimit.truncated',False):
                with th.no_grad():
                    terminal = self.policy.obs_to_tensor(infos[0]['terminal_observation'])[0]
                    terminal_value = float(self.policy.predict_values(terminal).flatten()[0])
                rewards[0] += self.gamma**(dt/self.settled_reference_seconds)*terminal_value
            buffer.active_seconds[i] = dt
            buffer.actor_masks[i] = float(actor)
            buffer.add(old_obs,actions.reshape(-1,1),rewards,self._last_episode_starts,values,log_probs)
            self._record(dict(event='collected',transition=int(self.num_timesteps),
                rollout=self._settled_trained_this_run//self.n_steps+1,index=i,
                episode=episode_before,kind='passive_resync' if passive else 'policy',
                sampled_action=None if passive else int(actions[0]),
                actor_mask=actor,command=dispatched,dispatch_at=dispatch_at,source_capture=prior_capture,
                reaction_latency=info.get('reaction_latency'),
                next_capture=getattr(raw,'_learning_observation_capture',None),
                active_seconds=dt,discount=self.gamma**(dt/self.settled_reference_seconds),
                environment_reward=raw_reward,buffer_reward=float(rewards[0]),
                done=bool(dones[0]),truncated=bool(infos[0].get('TimeLimit.truncated',False)),
                reward_substeps=parts))
            self._settled_stats['collected_transitions'] += 1
            self._settled_stats['actor_transitions' if actor else 'passive_transitions'] += 1
            self._last_obs,self._last_episode_starts = new_obs,dones
            prior_capture = getattr(raw,'_learning_observation_capture',None)
            self._settled_last_endpoint_capture = prior_capture
            passive = False
        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs,self.device))
        buffer.compute_returns_and_advantage(last_values,dones)
        return True

    def train(self):
        if not hasattr(self,'_settled_stats'):
            return super().train()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        metrics = []
        stop = False
        used_indices = set()
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values,log_prob,entropy = self.policy.evaluate_actions(data.observations,data.actions.long().flatten())
                mask = data.actor_mask.bool()
                if mask.any():
                    advantage = data.advantages[mask]
                    if self.normalize_advantage and len(advantage)>1:
                        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                    ratio = th.exp(log_prob[mask]-data.old_log_prob[mask])
                    pg = -th.minimum(advantage*ratio,advantage*th.clamp(ratio,1-clip,1+clip)).mean()
                    ent = -entropy[mask].mean() if entropy is not None else log_prob[mask].mean()
                    with th.no_grad():
                        lr = log_prob[mask]-data.old_log_prob[mask]
                        kl = float(((th.exp(lr)-1)-lr).mean())
                        clip_fraction=float(((ratio-1).abs()>clip).float().mean())
                else:
                    # Do not create even zero actor gradients: Adam momentum
                    # must not update actor-only weights on a passive batch.
                    pg = values.sum()*0
                    ent = pg
                    kl = 0.
                    clip_fraction=0.
                predicted = values.flatten()
                if self.clip_range_vf is not None:
                    vf_clip = self.clip_range_vf(self._current_progress_remaining)
                    predicted = data.old_values+th.clamp(predicted-data.old_values,-vf_clip,vf_clip)
                vf = F.mse_loss(predicted,data.returns)
                loss = pg+self.ent_coef*ent+self.vf_coef*vf
                metrics.append((float(pg.detach()),float(vf.detach()),float(ent.detach()),kl,clip_fraction))
                if self.target_kl is not None and kl>1.5*self.target_kl:
                    stop = True
                    print(f'[SETTLED KL] Early stop at epoch {epoch}; approximate KL={kl:.5f}.',flush=True)
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.policy.optimizer.step()
                used_indices.update(int(index) for index in data.sample_indices)
                self._settled_stats['optimizer_sample_presentations'] += len(data.sample_indices)
            self._n_updates += 1
            if stop:
                break
        for key,value in zip(('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction'),np.mean(metrics,axis=0)):
            self.logger.record('train/'+key,value)
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip)
        self.logger.record('train/explained_variance',explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        self._settled_trained_this_run += self.n_steps
        self._settled_stats['completed_rollout_transitions'] = self._settled_trained_this_run
        self._settled_stats['optimized_transitions'] += len(used_indices)
        self._record(dict(event='optimized',through_transition=int(self.num_timesteps),
            optimized_transitions=self._settled_stats['optimized_transitions'],
            used_indices=sorted(used_indices),epochs=epoch+1))
        if self._settled_checkpoint_due:
            from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
            self._settled_checkpoint_due=False
            save_atomic_checkpoint(self,self._settled_output/'latest.zip')
            print(f'[SETTLED SAVE] Atomic checkpoint after optimization through {self.num_timesteps} transitions.',flush=True)
        if time.perf_counter()>=self._settled_deadline:
            self._settled_stats['stop_reason'] = 'wall_time_at_optimized_rollout_boundary'
            self._total_timesteps = self.num_timesteps

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_settled_raw','_settled_output','_settled_ledger','_settled_deadline']


def _ledger_rows(path):
    with path.open(encoding='utf8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_settled_ledger(path, minimum=1, gamma=.995, reference=.05):
    """Replay every raw substep and verify coverage, masks and discounts."""
    metadata=path.with_name('learning_protocol.json')
    if metadata.exists():
        protocol=json.loads(metadata.read_text())
        gamma,reference=protocol['gamma'],protocol['reference_seconds']
        calc=VerifiedProgressRewards(**protocol['reward_coefficients'])
    else:
        reference=.05 if reference is None else reference
        calc=VerifiedProgressRewards()
    episode=None
    pending=[]
    count=processed=passive=terminals=0
    optimized_unique=0
    failures=[]
    prior_next=None
    for row in _ledger_rows(path):
        if row['event']=='optimized':
            if not pending or [r['index'] for r in pending]!=list(range(len(pending))):
                failures.append('Incomplete or duplicate optimized rollout')
            if pending and row['through_transition']!=pending[-1]['transition']:
                failures.append('Optimizer endpoint does not match collected rollout')
            if len(set(row['used_indices']))!=len(row['used_indices']) or any(
                    i<0 or i>=len(pending) for i in row['used_indices']):
                failures.append('Invalid optimizer sample coverage')
            processed+=len(pending)
            pending=[]
            optimized_unique+=len(row['used_indices'])
            continue
        pending.append(dict(index=row['index'],transition=row['transition']))
        count+=1
        passive+=row['kind']=='passive_resync'
        terminals+=row['done'] and not row['truncated']
        if prior_next is not None and row['source_capture']!=prior_next:
            failures.append('Discontinuous observation capture lineage')
        prior_next=row['next_capture']
        if row.get('dispatch_at') is not None and row['source_capture'] is not None:
            if row['dispatch_at']<row['source_capture']['captured_at']:
                failures.append('Dispatch precedes policy source capture')
        if row['episode']!=episode:
            calc.reset()
            episode=row['episode']
        before=calc.last_duration
        expected=0.
        for part in row['reward_substeps']:
            reward,pieces=calc.step(part['duration'],part['died'])
            expected+=reward
            if abs(part['reward']-reward)>1e-5:
                failures.append('Raw substep reward mismatch')
            if any(abs(part['pieces'][k]-v)>.00011 for k,v in pieces.items()):
                failures.append('Reward component mismatch')
        if abs(expected-row['environment_reward'])>1e-5:
            failures.append('Combined environment reward mismatch')
        if abs(calc.last_duration-before-row['active_seconds'])>1e-7:
            failures.append('Active duration mismatch')
        if abs(gamma**(row['active_seconds']/reference)-row['discount'])>1e-9:
            failures.append('Duration discount mismatch')
        if row['kind']=='passive_resync' and (row['sampled_action'] is not None or row['actor_mask'] or row['command'] is not None):
            failures.append('Passive transition has actor/input attribution')
        if row['actor_mask'] and row['sampled_action'] is None:
            failures.append('Actor transition has no sampled action')
        if row['done'] and not row['truncated'] and abs(row['buffer_reward']-expected)>1e-5:
            failures.append('Death transition was bootstrapped')
    if count<minimum:failures.append('Too few transitions')
    if pending:failures.append('Collected rollout was not processed by optimizer')
    return dict(status='PASS_SETTLED_LEARNING' if not failures else 'FAIL_SETTLED_LEARNING',
        transitions=count,processed_rollout_transitions=processed,
        unique_transitions_used_by_optimizer=optimized_unique,
        passive_resync_transitions=passive,terminal_transitions=terminals,
        failures=failures)
