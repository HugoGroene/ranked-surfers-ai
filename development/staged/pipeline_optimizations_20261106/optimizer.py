"""Staged PPO loop: identical losses, sample order, clipping and Adam steps.

Only reporting synchronization is deferred; CUDA masks still have their original
semantics. CPU actor-count branch preserves the all-passive Adam behavior.
KL early-stop models use the original method, never a deferred KL decision.
Derived from tools/ppo_settled_learning.py; paired correctness tests are provided.
"""
import time
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance

def deferred_train(self):
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip = self.clip_range(self._current_progress_remaining)
    metrics = []
    used_indices = set()
    for epoch in range(self.n_epochs):
        for data in self.rollout_buffer.get(self.batch_size):
            values,log_prob,entropy = self.policy.evaluate_actions(data.observations,data.actions.long().flatten())
            mask = data.actor_mask.bool()
            actor_count = int(np.count_nonzero(self.rollout_buffer.actor_masks.reshape(-1)[data.sample_indices]))
            if actor_count:
                advantage = data.advantages[mask]
                if self.normalize_advantage and len(advantage)>1:
                    advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                ratio = th.exp(log_prob[mask]-data.old_log_prob[mask])
                pg = -th.minimum(advantage*ratio,advantage*th.clamp(ratio,1-clip,1+clip)).mean()
                ent = -entropy[mask].mean() if entropy is not None else log_prob[mask].mean()
                with th.no_grad():
                    lr = log_prob[mask]-data.old_log_prob[mask]
                    kl = ((th.exp(lr)-1)-lr).mean()
                    clip_fraction=((ratio-1).abs()>clip).float().mean()
            else:
                # Do not create even zero actor gradients: Adam momentum
                # must not update actor-only weights on a passive batch.
                pg = values.sum()*0
                ent = pg
                kl = values.new_zeros(())
                clip_fraction = values.new_zeros(())
            predicted = values.flatten()
            if self.clip_range_vf is not None:
                vf_clip = self.clip_range_vf(self._current_progress_remaining)
                predicted = data.old_values+th.clamp(predicted-data.old_values,-vf_clip,vf_clip)
            vf = F.mse_loss(predicted,data.returns)
            loss = pg+self.ent_coef*ent+self.vf_coef*vf
            metrics.append(th.stack((pg.detach(),vf.detach(),ent.detach(),kl.detach(),clip_fraction.detach())).to(th.float64))
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
            self.policy.optimizer.step()
            used_indices.update(int(index) for index in data.sample_indices)
            self._settled_stats['optimizer_sample_presentations'] += len(data.sample_indices)
        self._n_updates += 1
    for key,value in zip(('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction'),th.stack(metrics).mean(dim=0).cpu().numpy()):
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
