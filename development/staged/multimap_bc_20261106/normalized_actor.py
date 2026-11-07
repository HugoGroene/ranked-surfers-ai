"""Serializable actor-only feature standardization; critic sees original features."""
import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class FixedFeatureStandardization(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))

    def forward(self, x):
        return (x-self.mean)/self.scale

    def fit(self, training_features):
        if len(training_features)<2 or not torch.isfinite(training_features).all():
            raise ValueError('Finite training features required')
        with torch.no_grad():
            self.mean.copy_(training_features.mean(0))
            self.scale.copy_(training_features.std(0).clamp_min(.1))


class NormalizedActorPolicy(MultiInputActorCriticPolicy):
    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net=nn.Sequential(
            FixedFeatureStandardization(self.features_dim),
            *list(self.mlp_extractor.policy_net.children())).to(self.device)


def replace_actor(model, training_features):
    """New policy head, exact frozen extractor/critic state, new optimizer state."""
    old=model.policy
    if not old.share_features_extractor: raise ValueError('Shared encoder required')
    policy=NormalizedActorPolicy(model.observation_space,model.action_space,
                                 model.lr_schedule,use_sde=model.use_sde,**model.policy_kwargs).to(model.device)
    keep={k:v for k,v in old.state_dict().items()
          if not k.startswith(('mlp_extractor.policy_net.','action_net.'))}
    incompatible=policy.load_state_dict(keep,strict=False)
    if incompatible.unexpected_keys or any(not k.startswith(('mlp_extractor.policy_net.','action_net.'))
                                           for k in incompatible.missing_keys):
        raise ValueError('Unexpected policy state mismatch')
    for k,v in keep.items():
        if not torch.equal(v,policy.state_dict()[k]): raise ValueError('Frozen weights changed: '+k)
    policy.mlp_extractor.policy_net[0].fit(training_features)
    model.policy_class=NormalizedActorPolicy
    model.policy=policy
    model._last_obs=None
    return keep
