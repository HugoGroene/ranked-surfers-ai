"""Compact finer-grid encoder, with the same 128+64 feature interface."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SpatialExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,features_dim=192):
        if features_dim!=192:raise ValueError('Expected 192 features')
        super().__init__(observation_space,features_dim)
        channels=observation_space['image'].shape[0]
        self.rgb=nn.Sequential(
            nn.Conv2d(channels,16,5,stride=2,padding=2),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,48,3,stride=2,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)),nn.Flatten(),
            nn.Linear(48*8*8,128),nn.ReLU())
        self.aux=nn.Sequential(nn.Linear(observation_space['aux'].shape[0],96),nn.ReLU(),
                               nn.Linear(96,64),nn.ReLU())

    def forward(self,obs):
        return torch.cat((self.rgb(obs['image']),self.aux(obs['aux'])),dim=1)
