import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=7, act_dim=2):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # Actor head: outputs mean of action distribution
        self.actor_mean = nn.Linear(64, act_dim)
        # Learnable log std (not input-dependent)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head: outputs scalar value V(s)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared(state)
        mean = torch.tanh(self.actor_mean(features))  # bound to [-1,1]
        std = self.actor_log_std.exp()
        value = self.critic(features)
        return mean, std, value

    def get_action(self, state):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)  # sum over action dims
        return action.clamp(-1, 1), log_prob, value.squeeze(-1)

    def evaluate(self, states, actions):
        """Re-evaluate actions under current policy (used in PPO update)."""
        mean, std, value = self.forward(states)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value.squeeze(-1)
