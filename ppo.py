import torch
import numpy as np

class PPOTrainer:
    def __init__(self, model, lr=3e-4, clip_eps=0.2, gamma=0.99, lam=0.95):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps  # the "P" in PPO — clipping range
        self.gamma = gamma        # discount factor (future reward weight)
        self.lam = lam            # GAE lambda

    def collect_rollout(self, env, n_steps=2048):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        state, _ = env.reset()

        for _ in range(n_steps):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(state_t)

            next_state, reward, done, truncated, _ = env.step(action.numpy())

            states.append(state)
            actions.append(action.numpy())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done or truncated)

            state = next_state if not (done or truncated) else env.reset()[0]

        return states, actions, rewards, log_probs, values, dones

    def compute_advantages(self, rewards, values, dones):
        """GAE - Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t+1] if t+1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_val * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1-dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def update(self, states, actions, log_probs_old, advantages, returns, epochs=10):
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        old_lp_t = torch.FloatTensor(log_probs_old)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)

        # Normalize advantages (stabilizes training — like BatchNorm)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(epochs):
            mean, std, values = self.model(states_t)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(-1)

            # THE PPO CLIPPED OBJECTIVE
            ratio = (new_log_probs - old_lp_t).exp()   # π_new / π_old
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (like MSE regression)
            critic_loss = (ret_t - values.squeeze()).pow(2).mean()

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().sum(-1).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item()