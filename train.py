from env import CarEnv
from model import ActorCritic
from ppo import PPOTrainer
import torch

env = CarEnv()
model = ActorCritic(obs_dim=7, act_dim=2)
trainer = PPOTrainer(model)

for iteration in range(500):
    # Collect experience
    states, actions, rewards, log_probs, values, dones = trainer.collect_rollout(env)

    # Compute advantages
    advantages, returns = trainer.compute_advantages(rewards, values, dones)

    # Update policy
    a_loss, c_loss = trainer.update(states, actions, log_probs, advantages, returns)

    avg_reward = sum(rewards) / len(rewards)
    print(f"Iter {iteration:4d} | Avg Reward: {avg_reward:.3f} | Actor Loss: {a_loss:.3f} | Critic Loss: {c_loss:.3f}")

    # Save checkpoint every 50 iters
    if iteration % 50 == 0:
        torch.save(model.state_dict(), f"checkpoint_{iteration}.pt")