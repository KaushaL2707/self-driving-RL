from env import CarEnv
from model import ActorCritic
from ppo import PPOTrainer
import torch
import os
import time

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

env = CarEnv()
model = ActorCritic(obs_dim=7, act_dim=2)
trainer = PPOTrainer(model)

best_avg_reward = -float("inf")
start_time = time.time()

for iteration in range(500):
    # Collect experience
    states, actions, rewards, log_probs, values, dones = trainer.collect_rollout(env)

    # Compute per-episode stats
    episode_rewards = []
    ep_reward = 0
    for r, d in zip(rewards, dones):
        ep_reward += r
        if d:
            episode_rewards.append(ep_reward)
            ep_reward = 0

    avg_ep_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
    max_ep_reward = max(episode_rewards) if episode_rewards else 0

    # Compute advantages & update
    advantages, returns = trainer.compute_advantages(rewards, values, dones)
    a_loss, c_loss = trainer.update(states, actions, log_probs, advantages, returns)

    elapsed = time.time() - start_time
    print(
        f"Iter {iteration:4d} | "
        f"Episodes: {len(episode_rewards):3d} | "
        f"Avg Reward: {avg_ep_reward:8.2f} | "
        f"Max Reward: {max_ep_reward:8.2f} | "
        f"A_Loss: {a_loss:.3f} | "
        f"C_Loss: {c_loss:.3f} | "
        f"Time: {elapsed:.0f}s"
    )

    # Save best model
    if avg_ep_reward > best_avg_reward:
        best_avg_reward = avg_ep_reward
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
        print(f"  -> New best model saved! (avg reward: {best_avg_reward:.2f})")

    # Save checkpoint every 50 iters
    if iteration % 50 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"iter_{iteration}.pt"))

# Save final model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
print(f"\nTraining done! Best avg reward: {best_avg_reward:.2f}")
print(f"Run 'python play.py' to watch the trained agent.")
