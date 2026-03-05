# Self-Driving Car with PPO (Reinforcement Learning)

A self-driving car agent trained using **Proximal Policy Optimization (PPO)** from scratch in Python. The car learns to navigate an oval track using only distance sensors — no hardcoded rules, pure learning.

Built with PyTorch, Gymnasium, and Pygame.

---

## Project Structure

```
self_driving_rl/
├── env.py              # Custom Gymnasium environment (car, track, physics, sensors)
├── model.py            # Actor-Critic neural network (policy + value heads)
├── ppo.py              # PPO training algorithm (rollout, GAE, clipped update)
├── train.py            # Headless training loop (fast, terminal output only)
├── train_visual.py     # Visual training — watch the agent learn in real-time
├── visualize.py        # Manual driving — drive the car yourself with arrow keys
├── play.py             # Watch a trained agent drive (load checkpoint)
├── checkpoints/        # Saved model weights (created during training)
│   ├── best.pt         # Best model (highest avg episode reward)
│   ├── final.pt        # Model at end of training
│   └── iter_*.pt       # Periodic checkpoints every 50 iterations
├── env/                # Python virtual environment
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate

# Install dependencies
pip install numpy gymnasium pygame torch
```

> For GPU training, install PyTorch with CUDA from https://pytorch.org/get-started/locally/

---

## Commands

| Command                                  | What it does                                                |
| ---------------------------------------- | ----------------------------------------------------------- |
| `python visualize.py`                    | Drive the car manually with arrow keys                      |
| `python train.py`                        | Train the agent (headless, fast, terminal output)           |
| `python train_visual.py`                 | Train with live visualization (watch it learn)              |
| `python play.py`                         | Watch the trained agent drive (loads `checkpoints/best.pt`) |
| `python play.py checkpoints/iter_200.pt` | Watch a specific checkpoint                                 |

### Keyboard Controls

**visualize.py** (manual driving):

- Arrow keys — steer and accelerate/brake
- R — reset
- Q — quit

**train_visual.py** (live training):

- S — cycle speed: uncapped / 30 FPS / 60 FPS
- Q — quit and save

**play.py** (watch agent):

- M — toggle between AI and manual mode
- R — reset episode
- Q — quit

---

## How It Works

### The Environment (`env.py`)

A 2D car drives on an oval track. The environment follows the Gymnasium API.

**State (what the agent sees) — 7 numbers:**

- 5 distance sensors (raycasts at angles -0.6, -0.3, 0, 0.3, 0.6 from heading)
- Normalized speed (0 to 1)
- Angle to the next waypoint (-1 to 1)

**Actions (what the agent controls) — 2 continuous values:**

- Steering: -1 (left) to +1 (right)
- Throttle: -1 (brake) to +1 (accelerate)

**Rewards:**

- `+speed * 0.1` each step (encourages moving)
- `+1.0` for reaching a waypoint (encourages following the track)
- `-10.0` for hitting a wall (punishes crashing)

**Episode ends** when the car crashes or after 1000 steps.

### The Model (`model.py`)

An Actor-Critic neural network with shared feature layers:

```
Observation (7)
      |
  [Linear 7 -> 64 + Tanh]     Shared feature layers
  [Linear 64 -> 64 + Tanh]    (learns to interpret sensors)
      |              |
  [Actor 64 -> 2]  [Critic 64 -> 1]
      |              |
  action mean      state value V(s)
  + learnable std
```

- **Actor (policy):** Outputs mean + std of a Gaussian distribution over actions. Actions are sampled from this distribution, enabling exploration.
- **Critic (value):** Outputs a scalar estimating "how much total reward can I expect from this state?" Used to compute advantages.

Total parameters: ~4,900 (very lightweight).

### PPO Algorithm (`ppo.py`)

PPO trains the agent in 3 steps per iteration:

#### 1. Collect Rollout

The agent plays in the environment for 2048 steps, storing:

- States, actions, rewards, log probabilities, values, done flags

#### 2. Compute Advantages (GAE)

For each timestep, compute: "was this action better or worse than expected?"

```
delta_t = reward_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
advantage_t = delta_t + gamma * lambda * (1 - done) * advantage_{t+1}
```

- `gamma = 0.99` — discount factor (how much to value future rewards)
- `lambda = 0.95` — GAE smoothing (bias-variance tradeoff)
- Positive advantage = action was better than expected
- Negative advantage = action was worse than expected

#### 3. PPO Clipped Update

Run 10 epochs of gradient descent on the collected data:

```
ratio = pi_new(a|s) / pi_old(a|s)          # how much the policy changed
surr1 = ratio * advantage                   # unclipped objective
surr2 = clip(ratio, 0.8, 1.2) * advantage   # clipped objective
actor_loss = -min(surr1, surr2)              # pessimistic bound
```

**Why clipping?** Without it, a single update could change the policy drastically, causing training instability. The clip (epsilon=0.2) ensures the policy can only change by ~20% per update — the core idea of PPO.

**Full loss:**

```
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
```

- **Critic loss:** MSE between predicted values and actual returns
- **Entropy bonus:** Encourages exploration (prevents premature convergence)
- **Gradient clipping** (max_norm=0.5): Prevents exploding gradients

### Training (`train.py` / `train_visual.py`)

Runs 500 iterations of the PPO loop. Each iteration:

1. Collect 2048 steps of experience
2. Compute advantages with GAE
3. Update the network for 10 epochs
4. Log stats and save checkpoints

`train_visual.py` does the same but renders every step with Pygame, plus a reward graph showing learning progress.

---

## Hyperparameters

| Parameter     | Value | Purpose                        |
| ------------- | ----- | ------------------------------ |
| Learning rate | 3e-4  | Adam optimizer step size       |
| Clip epsilon  | 0.2   | PPO clipping range             |
| Gamma         | 0.99  | Discount factor                |
| GAE Lambda    | 0.95  | Advantage estimation smoothing |
| Rollout steps | 2048  | Steps per iteration            |
| Update epochs | 10    | SGD passes per update          |
| Hidden size   | 64    | Neurons per hidden layer       |
| Entropy coeff | 0.01  | Exploration bonus weight       |
| Critic coeff  | 0.5   | Value loss weight              |
| Max grad norm | 0.5   | Gradient clipping threshold    |

---

## Theory: PPO in a Nutshell

**Reinforcement Learning:** An agent interacts with an environment, receives rewards, and learns a policy (state -> action mapping) that maximizes cumulative reward.

**Policy Gradient:** Directly optimize the policy by computing gradients of expected reward with respect to policy parameters. Problem: high variance, unstable updates.

**PPO (Proximal Policy Optimization):** A policy gradient method that constrains how much the policy can change per update using a clipped surrogate objective. This gives stable, monotonic improvement — the key breakthrough that made RL practical for continuous control.

**Actor-Critic:** Combines two ideas:

- **Actor** learns the policy (what to do)
- **Critic** learns the value function (how good is this state)
- The critic's predictions reduce variance in the actor's gradient estimates

**GAE (Generalized Advantage Estimation):** A method to estimate advantages (how much better was this action than average) that smoothly trades off between bias and variance using the lambda parameter.
