import pygame
import numpy as np
import torch
import os
from env import CarEnv
from model import ActorCritic
from ppo import PPOTrainer

# --- Display ---
WIDTH, HEIGHT = 800, 600
FPS = 0  # 0 = uncapped (train fast), press S to slow down
BG_COLOR = (30, 30, 30)
TRACK_COLOR = (80, 80, 80)
TRACK_EDGE_COLOR = (200, 200, 200)
CAR_COLOR = (255, 80, 80)
WAYPOINT_COLOR = (50, 50, 100)
NEXT_WP_COLOR = (0, 255, 100)
SENSOR_COLOR = (255, 255, 0)
INFO_COLOR = (220, 220, 220)
GRAPH_COLOR = (0, 200, 255)

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


def draw_track(screen, env):
    track = env.track
    n = len(track)
    track_width = 80
    inner_pts, outer_pts = [], []
    for i in range(n):
        px, py = track[i]
        nx, ny = track[(i + 1) % n]
        dx, dy = nx - px, ny - py
        length = np.sqrt(dx**2 + dy**2)
        norm_x, norm_y = -dy / length, dx / length
        inner_pts.append((px - norm_x * track_width, py - norm_y * track_width))
        outer_pts.append((px + norm_x * track_width, py + norm_y * track_width))
    for i in range(n):
        j = (i + 1) % n
        quad = [inner_pts[i], inner_pts[j], outer_pts[j], outer_pts[i]]
        pygame.draw.polygon(screen, TRACK_COLOR, quad)
    pygame.draw.lines(screen, TRACK_EDGE_COLOR, True, inner_pts, 2)
    pygame.draw.lines(screen, TRACK_EDGE_COLOR, True, outer_pts, 2)
    for i, (wx, wy) in enumerate(track):
        color = NEXT_WP_COLOR if i == env.waypoint_idx % n else WAYPOINT_COLOR
        pygame.draw.circle(screen, color, (int(wx), int(wy)), 4)


def draw_car(screen, env):
    x, y, angle = env.x, env.y, env.angle
    l, hw = 20, 8
    tip = (int(x + np.cos(angle) * l), int(y + np.sin(angle) * l))
    left = (int(x + np.cos(angle + 2.5) * hw * 2), int(y + np.sin(angle + 2.5) * hw * 2))
    right = (int(x + np.cos(angle - 2.5) * hw * 2), int(y + np.sin(angle - 2.5) * hw * 2))
    pygame.draw.polygon(screen, CAR_COLOR, [tip, left, right])
    for ra in [-0.6, -0.3, 0, 0.3, 0.6]:
        ray_a = angle + ra
        dist = env._cast_single_ray(ray_a)
        ex = int(x + np.cos(ray_a) * dist)
        ey = int(y + np.sin(ray_a) * dist)
        pygame.draw.line(screen, SENSOR_COLOR, (int(x), int(y)), (ex, ey), 1)


def draw_reward_graph(screen, font, reward_history, x_start, y_start, w, h):
    pygame.draw.rect(screen, (50, 50, 50), (x_start, y_start, w, h))
    pygame.draw.rect(screen, (100, 100, 100), (x_start, y_start, w, h), 1)
    label = font.render("Avg Episode Reward", True, INFO_COLOR)
    screen.blit(label, (x_start + 4, y_start - 18))

    if len(reward_history) < 2:
        return

    min_r = min(reward_history)
    max_r = max(reward_history)
    r_range = max_r - min_r if max_r != min_r else 1

    points = []
    n = len(reward_history)
    for i, r in enumerate(reward_history):
        px = int(x_start + (i / max(n - 1, 1)) * w)
        py = int(y_start + h - ((r - min_r) / r_range) * h)
        points.append((px, py))

    pygame.draw.lines(screen, GRAPH_COLOR, False, points, 2)

    # Min/max labels
    min_label = font.render(f"{min_r:.0f}", True, INFO_COLOR)
    max_label = font.render(f"{max_r:.0f}", True, INFO_COLOR)
    screen.blit(max_label, (x_start + 4, y_start + 2))
    screen.blit(min_label, (x_start + 4, y_start + h - 16))


def draw_info(screen, font, iteration, ep_count, avg_reward, best_reward, speed_mode, step_in_rollout, n_steps):
    lines = [
        f"Iteration: {iteration}/500",
        f"Rollout: {step_in_rollout}/{n_steps}",
        f"Episodes this iter: {ep_count}",
        f"Avg Ep Reward: {avg_reward:.1f}",
        f"Best Avg Reward: {best_reward:.1f}",
        "",
        f"Speed: {'FAST (uncapped)' if speed_mode == 0 else f'{speed_mode} FPS'}",
        "S: toggle speed | Q: quit",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, INFO_COLOR)
        screen.blit(surf, (10, 10 + i * 20))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving RL - Live Training")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    env = CarEnv()
    model = ActorCritic(obs_dim=7, act_dim=2)
    trainer = PPOTrainer(model)

    n_steps = 2048
    reward_history = []
    best_avg = -float("inf")
    speed_modes = [0, 30, 60]  # uncapped, 30fps, 60fps
    speed_idx = 0
    running = True

    for iteration in range(500):
        if not running:
            break

        # --- Collect rollout visually ---
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        state, _ = env.reset()
        ep_rewards_this_iter = []
        ep_reward = 0

        for step in range(n_steps):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        speed_idx = (speed_idx + 1) % len(speed_modes)

            if not running:
                break

            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_t)

            next_state, reward, done, truncated, _ = env.step(action.numpy())

            states.append(state)
            actions.append(action.numpy())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done or truncated)

            ep_reward += reward
            if done or truncated:
                ep_rewards_this_iter.append(ep_reward)
                ep_reward = 0
                state = env.reset()[0]
            else:
                state = next_state

            # Draw every 4th frame to keep it responsive
            if step % 4 == 0:
                avg_r = sum(ep_rewards_this_iter) / max(len(ep_rewards_this_iter), 1) if ep_rewards_this_iter else 0
                screen.fill(BG_COLOR)
                draw_track(screen, env)
                draw_car(screen, env)
                draw_info(screen, font, iteration, len(ep_rewards_this_iter), avg_r, best_avg, speed_modes[speed_idx], step, n_steps)
                if reward_history:
                    draw_reward_graph(screen, font, reward_history[-100:], 540, 30, 240, 120)
                pygame.display.flip()
                clock.tick(speed_modes[speed_idx])

        if not running:
            break

        # --- PPO update (happens off-screen, fast) ---
        avg_ep = sum(ep_rewards_this_iter) / max(len(ep_rewards_this_iter), 1)
        reward_history.append(avg_ep)

        advantages, returns = trainer.compute_advantages(rewards, values, dones)
        a_loss, c_loss = trainer.update(states, actions, log_probs, advantages, returns)

        if avg_ep > best_avg:
            best_avg = avg_ep
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pt"))

        print(f"Iter {iteration:4d} | Eps: {len(ep_rewards_this_iter):3d} | Avg: {avg_ep:8.1f} | Best: {best_avg:8.1f} | A_Loss: {a_loss:.3f} | C_Loss: {c_loss:.3f}")

        if iteration % 50 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"iter_{iteration}.pt"))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
    print(f"\nDone! Best avg reward: {best_avg:.1f}")
    pygame.quit()


if __name__ == "__main__":
    main()
