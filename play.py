import pygame
import numpy as np
import torch
import sys
from env import CarEnv
from model import ActorCritic

WIDTH, HEIGHT = 800, 600
FPS = 60
BG_COLOR = (30, 30, 30)
TRACK_COLOR = (80, 80, 80)
TRACK_EDGE_COLOR = (200, 200, 200)
CAR_COLOR = (255, 80, 80)
WAYPOINT_COLOR = (50, 50, 100)
NEXT_WP_COLOR = (0, 255, 100)
SENSOR_COLOR = (255, 255, 0)
INFO_COLOR = (220, 220, 220)


def draw_track(screen, env):
    track = env.track
    n = len(track)
    track_width = 80

    inner_pts = []
    outer_pts = []
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
    length, half_w = 20, 8
    tip = (int(x + np.cos(angle) * length), int(y + np.sin(angle) * length))
    left = (int(x + np.cos(angle + 2.5) * half_w * 2), int(y + np.sin(angle + 2.5) * half_w * 2))
    right = (int(x + np.cos(angle - 2.5) * half_w * 2), int(y + np.sin(angle - 2.5) * half_w * 2))
    pygame.draw.polygon(screen, CAR_COLOR, [tip, left, right])

    ray_angles = [-0.6, -0.3, 0, 0.3, 0.6]
    for ra in ray_angles:
        ray_a = angle + ra
        dist = env._cast_single_ray(ray_a)
        end_x = int(x + np.cos(ray_a) * dist)
        end_y = int(y + np.sin(ray_a) * dist)
        pygame.draw.line(screen, SENSOR_COLOR, (int(x), int(y)), (end_x, end_y), 1)


def draw_info(screen, font, env, total_reward, episode, mode):
    lines = [
        f"Mode: {mode}",
        f"Episode: {episode}  |  Waypoint: {env.waypoint_idx}/{len(env.track)}",
        f"Speed: {env.speed:.1f}  |  Step: {env.step_count}",
        f"Episode Reward: {total_reward:.1f}",
        "",
        "M: toggle manual/AI  |  R: reset  |  Q: quit",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, INFO_COLOR)
        screen.blit(surf, (10, 10 + i * 22))


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"

    model = ActorCritic(obs_dim=7, act_dim=2)
    try:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded model from {checkpoint}")
    except FileNotFoundError:
        print(f"No checkpoint at {checkpoint}, using random policy")
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving RL - Play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    env = CarEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    episode = 1
    ai_mode = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_reward = 0.0
                    episode += 1
                elif event.key == pygame.K_m:
                    ai_mode = not ai_mode

        if ai_mode:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs)
                action, _, _ = model.get_action(obs_t)
                action = action.numpy()
        else:
            keys = pygame.key.get_pressed()
            steering = 0.0
            throttle = 0.0
            if keys[pygame.K_LEFT]:
                steering = -1.0
            if keys[pygame.K_RIGHT]:
                steering = 1.0
            if keys[pygame.K_UP]:
                throttle = 1.0
            if keys[pygame.K_DOWN]:
                throttle = -1.0
            action = np.array([steering, throttle], dtype=np.float32)

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if done or truncated:
            print(f"Episode {episode}: reward={total_reward:.1f}, waypoints={env.waypoint_idx}, steps={env.step_count}")
            obs, _ = env.reset()
            total_reward = 0.0
            episode += 1

        screen.fill(BG_COLOR)
        draw_track(screen, env)
        draw_car(screen, env)
        mode_str = "AI" if ai_mode else "MANUAL"
        draw_info(screen, font, env, total_reward, episode, mode_str)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
