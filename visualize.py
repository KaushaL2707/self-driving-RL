import pygame
import numpy as np
from env import CarEnv

# --- Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60
BG_COLOR = (30, 30, 30)
TRACK_COLOR = (80, 80, 80)
TRACK_EDGE_COLOR = (200, 200, 200)
CAR_COLOR = (255, 80, 80)
WAYPOINT_COLOR = (50, 50, 100)
NEXT_WP_COLOR = (0, 255, 100)
SENSOR_COLOR = (255, 255, 0, 120)
INFO_COLOR = (220, 220, 220)

def draw_track(screen, env):
    track = env.track
    n = len(track)
    track_width = 80  # matches _is_wall threshold

    # Build inner and outer edge points
    inner_pts = []
    outer_pts = []
    for i in range(n):
        px, py = track[i]
        nx, ny = track[(i + 1) % n]
        # direction along track
        dx, dy = nx - px, ny - py
        length = np.sqrt(dx**2 + dy**2)
        # normal (pointing outward)
        norm_x, norm_y = -dy / length, dx / length
        inner_pts.append((px - norm_x * track_width, py - norm_y * track_width))
        outer_pts.append((px + norm_x * track_width, py + norm_y * track_width))

    # Draw filled track by drawing quads between inner and outer edges
    for i in range(n):
        j = (i + 1) % n
        quad = [inner_pts[i], inner_pts[j], outer_pts[j], outer_pts[i]]
        pygame.draw.polygon(screen, TRACK_COLOR, quad)

    # Draw edge lines
    pygame.draw.lines(screen, TRACK_EDGE_COLOR, True, inner_pts, 2)
    pygame.draw.lines(screen, TRACK_EDGE_COLOR, True, outer_pts, 2)

    # Draw waypoint dots
    for i, (wx, wy) in enumerate(track):
        color = NEXT_WP_COLOR if i == env.waypoint_idx % n else WAYPOINT_COLOR
        pygame.draw.circle(screen, color, (int(wx), int(wy)), 4)


def draw_car(screen, env):
    x, y, angle = env.x, env.y, env.angle
    # Car body (triangle pointing in direction of travel)
    length, half_w = 20, 8
    tip = (int(x + np.cos(angle) * length), int(y + np.sin(angle) * length))
    left = (int(x + np.cos(angle + 2.5) * half_w * 2), int(y + np.sin(angle + 2.5) * half_w * 2))
    right = (int(x + np.cos(angle - 2.5) * half_w * 2), int(y + np.sin(angle - 2.5) * half_w * 2))
    pygame.draw.polygon(screen, CAR_COLOR, [tip, left, right])

    # Draw sensor rays
    ray_angles = [-0.6, -0.3, 0, 0.3, 0.6]
    for ra in ray_angles:
        ray_a = angle + ra
        dist = env._cast_single_ray(ray_a)
        end_x = x + np.cos(ray_a) * dist
        end_y = y + np.sin(ray_a) * dist
        pygame.draw.line(screen, SENSOR_COLOR, (int(x), int(y)), (int(end_x), int(end_y)), 1)


def draw_info(screen, font, env, reward, action):
    lines = [
        f"Speed: {env.speed:.1f}",
        f"Waypoint: {env.waypoint_idx}/{len(env.track)}",
        f"Step: {env.step_count}",
        f"Reward: {reward:.2f}",
        f"Steer: {action[0]:.2f}  Throttle: {action[1]:.2f}",
        "",
        "Arrow keys to drive | R to reset | Q to quit",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, INFO_COLOR)
        screen.blit(surf, (10, 10 + i * 22))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving RL - Track Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    env = CarEnv()
    obs, _ = env.reset()
    reward = 0.0
    action = np.array([0.0, 0.0])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, _ = env.reset()
                reward = 0.0

        # Read held keys for continuous control
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

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
            reward = 0.0

        # Draw
        screen.fill(BG_COLOR)
        draw_track(screen, env)
        draw_car(screen, env)
        draw_info(screen, font, env, reward, action)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
