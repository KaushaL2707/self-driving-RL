import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CarEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 5 distance sensors + speed + angle_to_track = 7 inputs
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )

        # Actions: [steering, throttle] both continuous in [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.track = self._build_track()
        self.reset()

    def _build_track(self):
        # Define a simple oval track as a list of (x, y) waypoints
        waypoints = []
        for i in range(36):
            angle = 2 * np.pi * i / 36
            x = 400 + 300 * np.cos(angle)
            y = 300 + 180 * np.sin(angle)
            waypoints.append((x, y))
        return waypoints

    def reset(self, seed=None):
        self.x, self.y = 700.0, 300.0   # start position
        self.angle = np.pi / 2           # facing up
        self.speed = 0.0
        self.step_count = 0
        self.waypoint_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        steering, throttle = action

        # Update physics
        self.angle += steering * 0.1
        self.speed = np.clip(self.speed + throttle * 0.5, 0, 10)
        self.x += np.cos(self.angle) * self.speed
        self.y += np.sin(self.angle) * self.speed
        self.step_count += 1

        # Check if reached next waypoint
        wx, wy = self.track[self.waypoint_idx % len(self.track)]
        dist_to_wp = np.sqrt((self.x - wx)**2 + (self.y - wy)**2)
        if dist_to_wp < 40:
            self.waypoint_idx += 1

        # Reward: progress along track - penalty for being off-center
        reward = self.speed * 0.1 + (1.0 if dist_to_wp < 40 else 0)

        # Check wall collision (out of bounds)
        done = self._check_collision()
        if done:
            reward = -10.0

        truncated = self.step_count > 1000
        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        sensors = self._raycast()         # 5 values
        speed_norm = self.speed / 10.0    # 1 value
        # angle to next waypoint
        wx, wy = self.track[self.waypoint_idx % len(self.track)]
        angle_to_wp = np.arctan2(wy - self.y, wx - self.x) - self.angle
        angle_norm = angle_to_wp / np.pi  # 1 value
        return np.array([*sensors, speed_norm, angle_norm], dtype=np.float32)

    def _raycast(self):
        # Cast 5 rays at different angles, return normalized distances
        ray_angles = [-0.6, -0.3, 0, 0.3, 0.6]
        distances = []
        for ra in ray_angles:
            dist = self._cast_single_ray(self.angle + ra)
            distances.append(dist / 200.0)  # normalize
        return distances

    def _cast_single_ray(self, angle, max_dist=200):
        for d in range(1, max_dist, 5):
            rx = self.x + np.cos(angle) * d
            ry = self.y + np.sin(angle) * d
            if self._is_wall(rx, ry):
                return d
        return max_dist

    def _is_wall(self, x, y):
        # Check distance from track center
        min_dist = min(
            np.sqrt((x - wx)**2 + (y - wy)**2)
            for wx, wy in self.track
        )
        return min_dist > 80  # track width = 80px

    def _check_collision(self):
        return self._is_wall(self.x, self.y)
