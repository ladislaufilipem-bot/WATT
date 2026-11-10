#!/usr/bin/env python3
# Webots Supervisor Controller
# Manages environment, sends actions, receives observations

import sys
import json
import numpy as np
from controller import Supervisor, Emitter, Receiver

class PPOSupervisor:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Communication
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.timestep)

        # Robot node
        self.robot_node = self.supervisor.getFromDef("PPO_ROBOT")
        if not self.robot_node:
            print("ERROR: Robot node PPO_ROBOT not found", file=sys.stderr)

        # State
        self.episode_count = 0
        self.step_count = 0
        self.max_episode_steps = 1000
        self.waiting_for_obs = False

        # Action/observation spaces
        self.obs_keys = [
            "joint_pos", "joint_vel", "imu", "gyro",
            "accelerometer", "gps", "lidar"
        ]
        self.action_dim = 12

        print("PPO Supervisor initialized", file=sys.stderr)

    def send_action(self, action, reset=False):
        """Send action to agent controller."""
        message = {
            "type": "step",
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "reset": reset
        }
        json_str = json.dumps(message)
        self.emitter.send(json_str.encode('utf-8'))
        self.waiting_for_obs = True

    def send_reset(self, seed=None, config=None):
        """Send reset command."""
        message = {
            "type": "reset",
            "seed": seed,
            "config": config or {}
        }
        json_str = json.dumps(message)
        self.emitter.send(json_str.encode('utf-8'))
        self.waiting_for_obs = True

    def send_configure(self, obs_space, action_space, config):
        """Send configuration."""
        message = {
            "type": "configure",
            "obs_space": obs_space,
            "action_space": action_space,
            "config": config
        }
        json_str = json.dumps(message)
        self.emitter.send(json_str.encode('utf-8'))

    def receive_observation(self):
        """Receive observation from agent."""
        if self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()

            try:
                obs = json.loads(data)
                self.waiting_for_obs = False
                return obs
            except json.JSONDecodeError:
                print(f"Failed to parse observation: {data}", file=sys.stderr)

        return None

    def reset_robot(self):
        """Reset robot position in simulation."""
        if self.robot_node:
            # Reset position
            self.robot_node.getField("translation").setSFVec3f([0, 0.5, 0])
            self.robot_node.getField("rotation").setSFRotation([0, 0, 1, 0])

            # Reset velocity
            self.robot_node.resetPhysics()

            # Reset joint positions
            joint_names = [
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
                "left_ankle_pitch", "left_ankle_roll",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
                "right_ankle_pitch", "right_ankle_roll"
            ]

            for name in joint_names:
                joint = self.robot_node.getFromProtoDef(name.upper())
                if joint:
                    joint.getField("position").setSFFloat(0.0)

    def get_robot_state(self):
        """Get robot state from simulation."""
        if not self.robot_node:
            return {}

        # Position
        pos = self.robot_node.getPosition()
        rot = self.robot_node.getOrientation()

        # Velocity
        vel = self.robot_node.getVelocity()

        return {
            "position": pos,
            "rotation": rot,
            "linear_velocity": vel[:3],
            "angular_velocity": vel[3:]
        }

    def step_simulation(self):
        """Step the simulation."""
        return self.supervisor.step(self.timestep)

    def run(self):
        """Main supervisor loop - for testing without external RL."""
        # Send initial configuration
        obs_space = {
            "joint_pos": {"shape": [12], "min": -3.14, "max": 3.14},
            "joint_vel": {"shape": [12], "min": -10.0, "max": 10.0},
            "imu": {"shape": [3], "min": -3.14, "max": 3.14},
            "gyro": {"shape": [3], "min": -10.0, "max": 10.0},
            "accelerometer": {"shape": [3], "min": -20.0, "max": 20.0},
            "gps": {"shape": [3], "min": -10.0, "max": 10.0},
            "lidar": {"shape": [360], "min": 0.0, "max": 10.0}
        }

        action_space = {
            "motor_pos": {"shape": [12], "min": -3.14, "max": 3.14}
        }

        self.send_configure(obs_space, action_space, {})

        # Wait for ready
        while self.step_simulation() != -1:
            obs = self.receive_observation()
            if obs and obs.get("type") == "ready":
                break

        print("Agent ready, starting episodes", file=sys.stderr)

        # Run episodes with random policy (for testing)
        for episode in range(10):
            self.episode_count += 1
            self.step_count = 0

            # Reset
            self.reset_robot()
            self.send_reset(seed=episode)

            # Wait for initial observation
            while self.waiting_for_obs and self.step_simulation() != -1:
                self.receive_observation()

            episode_reward = 0.0

            # Episode loop
            while self.step_count < self.max_episode_steps and self.step_simulation() != -1:
                # Random action
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)

                # Send action
                self.send_action(action)

                # Wait for observation
                while self.waiting_for_obs and self.step_simulation() != -1:
                    obs = self.receive_observation()

                    if obs:
                        episode_reward += obs.get("reward", 0.0)
                        self.step_count += 1

                        if obs.get("terminated", False):
                            break

            print(f"Episode {episode}: reward={episode_reward:.2f}, steps={self.step_count}", file=sys.stderr)

        print("Supervisor finished", file=sys.stderr)


if __name__ == "__main__":
    supervisor = PPOSupervisor()
    supervisor.run()