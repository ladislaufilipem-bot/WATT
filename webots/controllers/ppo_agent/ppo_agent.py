#!/usr/bin/env python3
# Webots PPO Agent Controller
# Receives actions from supervisor, sends observations

import sys
import json
import struct
from controller import Robot, Motor, PositionSensor, InertialUnit, Gyro, Accelerometer, GPS, Lidar, Emitter, Receiver

class PPOAgentController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Communication
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.timestep)

        # Sensor setup
        self.setup_sensors()

        # Actuator setup
        self.setup_actuators()

        # State
        self.waiting_for_action = True
        self.current_action = None
        self.step_count = 0

        print("PPO Agent Controller initialized", file=sys.stderr)

    def setup_sensors(self):
        """Initialize sensors."""
        # IMU
        self.imu = self.robot.getDevice("imu")
        self.imu.enable(self.timestep)

        # Gyro
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)

        # Accelerometer
        self.accelerometer = self.robot.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep)

        # GPS
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        # LiDAR
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Joint position sensors
        self.joint_sensors = {}
        joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
            "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
            "right_ankle_pitch", "right_ankle_roll"
        ]

        for name in joint_names:
            sensor = self.robot.getDevice(name + "_sensor")
            if sensor:
                sensor.enable(self.timestep)
                self.joint_sensors[name] = sensor

    def setup_actuators(self):
        """Initialize actuators (motors)."""
        self.motors = {}
        joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
            "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
            "right_ankle_pitch", "right_ankle_roll"
        ]

        for name in joint_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))  # Velocity control mode
                motor.setVelocity(0.0)
                self.motors[name] = motor

    def get_observations(self):
        """Get all sensor observations."""
        obs = {}

        # Joint positions and velocities
        joint_pos = []
        joint_vel = []
        for name in sorted(self.joint_sensors.keys()):
            sensor = self.joint_sensors[name]
            joint_pos.append(sensor.getValue())
            # Velocity not directly available, estimate from position diff
            joint_vel.append(0.0)  # Placeholder

        obs["joint_pos"] = joint_pos
        obs["joint_vel"] = joint_vel

        # IMU (roll, pitch, yaw)
        rpy = self.imu.getRollPitchYaw()
        obs["imu"] = list(rpy)

        # Gyro
        obs["gyro"] = list(self.gyro.getValues())

        # Accelerometer
        obs["accelerometer"] = list(self.accelerometer.getValues())

        # GPS
        obs["gps"] = list(self.gps.getValues())

        # LiDAR
        lidar_data = self.lidar.getRangeImage()
        obs["lidar"] = list(lidar_data) if lidar_data else [0.0] * 360

        return obs

    def apply_action(self, action):
        """Apply action to motors."""
        # Action is array of 12 joint positions
        joint_names = sorted(self.motors.keys())

        for i, name in enumerate(joint_names):
            if i < len(action):
                self.motors[name].setPosition(action[i])

    def send_observations(self, obs, reward, done, info):
        """Send observations to supervisor."""
        message = {
            "type": "observation",
            "observations": obs,
            "reward": reward,
            "terminated": done,
            "truncated": False,
            "info": info
        }

        # Serialize to JSON
        json_str = json.dumps(message)
        self.emitter.send(json_str.encode('utf-8'))

    def send_ready(self):
        """Signal ready to supervisor."""
        message = {"type": "ready"}
        json_str = json.dumps(message)
        self.emitter.send(json_str.encode('utf-8'))

    def receive_command(self):
        """Receive command from supervisor."""
        if self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()

            try:
                command = json.loads(data)
                return command
            except json.JSONDecodeError:
                print(f"Failed to parse command: {data}", file=sys.stderr)

        return None

    def compute_reward(self, obs, action, prev_obs):
        """Compute reward (can be overridden by supervisor)."""
        # Default: forward velocity reward
        reward = 0.0

        # Forward velocity (assuming x-axis forward)
        if "gps" in obs and "gps" in prev_obs:
            forward_vel = (obs["gps"][0] - prev_obs["gps"][0]) / (self.timestep / 1000.0)
            reward += 1.0 * forward_vel

        # Energy penalty
        if action is not None:
            energy = sum(a * a for a in action)
            reward -= 0.01 * energy

        # Alive bonus
        reward += 0.1

        return reward

    def check_termination(self, obs):
        """Check if episode should terminate."""
        # Terminate if fallen (body too low or tilted too much)
        if "imu" in obs:
            roll, pitch, yaw = obs["imu"]
            if abs(roll) > 1.0 or abs(pitch) > 1.0:
                return True

        # Terminate if height too low
        if "gps" in obs:
            if obs["gps"][1] < 0.3:  # Body height < 0.3m
                return True

        return False

    def run(self):
        """Main control loop."""
        prev_obs = None
        prev_action = None

        # Send ready signal
        self.send_ready()

        while self.robot.step(self.timestep) != -1:
            # Receive command
            command = self.receive_command()

            if command:
                cmd_type = command.get("type")

                if cmd_type == "step":
                    # Apply action
                    action = command.get("action")
                    if action:
                        self.apply_action(action)
                        prev_action = action

                    # Get observations
                    obs = self.get_observations()

                    # Compute reward
                    reward = self.compute_reward(obs, prev_action, prev_obs) if prev_obs else 0.0

                    # Check termination
                    terminated = self.check_termination(obs)

                    # Send observations
                    info = {
                        "step": self.step_count,
                        "terminated": terminated
                    }
                    self.send_observations(obs, reward, terminated, info)

                    # Update state
                    prev_obs = obs
                    self.step_count += 1

                    # Reset if terminated
                    if terminated:
                        prev_obs = None
                        prev_action = None
                        self.step_count = 0

                elif cmd_type == "reset":
                    # Reset episode
                    prev_obs = None
                    prev_action = None
                    self.step_count = 0

                    # Get initial observations
                    obs = self.get_observations()
                    self.send_observations(obs, 0.0, False, {"reset": True})

                elif cmd_type == "configure":
                    # Configuration update
                    print(f"Received config: {command.get('config')}", file=sys.stderr)


if __name__ == "__main__":
    controller = PPOAgentController()
    controller.run()