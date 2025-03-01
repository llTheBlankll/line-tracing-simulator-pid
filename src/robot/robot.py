"""
Robot class for the line following robot simulation.
"""
import numpy as np
from collections import deque
import pygame
from src.utils.constants import (
    BASE_SPEED, FRICTION, INERTIA, BLUE, RED, GREEN, 
    HISTORY_SIZE, ENABLE_ROBOT_TRAIL
)

class Robot:
    def __init__(self, x, y, theta=0):
        self.x = x
        self.y = y
        self.theta = theta  # in radians
        self.width = 20
        self.length = 18
        self.sensor_width = 10  # Distance between left and right sensors
        self.sensor_distance = 2  # Distance of sensors from front of robot

        # PID variables
        self.error = 1  # 0-LEFT, 1-FORWARD, 2-RIGHT
        self.last_error = 0
        self.integral = 0

        # Motor speeds with inertia
        self.target_left_speed = BASE_SPEED
        self.target_right_speed = BASE_SPEED
        self.current_left_speed = 0
        self.current_right_speed = 0
        
        # Physics parameters (default to global values)
        self.friction = FRICTION
        self.inertia = INERTIA

        # State
        self.is_stopped = False

        # For visualization - increased history size for better data retention
        self.history_x = deque(maxlen=HISTORY_SIZE)
        self.history_y = deque(maxlen=HISTORY_SIZE)
        self.history_x.append(x)
        self.history_y.append(y)
        self.error_history = deque(maxlen=HISTORY_SIZE)
        self.error_history.append(0)
        self.motor_diff_history = deque(maxlen=HISTORY_SIZE)
        self.motor_diff_history.append(0)
        self.left_speed_history = deque(maxlen=HISTORY_SIZE)
        self.left_speed_history.append(0)
        self.right_speed_history = deque(maxlen=HISTORY_SIZE)
        self.right_speed_history.append(0)
        self.proportional_history = deque(maxlen=HISTORY_SIZE)
        self.proportional_history.append(0)
        self.integral_history = deque(maxlen=HISTORY_SIZE)
        self.integral_history.append(0)
        self.derivative_history = deque(maxlen=HISTORY_SIZE)
        self.derivative_history.append(0)

    def get_sensor_positions(self):
        # Calculate positions of left and right sensors relative to robot center
        sensor_offset = self.sensor_width / 2

        # Front of the robot plus a small offset for sensor placement
        front_x = self.x + (self.length / 2 + self.sensor_distance) * np.cos(self.theta)
        front_y = self.y + (self.length / 2 + self.sensor_distance) * np.sin(self.theta)

        # Calculate perpendicular direction for sensor placement
        perp_theta = self.theta + np.pi / 2

        # Left sensor
        left_sensor_x = front_x + sensor_offset * np.cos(perp_theta)
        left_sensor_y = front_y + sensor_offset * np.sin(perp_theta)

        # Right sensor
        right_sensor_x = front_x - sensor_offset * np.cos(perp_theta)
        right_sensor_y = front_y - sensor_offset * np.sin(perp_theta)

        return (left_sensor_x, left_sensor_y), (right_sensor_x, right_sensor_y)

    def update_pid(self, left_on_line, right_on_line, kp, ki, kd, base_speed):
        # If both sensors detect the line, go straight
        if left_on_line and right_on_line:
            self.is_stopped = False
            self.error = 1  # Center (go straight)
            # Reset integral to prevent windup
            self.integral = 0
        # If no sensors detect the line, keep the last error
        elif not left_on_line and not right_on_line:
            self.is_stopped = False
            # Keep the last error to continue in the same direction
        # If only one sensor detects the line, turn accordingly
        else:
            self.is_stopped = False
            if left_on_line:
                self.error = 2  # Turn RIGHT when LEFT sensor detects line
            elif right_on_line:
                self.error = 0  # Turn LEFT when RIGHT sensor detects line

        # PID calculations
        proportional = kp * (self.error - 1)  # Adjusted so 1 (forward) is zero error
        self.integral += (self.error - 1) * ki
        derivative = kd * ((self.error - 1) - (self.last_error - 1))

        motor_speed_diff = proportional + self.integral + derivative

        # Update target motor speeds
        self.target_left_speed = base_speed - motor_speed_diff
        self.target_right_speed = base_speed + motor_speed_diff

        # Store values for visualization
        self.motor_diff_history.append(motor_speed_diff)
        self.error_history.append(self.error - 1)  # Store centered error for plotting
        self.left_speed_history.append(self.current_left_speed)
        self.right_speed_history.append(self.current_right_speed)
        self.proportional_history.append(proportional)
        self.integral_history.append(self.integral)
        self.derivative_history.append(derivative)

        # Update last error
        self.last_error = self.error

    def update_physics(self, dt):
        # Apply inertia to motor speeds (smooth acceleration/deceleration)
        self.current_left_speed = (
            1 - self.inertia
        ) * self.current_left_speed + self.inertia * self.target_left_speed
        self.current_right_speed = (
            1 - self.inertia
        ) * self.current_right_speed + self.inertia * self.target_right_speed

        # Apply friction
        self.current_left_speed *= self.friction
        self.current_right_speed *= self.friction

    def move(self, dt):
        # Update physics first
        self.update_physics(dt)

        # Calculate robot movement based on motor speeds
        # Forward speed is average of two motors
        forward_speed = (self.current_left_speed + self.current_right_speed) / 2

        # Rotation depends on difference between motors
        rotation_speed = (
            self.current_right_speed - self.current_left_speed
        ) / self.width

        # Update position and orientation
        distance = forward_speed * dt
        delta_theta = rotation_speed * dt

        self.theta += delta_theta
        self.x += distance * np.cos(self.theta)
        self.y += distance * np.sin(self.theta)

        # Record position for trail
        self.history_x.append(self.x)
        self.history_y.append(self.y)

    def reset(self, x, y, theta):
        # Remember current parameters
        width = self.width
        length = self.length
        sensor_width = self.sensor_width
        sensor_distance = self.sensor_distance
        friction = self.friction
        inertia = self.inertia
        
        # Reset position and motion
        self.x = x
        self.y = y
        self.theta = theta
        self.error = 1
        self.last_error = 0
        self.integral = 0
        self.target_left_speed = BASE_SPEED
        self.target_right_speed = BASE_SPEED
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.is_stopped = False
        
        # Restore parameters
        self.width = width
        self.length = length
        self.sensor_width = sensor_width
        self.sensor_distance = sensor_distance
        self.friction = friction
        self.inertia = inertia
        
        # Clear history
        self.history_x.clear()
        self.history_y.clear()
        self.history_x.append(x)
        self.history_y.append(y)
        self.error_history.clear()
        self.error_history.append(0)
        self.motor_diff_history.clear()
        self.motor_diff_history.append(0)
        self.left_speed_history.clear()
        self.left_speed_history.append(0)
        self.right_speed_history.clear()
        self.right_speed_history.append(0)
        self.proportional_history.clear()
        self.proportional_history.append(0)
        self.integral_history.clear()
        self.integral_history.append(0)
        self.derivative_history.clear()
        self.derivative_history.append(0)

    def draw(self, screen, scale=1.0):
        # Draw trail with reduced number of points for better performance
        if ENABLE_ROBOT_TRAIL:
            # Only draw every Nth point from the history
            trail_stride = 5  # Increased from 2 to 5 for better performance
            points = []
            for i in range(0, len(self.history_x), trail_stride):
                points.append(
                    (int(self.history_x[i] * scale), int(self.history_y[i] * scale))
                )

            if len(points) > 1:
                pygame.draw.lines(screen, BLUE, False, points, 2)

        # Draw robot body
        corners = [
            (-self.width / 2, -self.length / 2),
            (self.width / 2, -self.length / 2),
            (self.width / 2, self.length / 2),
            (-self.width / 2, self.length / 2),
        ]

        # Pre-compute sin and cos values for optimization
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Rotate and translate corners
        rotated_corners = []
        for corner in corners:
            x, y = corner
            # Rotate
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            # Translate
            x_final = int((self.x + x_rot) * scale)
            y_final = int((self.y + y_rot) * scale)
            rotated_corners.append((x_final, y_final))

        pygame.draw.polygon(screen, BLUE, rotated_corners)

        # Draw sensors
        (left_x, left_y), (right_x, right_y) = self.get_sensor_positions()
        pygame.draw.circle(screen, RED, (int(left_x * scale), int(left_y * scale)), 3)
        pygame.draw.circle(
            screen, GREEN, (int(right_x * scale), int(right_y * scale)), 3
        )

    def get_data_for_plotting(self):
        # Create copies of the data for thread safety
        return {
            "error_history": list(self.error_history),
            "motor_diff_history": list(self.motor_diff_history),
            "left_speed_history": list(self.left_speed_history),
            "right_speed_history": list(self.right_speed_history),
            "proportional_history": list(self.proportional_history),
            "integral_history": list(self.integral_history),
            "derivative_history": list(self.derivative_history),
            "history_x": list(self.history_x),
            "history_y": list(self.history_y),
        } 