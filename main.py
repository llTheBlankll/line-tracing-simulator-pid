import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
import sys
import os
import threading
import queue
from collections import deque
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from pygame.locals import *  # Import pygame constants to fix linter errors

# Fix linter errors by explicitly importing pygame constants
from pygame.locals import (
    QUIT, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION,
    K_ESCAPE, K_SPACE, K_r
)

# Constants - initial values that will be adjustable via sliders
BASE_SPEED = 40
KP = 1.0
KI = 0.0
KD = 0.005

# Physics constants
FRICTION = 0.5  # Friction coefficient (1.0 = no friction)
INERTIA = 0.5  # How quickly the robot responds to speed changes (1.0 = instant)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Screen dimensions
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
STATS_HEIGHT = 200
WINDOW_HEIGHT = SCREEN_HEIGHT + STATS_HEIGHT

# Scale factor (pixels per simulation unit)
SCALE = 2

# Performance settings
HISTORY_SIZE = 100  # Reduced from 500 to improve performance
PLOT_UPDATE_INTERVAL = 5  # Update plots more frequently
PHYSICS_SUBSTEPS = 2  # Number of physics updates per frame for smoother simulation
FPS_UPDATE_INTERVAL = 10  # Update FPS display every N frames

# Performance optimization: Cache for line segment calculations in is_on_line function
segment_cache = {}

class Robot:
    def __init__(self, x, y, theta=0):
        self.x = x
        self.y = y
        self.theta = theta  # in radians
        self.width = 20
        self.length = 14
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

        # For visualization - reduced history size for better performance
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
        # Only draw every Nth point from the history
        trail_stride = 2
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


def generate_track():
    """
    Generate a smooth track with natural curves using Bezier curves.
    The track will be continuous and connect back to the start.
    """
    # Define the center of the track
    center_x, center_y = SCREEN_WIDTH / (2 * SCALE), SCREEN_HEIGHT / (2 * SCALE)
    
    # Create empty arrays for the track
    track_x = []
    track_y = []
    
    # Track parameters
    num_control_points = np.random.randint(5, 9)  # Number of control points for the track
    min_radius = 80  # Minimum distance from center
    max_radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) / (3 * SCALE)  # Maximum distance from center
    
    # Generate control points in a circle around the center
    angles = np.linspace(0, 2 * np.pi, num_control_points, endpoint=False)
    
    # Add some randomness to the angles to make the track less regular
    angles += np.random.uniform(-0.2, 0.2, num_control_points)
    
    # Generate control points with varying distances from center
    control_points = []
    for angle in angles:
        radius = np.random.uniform(min_radius, max_radius)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        control_points.append((x, y))
    
    # Add the first point again to close the loop
    control_points.append(control_points[0])
    
    # Generate Bezier curves between each pair of control points
    for i in range(len(control_points) - 1):
        p0 = control_points[i]
        p3 = control_points[i + 1]
        
        # Calculate the distance between points
        dist = np.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
        
        # Calculate the direction vector between points
        dir_x = p3[0] - p0[0]
        dir_y = p3[1] - p0[1]
        
        # Normalize the direction vector
        length = np.sqrt(dir_x**2 + dir_y**2)
        if length > 0:
            dir_x /= length
            dir_y /= length
        
        # Calculate perpendicular vector (for control points)
        perp_x = -dir_y
        perp_y = dir_x
        
        # Control point distances (30-50% of the distance between points)
        ctrl_dist = dist * np.random.uniform(0.3, 0.5)
        
        # Add some randomness to control points to create varied curves
        # Control points are offset perpendicular to the line between points
        perp_offset1 = np.random.uniform(-0.5, 0.5) * dist
        perp_offset2 = np.random.uniform(-0.5, 0.5) * dist
        
        # Calculate control points
        p1 = (
            p0[0] + dir_x * ctrl_dist + perp_x * perp_offset1,
            p0[1] + dir_y * ctrl_dist + perp_y * perp_offset1
        )
        
        p2 = (
            p3[0] - dir_x * ctrl_dist + perp_x * perp_offset2,
            p3[1] - dir_y * ctrl_dist + perp_y * perp_offset2
        )
        
        # Generate points along the Bezier curve
        num_points = 50  # Number of points per curve segment
        for t in np.linspace(0, 1, num_points):
            # Cubic Bezier formula
            x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
            y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
            track_x.append(x)
            track_y.append(y)
    
    # Convert to numpy arrays
    track_x = np.array(track_x)
    track_y = np.array(track_y)
    
    return track_x, track_y


def is_on_line(sensor_x, sensor_y, track_x, track_y, threshold=3.5):
    """
    Check if a sensor is on the line with improved detection reliability.
    Uses a more efficient algorithm to check proximity to line segments.
    Performance-optimized version with caching.
    
    Args:
        sensor_x, sensor_y: Coordinates of the sensor
        track_x, track_y: Arrays of track coordinates
        threshold: Maximum distance to be considered "on the line"
        
    Returns:
        True if the sensor is on the line, False otherwise
    """
    global segment_cache
    
    # Use a larger stride for initial check (optimization)
    stride = 8
    
    # Calculate distances to track points (vectorized)
    distances = np.sqrt(
        (track_x[::stride] - sensor_x) ** 2 + (track_y[::stride] - sensor_y) ** 2
    )
    
    # If we find a close point in the subset, do a more detailed check
    min_dist = np.min(distances)
    if min_dist < threshold * 3:  # Use a larger initial threshold
        # Find the index of the closest point in our strided array
        min_idx = np.argmin(distances) * stride
        
        # Define a smaller window around the closest point (optimization)
        window_size = 16
        start_idx = max(0, min_idx - window_size)
        end_idx = min(len(track_x), min_idx + window_size)
        
        # Check distances for the window of points
        window_distances = np.sqrt(
            (track_x[start_idx:end_idx] - sensor_x) ** 2
            + (track_y[start_idx:end_idx] - sensor_y) ** 2
        )
        
        # Check if any point in the window is close enough
        if np.min(window_distances) < threshold:
            return True
            
        # Check line segments, using cache when possible
        for i in range(start_idx, end_idx - 1):
            # Create a unique key for this segment
            segment_key = (i, i+1)
            
            # Calculate or retrieve cached values
            if segment_key not in segment_cache:
                x1, y1 = track_x[i], track_y[i]
                x2, y2 = track_x[i+1], track_y[i+1]
                
                # Precompute some values for line segment distance calculations
                dx = x2 - x1
                dy = y2 - y1
                line_length_sq = dx**2 + dy**2
                
                # Store these values in the cache
                if line_length_sq > 0:  # Avoid division by zero
                    segment_cache[segment_key] = (x1, y1, x2, y2, dx, dy, line_length_sq)
            
            # If segment is in cache, use the cached values
            if segment_key in segment_cache:
                x1, y1, x2, y2, dx, dy, line_length_sq = segment_cache[segment_key]
                
                # Calculate projection of point onto line segment
                t = max(0, min(1, ((sensor_x - x1) * dx + (sensor_y - y1) * dy) / line_length_sq))
                
                # Calculate closest point on line segment
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                
                # Check distance to closest point
                dist = np.sqrt((sensor_x - closest_x) ** 2 + (sensor_y - closest_y) ** 2)
                if dist < threshold:
                    return True
    
    return False


def point_to_line_segment_distance(px, py, x1, y1, x2, y2):
    """
    Calculate the distance from point (px, py) to line segment (x1, y1) - (x2, y2).
    This is more accurate than just checking distances to points.
    """
    # Calculate squared length of line segment
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    
    # If the line segment is just a point, return distance to that point
    if line_length_sq == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Calculate projection of point onto line segment
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
    
    # Calculate closest point on line segment
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    
    # Return distance to closest point
    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def is_out_of_bounds(x, y):
    # Check if the robot is out of the screen bounds
    margin = 50
    return (
        x < -margin
        or x > SCREEN_WIDTH / SCALE + margin
        or y < -margin
        or y > SCREEN_HEIGHT / SCALE + margin
    )


def draw_track(screen, track_x, track_y, scale=1.0, thickness=6):
    """
    Draw the track with improved visibility.
    Uses a thicker line and anti-aliasing effect for better appearance.
    Optimized for performance with point reduction.
    
    Args:
        screen: Pygame surface to draw on
        track_x, track_y: Track coordinates
        scale: Scale factor for drawing
        thickness: Thickness of the track line
    """
    # Convert track points to screen coordinates with greater stride for performance
    stride = 3  # Use every 3rd point for better performance
    points = [(int(track_x[i] * scale), int(track_y[i] * scale)) 
              for i in range(0, len(track_x), stride)]
    
    if len(points) > 1:
        # Draw a white border around the track for better visibility
        border_thickness = thickness + 4
        pygame.draw.lines(screen, WHITE, False, points, border_thickness)
        
        # Draw the main track line (thicker)
        pygame.draw.lines(screen, BLACK, False, points, thickness)
        
        # Draw a thinner gray line in the center for a more professional look
        center_thickness = max(1, thickness // 3)
        pygame.draw.lines(screen, (80, 80, 80), False, points, center_thickness)


class PIDSlider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.handle_radius = 10
        self.handle_pos = self._value_to_pos(initial_val)
        self.dragging = False

    def _value_to_pos(self, value):
        # Convert value to position
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + ratio * self.rect.width)

    def _pos_to_value(self, pos):
        # Convert position to value
        ratio = max(0, min(1, (pos - self.rect.x) / self.rect.width))
        return self.min_val + ratio * (self.max_val - self.min_val)

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if pygame.Rect(
                self.handle_pos - self.handle_radius,
                self.rect.y - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2,
            ).collidepoint(event.pos):
                self.dragging = True
        elif event.type == MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self.handle_pos = max(
                self.rect.x, min(self.rect.x + self.rect.width, event.pos[0])
            )
            self.value = self._pos_to_value(self.handle_pos)

    def draw(self, screen, font):
        # Draw slider track
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        # Draw handle
        pygame.draw.circle(
            screen,
            BLACK,
            (self.handle_pos, self.rect.y + self.rect.height // 2),
            self.handle_radius,
        )

        # Draw label and value
        label_text = font.render(f"{self.label}: {self.value:.3f}", True, BLACK)
        screen.blit(label_text, (self.rect.x, self.rect.y - 20))


class MatplotlibWindow:
    def __init__(self, data_queue):
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Line Following Robot - Data Visualization")
        self.root.geometry("1000x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make sure the window appears on top initially
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)

        # Store the data queue for thread communication
        self.data_queue = data_queue
        self.last_update_time = time.time()
        self.update_interval = 0.3  # Reduced update frequency for better performance (300ms)

        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create subplots
        self.ax1 = self.fig.add_subplot(321)  # Error
        self.ax2 = self.fig.add_subplot(322)  # Motor speeds
        self.ax3 = self.fig.add_subplot(323)  # PID components
        self.ax4 = self.fig.add_subplot(324)  # Motor difference
        self.ax5 = self.fig.add_subplot(325)  # Position X
        self.ax6 = self.fig.add_subplot(326)  # Position Y

        # Set up plot lines for faster updating
        x_dummy = [0]
        y_dummy = [0]
        
        # Set titles and initialize plot lines
        self.ax1.set_title("Error")
        self.error_line, = self.ax1.plot(x_dummy, y_dummy, "r-")
        self.ax1.set_ylim(-1.5, 1.5)
        
        self.ax2.set_title("Motor Speeds")
        self.left_speed_line, = self.ax2.plot(x_dummy, y_dummy, "b-", label="Left")
        self.right_speed_line, = self.ax2.plot(x_dummy, y_dummy, "g-", label="Right")
        self.ax2.legend()
        
        self.ax3.set_title("PID Components")
        self.proportional_line, = self.ax3.plot(x_dummy, y_dummy, "r-", label="P")
        self.integral_line, = self.ax3.plot(x_dummy, y_dummy, "g-", label="I")
        self.derivative_line, = self.ax3.plot(x_dummy, y_dummy, "b-", label="D")
        self.ax3.legend()
        
        self.ax4.set_title("Motor Speed Difference")
        self.motor_diff_line, = self.ax4.plot(x_dummy, y_dummy, "g-")
        
        self.ax5.set_title("Position X")
        self.pos_x_line, = self.ax5.plot(x_dummy, y_dummy, "k-")
        
        self.ax6.set_title("Position Y")
        self.pos_y_line, = self.ax6.plot(x_dummy, y_dummy, "k-")

        # Set grid
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.grid(True)

        self.fig.tight_layout()
        self.is_closed = False
        
        # Initialize with empty data
        self.current_data = {
            "error_history": [0],
            "motor_diff_history": [0],
            "left_speed_history": [0],
            "right_speed_history": [0],
            "proportional_history": [0],
            "integral_history": [0],
            "derivative_history": [0],
            "history_x": [0],
            "history_y": [0],
        }
        
    def check_queue(self):
        """Check for new data and update current data"""
        if self.is_closed:
            return
            
        try:
            # Process any data in the queue (only get the latest)
            latest_data = None
            while not self.data_queue.empty():
                latest_data = self.data_queue.get(block=False)
                
            # If we got new data, store it
            if latest_data:
                self.current_data = latest_data
        except Exception as e:
            print(f"Error in check_queue: {e}")

    def update_plots(self):
        if self.is_closed:
            return

        try:
            # Get current data
            error_data = self.current_data["error_history"]
            motor_diff_data = self.current_data["motor_diff_history"]
            left_speed = self.current_data["left_speed_history"]
            right_speed = self.current_data["right_speed_history"]
            proportional = self.current_data["proportional_history"]
            integral = self.current_data["integral_history"]
            derivative = self.current_data["derivative_history"]
            x_range = list(range(len(error_data)))
            
            # Downsample for better performance if data is large
            stride = max(1, len(error_data) // 50)  # More aggressive downsampling for performance
            
            # Update plot data
            self.error_line.set_data(x_range[::stride], error_data[::stride])
            self.motor_diff_line.set_data(x_range[::stride], motor_diff_data[::stride])
            self.left_speed_line.set_data(x_range[::stride], left_speed[::stride])
            self.right_speed_line.set_data(x_range[::stride], right_speed[::stride])
            self.proportional_line.set_data(x_range[::stride], proportional[::stride])
            self.integral_line.set_data(x_range[::stride], integral[::stride])
            self.derivative_line.set_data(x_range[::stride], derivative[::stride])
            
            # Update position plots if data exists
            if len(self.current_data.get("history_x", [])) > 0:
                x_data = self.current_data["history_x"]
                x_indices = list(range(len(x_data)))
                self.pos_x_line.set_data(x_indices[::stride], x_data[::stride])
                self.ax5.relim()
                self.ax5.autoscale_view()
                
            if len(self.current_data.get("history_y", [])) > 0:
                y_data = self.current_data["history_y"]
                y_indices = list(range(len(y_data)))
                self.pos_y_line.set_data(y_indices[::stride], y_data[::stride])
                self.ax6.relim()
                self.ax6.autoscale_view()
            
            # Update axis limits for other plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.relim()
                ax.autoscale_view()
                
            # Keep the y-limits fixed for the error plot
            self.ax1.set_ylim(-1.5, 1.5)

            # Draw the updates
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating plots: {e}")

    def on_closing(self):
        self.is_closed = True
        try:
            self.root.destroy()
        except:
            pass
    
    def update(self):
        """Update the Tkinter window and process events"""
        if not self.is_closed:
            # Always process Tkinter events
            self.root.update()
            
            # Always check the queue for new data
            self.check_queue()
            
            # Only update plots periodically for better performance
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.update_plots()
                self.last_update_time = current_time


def main():
    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Line Following Robot Simulation")

    # Set up the clock
    clock = pygame.time.Clock()

    # Generate initial track
    track_x, track_y = generate_track()

    # Initialize robot at the start of the track
    robot = Robot(track_x[0], track_y[0])
    robot.theta = np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0])

    # Create font for text
    font = pygame.font.SysFont(None, 24)

    # Create restart button
    button_rect = pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT + 10, 140, 40)

    # Create PID sliders
    kp_slider = PIDSlider(50, SCREEN_HEIGHT + 50, 200, 20, 0, 20, KP, "Kp")
    ki_slider = PIDSlider(50, SCREEN_HEIGHT + 100, 200, 20, 0, 0.1, KI, "Ki")
    kd_slider = PIDSlider(50, SCREEN_HEIGHT + 150, 200, 20, 0, 20, KD, "Kd")
    speed_slider = PIDSlider(
        350, SCREEN_HEIGHT + 50, 200, 20, 10, 100, BASE_SPEED, "Base Speed"
    )
    
    # Create physics and robot parameter sliders
    friction_slider = PIDSlider(350, SCREEN_HEIGHT + 100, 200, 20, 0.1, 1.0, robot.friction, "Friction")
    inertia_slider = PIDSlider(350, SCREEN_HEIGHT + 150, 200, 20, 0.1, 1.0, robot.inertia, "Inertia")
    sensor_width_slider = PIDSlider(600, SCREEN_HEIGHT + 50, 200, 20, 4, 16, robot.sensor_width, "Sensor Width")
    sensor_dist_slider = PIDSlider(600, SCREEN_HEIGHT + 100, 200, 20, 2, 10, robot.sensor_distance, "Sensor Distance")
    robot_width_slider = PIDSlider(600, SCREEN_HEIGHT + 150, 200, 20, 5, 20, robot.width, "Robot Width")
    robot_length_slider = PIDSlider(850, SCREEN_HEIGHT + 50, 200, 20, 5, 30, robot.length, "Robot Length")
    track_thickness_slider = PIDSlider(850, SCREEN_HEIGHT + 100, 200, 20, 2, 10, 6, "Track Thickness")
    
    # Add slow motion slider
    sim_speed_slider = PIDSlider(850, SCREEN_HEIGHT + 150, 200, 20, 0.1, 2.0, 1.0, "Sim Speed")

    # Time step
    dt = 0.1 / PHYSICS_SUBSTEPS  # Smaller time step for more accurate physics

    # Create a queue for thread communication
    data_queue = queue.Queue(maxsize=20)  # Increased queue size for better data flow

    # Create Matplotlib window
    matplotlib_window = MatplotlibWindow(data_queue)

    # Main game loop
    running = True
    paused = False
    left_on_line = False
    right_on_line = False
    frame_count = 0
    fps = 0
    
    # Reduce plot update interval for more frequent updates
    plot_update_interval = 15  # Further increased to update less frequently for better performance
    current_track_thickness = 6  # Default track thickness
    
    # Current physics parameters
    current_friction = FRICTION
    current_inertia = INERTIA
    
    # Time tracking
    last_data_send_time = time.time()
    plot_update_delay = 0.2  # Only send data to plots every 200ms
    
    # For FPS calculation
    fps_history = deque(maxlen=30)  # Store recent FPS values for smoother display
    
    # Track rendering optimization
    track_points = []
    for i in range(0, len(track_x), 3):
        track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))

    try:
        while running:
            # Start frame timing
            frame_start_time = time.time()
            
            # Process Tkinter events at a reduced rate (every 2 frames)
            if frame_count % 2 == 0:
                if matplotlib_window.is_closed:
                    running = False
                else:
                    matplotlib_window.update()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        paused = not paused
                    elif event.key == K_r:
                        # Generate new track
                        track_x, track_y = generate_track()
                        # Reset robot
                        robot.reset(
                            track_x[0],
                            track_y[0],
                            np.arctan2(
                                track_y[1] - track_y[0], track_x[1] - track_x[0]
                            ),
                        )
                        # Clear segment cache when track changes
                        segment_cache.clear()
                        
                        # Update pre-rendered track points
                        track_points = []
                        for i in range(0, len(track_x), 3):
                            track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))
                            
                elif event.type == MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        # Generate new track
                        track_x, track_y = generate_track()
                        # Reset robot when button is clicked
                        robot.reset(
                            track_x[0],
                            track_y[0],
                            np.arctan2(
                                track_y[1] - track_y[0], track_x[1] - track_x[0]
                            ),
                        )
                        # Clear segment cache when track changes
                        segment_cache.clear()
                        
                        # Update pre-rendered track points
                        track_points = []
                        for i in range(0, len(track_x), 3):
                            track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))

                # Handle slider events
                kp_slider.handle_event(event)
                ki_slider.handle_event(event)
                kd_slider.handle_event(event)
                speed_slider.handle_event(event)
                
                # Handle new slider events
                friction_slider.handle_event(event)
                inertia_slider.handle_event(event)
                sensor_width_slider.handle_event(event)
                sensor_dist_slider.handle_event(event)
                robot_width_slider.handle_event(event)
                robot_length_slider.handle_event(event)
                track_thickness_slider.handle_event(event)
                sim_speed_slider.handle_event(event)  # Handle slow-motion slider
                
                # Update parameters from sliders
                robot.sensor_width = sensor_width_slider.value
                robot.sensor_distance = sensor_dist_slider.value
                robot.width = robot_width_slider.value
                robot.length = robot_length_slider.value
                current_track_thickness = int(track_thickness_slider.value)
                current_friction = friction_slider.value
                current_inertia = inertia_slider.value

            if not paused:
                # Get sensor positions
                (left_x, left_y), (right_x, right_y) = robot.get_sensor_positions()

                # Check if sensors are on the line
                left_on_line = is_on_line(left_x, left_y, track_x, track_y)
                right_on_line = is_on_line(right_x, right_y, track_x, track_y)

                # Update robot based on PID control with slider values
                robot.update_pid(
                    left_on_line,
                    right_on_line,
                    kp_slider.value,
                    ki_slider.value,
                    kd_slider.value,
                    speed_slider.value,
                )
                
                # Store current physics parameters for the robot to use
                robot.friction = current_friction
                robot.inertia = current_inertia

                # Apply slow motion by adjusting the time step
                sim_speed = sim_speed_slider.value
                current_dt = dt * sim_speed
                
                # Multiple physics updates per frame for smoother simulation
                for _ in range(PHYSICS_SUBSTEPS):
                    robot.move(current_dt)

                # Check if robot is out of bounds
                if is_out_of_bounds(robot.x, robot.y):
                    # Generate new track
                    track_x, track_y = generate_track()
                    # Reset robot to starting position
                    robot.reset(
                        track_x[0],
                        track_y[0],
                        np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0]),
                    )
                    # Clear segment cache when track changes
                    segment_cache.clear()
                    
                    # Update pre-rendered track points
                    track_points = []
                    for i in range(0, len(track_x), 3):
                        track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))

            # Clear the screen
            screen.fill(WHITE)

            # Draw the track with current thickness
            draw_track(screen, track_x, track_y, SCALE, current_track_thickness)

            # Draw the robot
            robot.draw(screen, SCALE)

            # Draw dividing line
            pygame.draw.line(
                screen, BLACK, (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT), 2
            )

            # Draw sliders
            kp_slider.draw(screen, font)
            ki_slider.draw(screen, font)
            kd_slider.draw(screen, font)
            speed_slider.draw(screen, font)
            
            # Draw new sliders
            friction_slider.draw(screen, font)
            inertia_slider.draw(screen, font)
            sensor_width_slider.draw(screen, font)
            sensor_dist_slider.draw(screen, font)
            robot_width_slider.draw(screen, font)
            robot_length_slider.draw(screen, font)
            track_thickness_slider.draw(screen, font)
            sim_speed_slider.draw(screen, font)  # Draw slow-motion slider

            # Draw restart button
            pygame.draw.rect(screen, GRAY, button_rect)
            pygame.draw.rect(screen, BLACK, button_rect, 2)
            restart_text = font.render("New Track", True, BLACK)
            screen.blit(
                restart_text,
                (
                    button_rect.centerx - restart_text.get_width() // 2,
                    button_rect.centery - restart_text.get_height() // 2,
                ),
            )

            # Draw status information
            status_text = font.render(
                f"Left Sensor: {'ON' if left_on_line else 'OFF'} | "
                f"Right Sensor: {'ON' if right_on_line else 'OFF'} | "
                f"{'STOPPED' if robot.is_stopped else 'RUNNING'} | "
                f"Sim Speed: {sim_speed_slider.value:.2f}x | "
                f"Press SPACE to pause, R to generate new track",
                True,
                BLACK,
            )
            screen.blit(status_text, (10, SCREEN_HEIGHT + 10))

            # Calculate FPS more accurately
            frame_time = time.time() - frame_start_time
            if frame_time > 0:
                current_fps = 1.0 / frame_time
                fps_history.append(current_fps)
                if len(fps_history) > 0:
                    fps = sum(fps_history) / len(fps_history)

            # Draw FPS
            fps_text = font.render(f"FPS: {fps:.1f}", True, BLACK)
            screen.blit(fps_text, (SCREEN_WIDTH - 100, SCREEN_HEIGHT + 150))

            # Update the display
            pygame.display.flip()

            # Send data to matplotlib window at a controlled rate
            # This is a major optimization to avoid slowing down the main loop
            current_time = time.time()
            if current_time - last_data_send_time >= plot_update_delay:
                try:
                    if not data_queue.full():
                        robot_data = robot.get_data_for_plotting()
                        data_queue.put(robot_data, block=False)
                        last_data_send_time = current_time
                except:
                    pass

            # Only sleep if we're running fast enough and not in slow motion mode
            if sim_speed_slider.value >= 1.0:
                sleep_time = max(0, 1/60 - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # In slow motion, we don't need to sleep as we're already running slower
                pass
                
            # Count frames
            frame_count += 1

    finally:
        # Quit pygame
        pygame.quit()

        # Make sure Tkinter is properly closed
        try:
            if not matplotlib_window.is_closed:
                matplotlib_window.on_closing()
        except:
            pass

        sys.exit()


if __name__ == "__main__":
    main()
