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

# Constants - initial values that will be adjustable via sliders
BASE_SPEED = 40
KP = 1.0
KI = 0.0
KD = 0.005

# Physics constants
FRICTION = 0.5  # Friction coefficient (1.0 = no friction)
INERTIA = 0.5    # How quickly the robot responds to speed changes (1.0 = instant)

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


class Robot:
    def __init__(self, x, y, theta=0):
        self.x = x
        self.y = y
        self.theta = theta  # in radians
        self.width = 10
        self.length = 15
        self.sensor_width = 8  # Distance between left and right sensors
        self.sensor_distance = 5  # Distance of sensors from front of robot

        # PID variables
        self.error = 1  # 0-LEFT, 1-FORWARD, 2-RIGHT
        self.last_error = 0
        self.integral = 0

        # Motor speeds with inertia
        self.target_left_speed = BASE_SPEED
        self.target_right_speed = BASE_SPEED
        self.current_left_speed = 0
        self.current_right_speed = 0
        
        # State
        self.is_stopped = False

        # For visualization
        self.history_x = deque(maxlen=500)
        self.history_y = deque(maxlen=500)
        self.history_x.append(x)
        self.history_y.append(y)
        self.error_history = deque(maxlen=500)
        self.error_history.append(0)
        self.motor_diff_history = deque(maxlen=500)
        self.motor_diff_history.append(0)
        self.left_speed_history = deque(maxlen=500)
        self.left_speed_history.append(0)
        self.right_speed_history = deque(maxlen=500)
        self.right_speed_history.append(0)
        self.proportional_history = deque(maxlen=500)
        self.proportional_history.append(0)
        self.integral_history = deque(maxlen=500)
        self.integral_history.append(0)
        self.derivative_history = deque(maxlen=500)
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
        # If both sensors detect the line, stop the robot
        if left_on_line and right_on_line:
            self.is_stopped = True
            self.target_left_speed = 0
            self.target_right_speed = 0
            self.error = 1  # Center
            return
        else:
            self.is_stopped = False
            
        # Update error based on sensor readings with CORRECTED turning direction
        if left_on_line:
            self.error = 2  # Turn RIGHT when LEFT sensor detects line
        elif right_on_line:
            self.error = 0  # Turn LEFT when RIGHT sensor detects line
        else:
            # If both sensors are off the line, keep the last error
            # This helps the robot to continue in the same direction to find the line again
            pass

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
        self.current_left_speed = (1 - INERTIA) * self.current_left_speed + INERTIA * self.target_left_speed
        self.current_right_speed = (1 - INERTIA) * self.current_right_speed + INERTIA * self.target_right_speed
        
        # Apply friction
        self.current_left_speed *= FRICTION
        self.current_right_speed *= FRICTION

    def move(self, dt):
        # Update physics first
        self.update_physics(dt)
            
        # Calculate robot movement based on motor speeds
        # Forward speed is average of two motors
        forward_speed = (self.current_left_speed + self.current_right_speed) / 2

        # Rotation depends on difference between motors
        rotation_speed = (self.current_right_speed - self.current_left_speed) / self.width

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
        # Draw trail
        points = []
        for i in range(len(self.history_x)):
            points.append((int(self.history_x[i] * scale), int(self.history_y[i] * scale)))
        
        if len(points) > 1:
            pygame.draw.lines(screen, BLUE, False, points, 2)
        
        # Draw robot body
        corners = [
            (-self.width / 2, -self.length / 2),
            (self.width / 2, -self.length / 2),
            (self.width / 2, self.length / 2),
            (-self.width / 2, self.length / 2)
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for corner in corners:
            x, y = corner
            # Rotate
            x_rot = x * np.cos(self.theta) - y * np.sin(self.theta)
            y_rot = x * np.sin(self.theta) + y * np.cos(self.theta)
            # Translate
            x_final = int((self.x + x_rot) * scale)
            y_final = int((self.y + y_rot) * scale)
            rotated_corners.append((x_final, y_final))
        
        pygame.draw.polygon(screen, BLUE, rotated_corners)
        
        # Draw sensors
        (left_x, left_y), (right_x, right_y) = self.get_sensor_positions()
        pygame.draw.circle(screen, RED, (int(left_x * scale), int(left_y * scale)), 3)
        pygame.draw.circle(screen, GREEN, (int(right_x * scale), int(right_y * scale)), 3)

    def get_data_for_plotting(self):
        # Create copies of the data for thread safety
        return {
            'error_history': list(self.error_history),
            'motor_diff_history': list(self.motor_diff_history),
            'left_speed_history': list(self.left_speed_history),
            'right_speed_history': list(self.right_speed_history),
            'proportional_history': list(self.proportional_history),
            'integral_history': list(self.integral_history),
            'derivative_history': list(self.derivative_history)
        }


def generate_track():
    # Create a more varied track with straight lines and curves
    # that connects back to the start
    
    # Define the center of the track
    center_x, center_y = SCREEN_WIDTH / (2 * SCALE), SCREEN_HEIGHT / (2 * SCALE)
    
    # Create a track with various segments
    track_x = []
    track_y = []
    
    # Starting point
    start_x, start_y = center_x - 100, center_y
    
    # Add segments to the track
    
    # Segment 1: Straight line going right
    x = np.linspace(start_x, start_x + 200, 100)
    y = np.ones_like(x) * start_y
    track_x.extend(x)
    track_y.extend(y)
    
    # Segment 2: Curve going up and right
    theta = np.linspace(np.pi, np.pi/2, 100)
    radius = 80
    x = start_x + 200 + radius * np.cos(theta)
    y = start_y + radius * np.sin(theta)
    track_x.extend(x)
    track_y.extend(y)
    
    # Segment 3: Straight line going up
    x = np.ones_like(y) * (start_x + 200 + radius)
    y = np.linspace(start_y + radius, start_y + radius + 100, 100)
    track_x.extend(x)
    track_y.extend(y)
    
    # Segment 4: Curve going left
    theta = np.linspace(np.pi/2, 0, 100)
    x = start_x + 200 + radius * np.cos(theta)
    y = start_y + radius + 100 + radius * np.sin(theta)
    track_x.extend(x)
    track_y.extend(y)
    
    # Segment 5: Straight line going left
    x = np.linspace(start_x + 200, start_x, 100)
    y = np.ones_like(x) * (start_y + radius + 100 + radius)
    track_x.extend(x)
    track_y.extend(y)
    
    # Segment 6: Curve going down
    theta = np.linspace(0, -np.pi/2, 100)
    x = start_x + radius * np.cos(theta)
    y = start_y + radius + 100 + radius + radius * np.sin(theta)
    track_x.extend(x)
    track_y.extend(y)
    
    # Segment 7: Straight line going down
    x = np.ones_like(y) * (start_x)
    y = np.linspace(start_y + radius + 100 + radius, start_y, 100)
    track_x.extend(x)
    track_y.extend(y)
    
    return np.array(track_x), np.array(track_y)


def is_on_line(sensor_x, sensor_y, track_x, track_y, threshold=3):
    # Check if a sensor is on the line
    distances = np.sqrt((track_x - sensor_x) ** 2 + (track_y - sensor_y) ** 2)
    return np.min(distances) < threshold


def is_out_of_bounds(x, y):
    # Check if the robot is out of the screen bounds
    margin = 50
    return (x < -margin or x > SCREEN_WIDTH/SCALE + margin or 
            y < -margin or y > SCREEN_HEIGHT/SCALE + margin)


def draw_track(screen, track_x, track_y, scale=1.0):
    points = []
    for i in range(len(track_x)):
        points.append((int(track_x[i] * scale), int(track_y[i] * scale)))
    
    if len(points) > 1:
        pygame.draw.lines(screen, BLACK, False, points, 3)


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
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.Rect(self.handle_pos - self.handle_radius, 
                          self.rect.y - self.handle_radius,
                          self.handle_radius * 2, 
                          self.handle_radius * 2).collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.handle_pos = max(self.rect.x, min(self.rect.x + self.rect.width, event.pos[0]))
            self.value = self._pos_to_value(self.handle_pos)
    
    def draw(self, screen, font):
        # Draw slider track
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw handle
        pygame.draw.circle(screen, BLACK, (self.handle_pos, self.rect.y + self.rect.height // 2), self.handle_radius)
        
        # Draw label and value
        label_text = font.render(f"{self.label}: {self.value:.3f}", True, BLACK)
        screen.blit(label_text, (self.rect.x, self.rect.y - 20))


class MatplotlibWindow:
    def __init__(self):
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Line Following Robot - Data Visualization")
        self.root.geometry("1000x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
        
        # Set titles
        self.ax1.set_title('Error')
        self.ax2.set_title('Motor Speeds')
        self.ax3.set_title('PID Components')
        self.ax4.set_title('Motor Speed Difference')
        self.ax5.set_title('Position X')
        self.ax6.set_title('Position Y')
        
        # Set grid
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.grid(True)
        
        self.fig.tight_layout()
        self.is_closed = False
        
    def update_plots(self, robot_data):
        if self.is_closed:
            return
            
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()
            ax.grid(True)
        
        # Get data
        error_data = robot_data['error_history']
        motor_diff_data = robot_data['motor_diff_history']
        left_speed = robot_data['left_speed_history']
        right_speed = robot_data['right_speed_history']
        proportional = robot_data['proportional_history']
        integral = robot_data['integral_history']
        derivative = robot_data['derivative_history']
        x_range = range(len(error_data))
        
        # Plot error
        self.ax1.plot(x_range, error_data, 'r-')
        self.ax1.set_title('Error')
        self.ax1.set_ylim(-1.5, 1.5)
        
        # Plot motor speeds
        self.ax2.plot(x_range, left_speed, 'b-', label='Left')
        self.ax2.plot(x_range, right_speed, 'g-', label='Right')
        self.ax2.set_title('Motor Speeds')
        self.ax2.legend()
        
        # Plot PID components
        self.ax3.plot(x_range, proportional, 'r-', label='P')
        self.ax3.plot(x_range, integral, 'g-', label='I')
        self.ax3.plot(x_range, derivative, 'b-', label='D')
        self.ax3.set_title('PID Components')
        self.ax3.legend()
        
        # Plot motor difference
        self.ax4.plot(x_range, motor_diff_data, 'g-')
        self.ax4.set_title('Motor Speed Difference')
        
        # Plot position X
        if len(robot_data.get('history_x', [])) > 0:
            self.ax5.plot(robot_data['history_x'], 'k-')
            self.ax5.set_title('Position X')
        
        # Plot position Y
        if len(robot_data.get('history_y', [])) > 0:
            self.ax6.plot(robot_data['history_y'], 'k-')
            self.ax6.set_title('Position Y')
        
        # Update the figure
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Process Tkinter events
        self.root.update()
    
    def on_closing(self):
        self.is_closed = True
        self.root.destroy()


def main():
    # Initialize pygame
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Line Following Robot Simulation")
    
    # Set up the clock
    clock = pygame.time.Clock()

# Generate track
track_x, track_y = generate_track()

# Initialize robot at the start of the track
robot = Robot(track_x[0], track_y[0])
robot.theta = np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0])

    # Create font for text
    font = pygame.font.SysFont(None, 24)
    
    # Create restart button
    button_rect = pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT + 10, 140, 40)
    
    # Create PID sliders
    kp_slider = PIDSlider(50, SCREEN_HEIGHT + 50, 200, 20, 0, 5, KP, "Kp")
    ki_slider = PIDSlider(50, SCREEN_HEIGHT + 100, 200, 20, 0, 0.1, KI, "Ki")
    kd_slider = PIDSlider(50, SCREEN_HEIGHT + 150, 200, 20, 0, 0.1, KD, "Kd")
    speed_slider = PIDSlider(350, SCREEN_HEIGHT + 50, 200, 20, 10, 100, BASE_SPEED, "Base Speed")

# Time step
dt = 0.1
    
    # Create Matplotlib window in the main thread
    matplotlib_window = MatplotlibWindow()
    
    # Main game loop
    running = True
    paused = False
    left_on_line = False
    right_on_line = False
    
    try:
        while running and not matplotlib_window.is_closed:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # Reset robot
                        robot.reset(track_x[0], track_y[0], 
                                  np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0]))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        # Reset robot when button is clicked
                        robot.reset(track_x[0], track_y[0], 
                                  np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0]))
                
                # Handle slider events
                kp_slider.handle_event(event)
                ki_slider.handle_event(event)
                kd_slider.handle_event(event)
                speed_slider.handle_event(event)
            
            if not paused:
    # Get sensor positions
    (left_x, left_y), (right_x, right_y) = robot.get_sensor_positions()

    # Check if sensors are on the line
    left_on_line = is_on_line(left_x, left_y, track_x, track_y)
    right_on_line = is_on_line(right_x, right_y, track_x, track_y)

                # Update robot based on PID control with slider values
                robot.update_pid(left_on_line, right_on_line, 
                               kp_slider.value, ki_slider.value, kd_slider.value, 
                               speed_slider.value)
    robot.move(dt)

                # Check if robot is out of bounds
                if is_out_of_bounds(robot.x, robot.y):
                    # Reset robot to starting position
                    robot.reset(track_x[0], track_y[0], 
                              np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0]))
            
            # Clear the screen
            screen.fill(WHITE)
            
            # Draw the track
            draw_track(screen, track_x, track_y, SCALE)
            
            # Draw the robot
            robot.draw(screen, SCALE)
            
            # Draw dividing line
            pygame.draw.line(screen, BLACK, (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT), 2)
            
            # Draw sliders
            kp_slider.draw(screen, font)
            ki_slider.draw(screen, font)
            kd_slider.draw(screen, font)
            speed_slider.draw(screen, font)
            
            # Draw restart button
            pygame.draw.rect(screen, GRAY, button_rect)
            pygame.draw.rect(screen, BLACK, button_rect, 2)
            restart_text = font.render("Restart", True, BLACK)
            screen.blit(restart_text, (button_rect.centerx - restart_text.get_width() // 2, 
                                      button_rect.centery - restart_text.get_height() // 2))
            
            # Draw status information
            status_text = font.render(f"Left Sensor: {'ON' if left_on_line else 'OFF'} | "
                                    f"Right Sensor: {'ON' if right_on_line else 'OFF'} | "
                                    f"{'STOPPED' if robot.is_stopped else 'RUNNING'} | "
                                    f"Press SPACE to pause, R to reset", True, BLACK)
            screen.blit(status_text, (10, SCREEN_HEIGHT + 10))
            
            # Draw FPS
            fps = clock.get_fps()
            fps_text = font.render(f"FPS: {fps:.1f}", True, BLACK)
            screen.blit(fps_text, (SCREEN_WIDTH - 100, SCREEN_HEIGHT + 50))
            
            # Update the display
            pygame.display.flip()
            
            # Update matplotlib window
            robot_data = robot.get_data_for_plotting()
            robot_data['history_x'] = list(robot.history_x)
            robot_data['history_y'] = list(robot.history_y)
            matplotlib_window.update_plots(robot_data)
            
            # Cap the frame rate
            clock.tick(60)
    
    finally:
        # Quit pygame
        pygame.quit()
        
        # Make sure Tkinter is properly closed
        try:
            if not matplotlib_window.is_closed:
                matplotlib_window.root.destroy()
        except:
            pass
            
        sys.exit()


if __name__ == "__main__":
    main()