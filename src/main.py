"""
Main entry point for the line following robot simulation.
This module initializes the simulation and contains the main game loop.
"""
import sys
import os
import time
import queue
from collections import deque
import pygame
import numpy as np

# Import our modules
from src.utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, STATS_HEIGHT, WINDOW_HEIGHT, SCALE,
    BASE_SPEED, KP, KI, KD, 
    BLACK, WHITE, GRAY, BLUE, RED, GREEN,
    HISTORY_SIZE, PLOT_UPDATE_INTERVAL, TARGET_FPS,
    ENABLE_ROBOT_TRAIL
)
from src.robot.robot import Robot
from src.robot.physics import PhysicsThread
from src.track.generator import generate_track, draw_track
from src.track.detection import segment_cache, is_out_of_bounds
from src.ui.components import PIDSlider
from src.ui.visualization import MatplotlibWindow

# Constants for event types since pygame.locals import causes linter errors
QUIT = 12  # pygame.QUIT
KEYDOWN = 2  # pygame.KEYDOWN
MOUSEBUTTONDOWN = 5  # pygame.MOUSEBUTTONDOWN
MOUSEBUTTONUP = 6  # pygame.MOUSEBUTTONUP
MOUSEMOTION = 4  # pygame.MOUSEMOTION
K_ESCAPE = 27  # pygame.K_ESCAPE
K_SPACE = 32  # pygame.K_SPACE
K_r = 114  # pygame.K_r

def main():
    """Main function for the line following robot simulation."""
    global ENABLE_ROBOT_TRAIL
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
    
    # Add slow motion slider - adjust range to be more precise
    sim_speed_slider = PIDSlider(850, SCREEN_HEIGHT + 150, 200, 20, 0.1, 2.0, 1.0, "Sim Speed")
    
    # Add toggle for robot trail
    trail_toggle_button = pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT + 60, 140, 40)
    robot_trail_enabled = ENABLE_ROBOT_TRAIL
    
    # Create a list of all sliders for easier event handling
    sliders = [
        kp_slider, ki_slider, kd_slider, speed_slider,
        friction_slider, inertia_slider, sensor_width_slider, sensor_dist_slider,
        robot_width_slider, robot_length_slider, track_thickness_slider, sim_speed_slider
    ]
    
    # Create a queue for thread communication
    data_queue = queue.Queue(maxsize=10)  # Reduced queue size to ensure fresh data

    # Create Matplotlib window
    matplotlib_window = MatplotlibWindow(data_queue)
    
    # Create and start physics thread
    physics_thread = PhysicsThread(robot, track_x, track_y, data_queue)
    physics_thread.start()

    # Main game loop variables
    running = True
    paused = False
    frame_count = 0
    current_track_thickness = 6
    last_render_time = time.time()
    render_times = deque(maxlen=30)  # Store recent render times for FPS calculation
    
    # Track rendering optimization
    track_points = []
    for i in range(0, len(track_x), 5):  # Increased stride for better performance
        track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))

    # Initialize drag state for sliders
    drag_active = False
    active_slider = None

    try:
        while running:
            # Measure render frame start time
            frame_start_time = time.time()
            
            # Process events
            for event in pygame.event.get():
                # Global events (quit, keyboard)
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        paused = not paused
                        physics_thread.set_paused(paused)
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
                        physics_thread.set_track(track_x, track_y)
                        # Clear segment cache when track changes
                        segment_cache.clear()
                        
                        # Update pre-rendered track points
                        track_points = []
                        for i in range(0, len(track_x), 5):
                            track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))
                
                # Mouse button down events
                elif event.type == MOUSEBUTTONDOWN:
                    # Check for button click
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
                        physics_thread.set_track(track_x, track_y)
                        # Clear segment cache when track changes
                        segment_cache.clear()
                        
                        # Update pre-rendered track points
                        track_points = []
                        for i in range(0, len(track_x), 5):
                            track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))
                    
                    # Handle trail toggle button
                    elif trail_toggle_button.collidepoint(event.pos):
                        robot_trail_enabled = not robot_trail_enabled
                        ENABLE_ROBOT_TRAIL = robot_trail_enabled
                        
                    # Check sliders
                    for slider in sliders:
                        # Check if the slider handle is clicked
                        handle_rect = pygame.Rect(
                            slider.handle_pos - slider.handle_radius,
                            slider.rect.y - slider.handle_radius,
                            slider.handle_radius * 2,
                            slider.handle_radius * 2
                        )
                        if handle_rect.collidepoint(event.pos):
                            drag_active = True
                            active_slider = slider
                            break
                
                # Mouse button up - release all dragging
                elif event.type == MOUSEBUTTONUP:
                    drag_active = False
                    active_slider = None
                
                # Mouse motion - handle slider dragging
                elif event.type == MOUSEMOTION:
                    if drag_active and active_slider:
                        # Update slider position
                        active_slider.handle_pos = max(
                            active_slider.rect.x, 
                            min(active_slider.rect.x + active_slider.rect.width, event.pos[0])
                        )
                        active_slider.value = active_slider._pos_to_value(active_slider.handle_pos)
                        
                        # Update parameters in physics thread (only when values change)
                        physics_thread.set_parameters(
                            kp_slider.value,
                            ki_slider.value,
                            kd_slider.value,
                            speed_slider.value,
                            sim_speed_slider.value,
                            friction_slider.value,
                            inertia_slider.value,
                            sensor_width_slider.value,
                            sensor_dist_slider.value,
                            robot_width_slider.value,
                            robot_length_slider.value
                        )
                        
                        # Update track thickness
                        current_track_thickness = int(track_thickness_slider.value)

            # Check if robot is out of bounds and generate new track if needed
            if physics_thread.out_of_bounds:
                # Generate new track
                track_x, track_y = generate_track()
                # Reset robot
                physics_thread.reset_robot(
                    track_x[0],
                    track_y[0],
                    np.arctan2(track_y[1] - track_y[0], track_x[1] - track_x[0])
                )
                physics_thread.set_track(track_x, track_y)
                # Clear segment cache
                segment_cache.clear()
                
                # Update pre-rendered track points
                track_points = []
                for i in range(0, len(track_x), 5):
                    track_points.append((int(track_x[i] * SCALE), int(track_y[i] * SCALE)))

            # Clear the screen
            screen.fill(WHITE)

            # Draw the track with current thickness
            draw_track(screen, track_x, track_y, SCALE, current_track_thickness)

            # Draw the robot
            with physics_thread.lock:  # Lock while accessing robot for drawing
                robot.draw(screen, SCALE)

            # Draw dividing line
            pygame.draw.line(
                screen, BLACK, (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT), 2
            )

            # Draw sliders
            for slider in sliders:
                slider.draw(screen, font)
            
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
            
            # Draw trail toggle button
            pygame.draw.rect(screen, GRAY if robot_trail_enabled else (200, 100, 100), trail_toggle_button)
            pygame.draw.rect(screen, BLACK, trail_toggle_button, 2)
            trail_text = font.render("Trail: " + ("ON" if robot_trail_enabled else "OFF"), True, BLACK)
            screen.blit(
                trail_text,
                (
                    trail_toggle_button.centerx - trail_text.get_width() // 2,
                    trail_toggle_button.centery - trail_text.get_height() // 2,
                ),
            )

            # Get sensor status from physics thread
            left_on_line = physics_thread.left_on_line
            right_on_line = physics_thread.right_on_line

            # Draw status information
            status_text = font.render(
                f"Left Sensor: {'ON' if left_on_line else 'OFF'} | "
                f"Right Sensor: {'ON' if right_on_line else 'OFF'} | "
                f"{'PAUSED' if paused else 'RUNNING'} | "
                f"Sim Speed: {sim_speed_slider.value:.2f}x | "
                f"Press SPACE to pause, R to generate new track",
                True,
                BLACK,
            )
            screen.blit(status_text, (10, SCREEN_HEIGHT + 10))

            # Calculate actual render frame rate
            frame_time = time.time() - frame_start_time
            render_times.append(frame_time)
            avg_frame_time = sum(render_times) / len(render_times)
            render_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Get physics FPS from physics thread
            physics_fps = physics_thread.fps

            # Draw FPS
            fps_text = font.render(f"Render FPS: {render_fps:.1f} | Physics FPS: {physics_fps:.1f}", True, BLACK)
            screen.blit(fps_text, (SCREEN_WIDTH - 400, SCREEN_HEIGHT + 150))

            # Update the display
            pygame.display.flip()

            # Check matplotlib window
            if matplotlib_window.is_closed:
                running = False
            else:
                # Process Tkinter events directly (tk will handle its own update timing)
                if frame_count % 5 == 0:  # Reduced frequency for better performance
                    try:
                        matplotlib_window.root.update()
                    except:
                        running = False
            
            # Frame rate control - don't limit too strictly to avoid blocking
            # Just sleep a small amount to allow other threads to run
            elapsed = time.time() - frame_start_time
            if elapsed < 0.016:  # Target ~60 FPS
                time.sleep(0.016 - elapsed)
                
            frame_count += 1

    finally:
        # Stop physics thread
        physics_thread.stop()
        physics_thread.join(timeout=1.0)  # Wait for thread to terminate
        
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