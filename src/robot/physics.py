"""
Physics thread for the line following robot simulation.
This class runs physics calculations in a separate thread to improve performance.
"""
import threading
import time
import numpy as np
import queue
from src.utils.constants import (
    TARGET_FPS, BASE_SPEED, KP, KI, KD
)
from src.track.detection import is_on_line, is_out_of_bounds

class PhysicsThread(threading.Thread):
    def __init__(self, robot, track_x, track_y, data_queue):
        threading.Thread.__init__(self, daemon=True)
        self.robot = robot
        self.track_x = np.array(track_x)  # Make a copy to avoid thread issues
        self.track_y = np.array(track_y)  # Make a copy to avoid thread issues
        self.data_queue = data_queue
        self.paused = False
        self.running = True
        self.left_on_line = False
        self.right_on_line = False
        self.dt = 0.05  # Reduced time step for better performance
        self.sim_speed = 1.0
        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.base_speed = BASE_SPEED
        self.lock = threading.Lock()
        self.frame_time = 1.0 / TARGET_FPS
        self.last_data_send_time = time.time()
        self.plot_update_delay = 0.1  # 100ms delay between data sends
        self.out_of_bounds = False
        self.fps = 0  # Store actual FPS for display
        
        # Thread communication queues
        self.parameter_queue = queue.Queue()
        self.track_queue = queue.Queue()
        self.reset_queue = queue.Queue()
        
    def set_parameters(self, kp, ki, kd, base_speed, sim_speed, friction, inertia,
                       sensor_width, sensor_distance, robot_width, robot_length):
        """
        Thread-safe way to update parameters.
        Places parameter updates in a queue to be processed by the physics thread.
        """
        params = {
            'kp': kp, 
            'ki': ki, 
            'kd': kd, 
            'base_speed': base_speed, 
            'sim_speed': sim_speed,
            'friction': friction, 
            'inertia': inertia,
            'sensor_width': sensor_width, 
            'sensor_distance': sensor_distance,
            'robot_width': robot_width, 
            'robot_length': robot_length
        }
        
        if not self.parameter_queue.full():
            self.parameter_queue.put(params, block=False)
            
    def set_track(self, track_x, track_y):
        """
        Thread-safe way to update the track.
        Places track data in a queue to be processed by the physics thread.
        """
        # Make copies of the track data to avoid thread issues
        track_data = {
            'track_x': np.array(track_x),
            'track_y': np.array(track_y)
        }
        
        if not self.track_queue.full():
            self.track_queue.put(track_data, block=False)
            
    def reset_robot(self, x, y, theta):
        """
        Thread-safe way to reset the robot.
        Places reset data in a queue to be processed by the physics thread.
        """
        reset_data = {'x': x, 'y': y, 'theta': theta}
        
        if not self.reset_queue.full():
            self.reset_queue.put(reset_data, block=False)
            
    def set_paused(self, paused):
        """Thread-safe way to pause/unpause the simulation."""
        with self.lock:
            self.paused = paused
        
    def stop(self):
        """Thread-safe way to stop the physics thread."""
        with self.lock:
            self.running = False
        
    def run(self):
        """Main physics loop running in separate thread"""
        last_time = time.time()
        fps_update_time = time.time()
        frame_count = 0
        
        while True:
            # First check if we should exit
            with self.lock:
                if not self.running:
                    break
                current_paused = self.paused
            
            # Skip physics if paused
            if current_paused:
                time.sleep(0.01)  # Avoid CPU spinning
                continue
            
            # Process any queued parameter updates
            while not self.parameter_queue.empty():
                try:
                    params = self.parameter_queue.get(block=False)
                    
                    # Update thread variables
                    self.kp = params['kp']
                    self.ki = params['ki']
                    self.kd = params['kd']
                    self.base_speed = params['base_speed']
                    self.sim_speed = params['sim_speed']
                    
                    # Update robot parameters under lock
                    with self.lock:
                        self.robot.friction = params['friction']
                        self.robot.inertia = params['inertia']
                        self.robot.sensor_width = params['sensor_width']
                        self.robot.sensor_distance = params['sensor_distance']
                        self.robot.width = params['robot_width']
                        self.robot.length = params['robot_length']
                except:
                    pass
            
            # Process any queued track updates
            while not self.track_queue.empty():
                try:
                    track_data = self.track_queue.get(block=False)
                    
                    # Update track data under lock
                    with self.lock:
                        self.track_x = track_data['track_x']
                        self.track_y = track_data['track_y']
                        self.out_of_bounds = False
                except:
                    pass
            
            # Process any queued robot resets
            while not self.reset_queue.empty():
                try:
                    reset_data = self.reset_queue.get(block=False)
                    
                    # Reset robot under lock
                    with self.lock:
                        self.robot.reset(
                            reset_data['x'],
                            reset_data['y'],
                            reset_data['theta']
                        )
                        self.out_of_bounds = False
                except:
                    pass
            
            # Calculate actual elapsed time for accurate physics
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time
            
            # Update frame counter for FPS calculation
            frame_count += 1
            if current_time - fps_update_time >= 0.5:  # Update FPS every 0.5 seconds
                self.fps = frame_count / (current_time - fps_update_time)
                fps_update_time = current_time
                frame_count = 0
            
            # Make a local copy of robot data to minimize lock time
            with self.lock:
                robot_x = self.robot.x
                robot_y = self.robot.y
                robot_theta = self.robot.theta
                sensor_width = self.robot.sensor_width
                sensor_distance = self.robot.sensor_distance
                robot_length = self.robot.length
            
            # Calculate sensor positions (outside the lock)
            sensor_offset = sensor_width / 2
            front_x = robot_x + (robot_length / 2 + sensor_distance) * np.cos(robot_theta)
            front_y = robot_y + (robot_length / 2 + sensor_distance) * np.sin(robot_theta)
            perp_theta = robot_theta + np.pi / 2
            left_x = front_x + sensor_offset * np.cos(perp_theta)
            left_y = front_y + sensor_offset * np.sin(perp_theta)
            right_x = front_x - sensor_offset * np.cos(perp_theta)
            right_y = front_y - sensor_offset * np.sin(perp_theta)
            
            # Make local copies of track data
            local_track_x = np.array(self.track_x)
            local_track_y = np.array(self.track_y)
            
            # Check if sensors are on the line (outside the lock)
            left_on_line = is_on_line(left_x, left_y, local_track_x, local_track_y)
            right_on_line = is_on_line(right_x, right_y, local_track_x, local_track_y)
            
            # Check if robot is out of bounds (outside the lock)
            out_of_bounds = is_out_of_bounds(robot_x, robot_y)
            
            # Update robot state under lock
            with self.lock:
                self.left_on_line = left_on_line
                self.right_on_line = right_on_line
                self.out_of_bounds = out_of_bounds
                
                if not out_of_bounds:
                    # Update PID and move robot
                    self.robot.update_pid(
                        left_on_line,
                        right_on_line,
                        self.kp,
                        self.ki,
                        self.kd,
                        self.base_speed
                    )
                    
                    # Apply time scaling for slow motion
                    adjusted_dt = self.dt * self.sim_speed
                    
                    # Single physics update for better performance
                    self.robot.move(adjusted_dt)
            
            # Send data to plot window at a controlled rate to avoid overhead
            if current_time - self.last_data_send_time >= self.plot_update_delay:
                try:
                    # Get data under lock but minimize lock time
                    with self.lock:
                        robot_data = self.robot.get_data_for_plotting()
                    
                    # Put data in queue outside of lock
                    if not self.data_queue.full():
                        self.data_queue.put(robot_data, block=False)
                    
                    self.last_data_send_time = current_time
                except:
                    pass
            
            # Sleep to avoid eating up CPU
            # Use a shorter sleep time for fast simulation or a longer one for slow motion
            sleep_time = 0.01  # Base sleep time
            
            # Scale sleep time based on simulation speed
            if self.sim_speed < 1.0:
                # For slow motion, sleep less to allow more frequent calculations
                sleep_time = max(0.001, 0.01 * self.sim_speed)
            elif self.sim_speed > 1.0:
                # For fast motion, just use minimum sleep
                sleep_time = 0.001
                
            time.sleep(sleep_time) 