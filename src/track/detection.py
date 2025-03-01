"""
Line detection functions for the line following robot simulation.
"""
import numpy as np
from src.utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT, SCALE

# Performance optimization: Cache for line segment calculations in is_on_line function
segment_cache = {}

def is_on_line(sensor_x, sensor_y, track_x, track_y, threshold=3.5):
    """
    Highly optimized version of the line detection function.
    This version uses vectorized operations and more efficient calculations.
    """
    global segment_cache
    
    # Use a larger stride for initial check (optimization)
    stride = 12  # Increased from 8 to 12 for better performance
    
    # Only check a subset of track points to find the closest ones
    if len(track_x) > 100:
        # For long tracks, use a larger stride
        check_indices = np.arange(0, len(track_x), stride)
        track_x_subset = track_x[check_indices]
        track_y_subset = track_y[check_indices]
        
        # Vectorized distance calculation
        dx = track_x_subset - sensor_x
        dy = track_y_subset - sensor_y
        distances = np.sqrt(dx*dx + dy*dy)
        
        # If no points are close, return early
        if np.min(distances) > threshold * 4:
            return False
            
        # Find the closest point
        min_idx = check_indices[np.argmin(distances)]
    else:
        # For short tracks, check all points
        dx = track_x - sensor_x
        dy = track_y - sensor_y
        distances = np.sqrt(dx*dx + dy*dy)
        min_idx = np.argmin(distances)
    
    # Define a smaller window around the closest point
    window_size = 12  # Reduced from 16 to 12
    start_idx = max(0, min_idx - window_size)
    end_idx = min(len(track_x), min_idx + window_size)
    
    # Check if any point in the window is close enough
    for i in range(start_idx, end_idx):
        dist = np.sqrt((track_x[i] - sensor_x) ** 2 + (track_y[i] - sensor_y) ** 2)
        if dist < threshold:
            return True
    
    # Check a few line segments, using cache when possible
    # Only check every other segment to save time
    for i in range(start_idx, end_idx - 1, 2):
        # Create a unique key for this segment
        segment_key = (i, i+1)
        
        # Calculate or retrieve cached values
        if segment_key not in segment_cache:
            x1, y1 = track_x[i], track_y[i]
            x2, y2 = track_x[i+1], track_y[i+1]
            
            # Precompute values for line segment distance calculations
            dx = x2 - x1
            dy = y2 - y1
            line_length_sq = dx**2 + dy**2
            
            # Store these values in the cache
            if line_length_sq > 0:  # Avoid division by zero
                segment_cache[segment_key] = (x1, y1, x2, y2, dx, dy, line_length_sq)
        
        # If segment is in cache, use the cached values
        if segment_key in segment_cache:
            x1, y1, x2, y2, dx, dy, line_length_sq = segment_cache[segment_key]
            
            # Fast distance calculation
            # Project point onto line segment
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
    """
    Check if the robot is out of the screen bounds.
    """
    # Check if the robot is out of the screen bounds
    margin = 50
    return (
        x < -margin
        or x > SCREEN_WIDTH / SCALE + margin
        or y < -margin
        or y > SCREEN_HEIGHT / SCALE + margin
    ) 