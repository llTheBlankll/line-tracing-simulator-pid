"""
Track generation functions for the line following robot simulation.
"""
import numpy as np
import pygame
from src.utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SCALE, BLACK, WHITE
)

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


def draw_track(screen, track_x, track_y, scale=1.0, thickness=6):
    """
    Optimized track drawing with better performance.
    """
    # Use a larger stride for track rendering to improve performance
    stride = 5  # Increased from 3 to 5
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