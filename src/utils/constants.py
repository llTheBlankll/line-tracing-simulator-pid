"""
Global constants for the line following robot simulation.
"""

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
HISTORY_SIZE = 200  # How many history points to keep for plotting
PLOT_UPDATE_INTERVAL = 3  # Update plots every N frames
PHYSICS_SUBSTEPS = 1  # Physics steps per frame
FPS_UPDATE_INTERVAL = 10  # Update FPS display every N frames
TARGET_FPS = 60  # Target frame rate for the simulation

# Enable/disable debug features to improve performance
ENABLE_ROBOT_TRAIL = False  # Set to False for better performance 