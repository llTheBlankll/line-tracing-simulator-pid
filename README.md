# Line Tracing Robot Simulation

A Pygame-based simulation of a line-following robot with PID control.

## Features

- Line-following robot with adjustable PID parameters
- Real-time visualization of sensor readings and motor speeds
- Dynamic track generation using Bezier curves
- Matplotlib visualization of performance metrics
- Slow-motion capability to observe robot behavior
- Adjustable robot parameters (size, sensor position, etc.)
- Multithreaded physics calculations for improved performance

## Directory Structure

The codebase is organized into the following modules:

```
├── src/
│   ├── robot/          # Robot class and physics simulation
│   ├── track/          # Track generation and line detection
│   ├── ui/             # UI components and visualization
│   ├── utils/          # Constants and utility functions
│   └── main.py         # Main simulation loop
├── main.py             # Entry point that imports from src
├── run_simulation.py   # Alternative launcher
└── requirements.txt    # Dependencies
```

## Getting Started

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the simulation:

```bash
python main.py
```

Or alternatively:

```bash
python run_simulation.py
```

## Controls

- **Space**: Pause/resume simulation
- **R**: Generate new track
- **Escape**: Exit simulation
- **New Track button**: Generate new track
- **Trail ON/OFF button**: Toggle robot trail display for better performance
- **Sliders**: Adjust various parameters:
  - Kp, Ki, Kd: PID control parameters
  - Base Speed: Robot's base movement speed
  - Friction: How quickly the robot slows down
  - Inertia: How quickly the robot responds to speed changes
  - Sensor Width: Distance between the left and right sensors
  - Sensor Distance: How far the sensors are from the robot's center
  - Robot Width/Length: Robot's dimensions
  - Track Thickness: Width of the track line
  - Sim Speed: Simulation speed (0.1x-2.0x)

## Visualization

The simulation opens a separate Matplotlib window showing:
- Error values
- Motor speeds
- PID components (P, I, D)
- Motor speed differences
- Robot position tracking

This helps understand the robot's behavior and tune PID parameters effectively. 