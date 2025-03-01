import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class LineFollowerSimulation:
    def __init__(self):
        # PID Constants (from the original code)
        self.KP = 1.7
        self.KI = 0.05
        self.KD = 0.5
        self.BASE_SPEED = 80

        # Error states: 0 - LEFT, 1 - FORWARD, 2 - RIGHT
        self.error = 1
        self.last_error = 0
        self.integral = 0

        # Simulation parameters
        self.time = 0
        self.time_step = 0.1  # seconds
        self.simulation_length = 10  # seconds
        self.num_steps = int(self.simulation_length / self.time_step)

        # Track data
        self.track_data = []
        self.line_positions = []
        self.robot_positions = []
        self.errors = []
        self.left_motor_speeds = []
        self.right_motor_speeds = []
        self.direction_labels = []

        # Create track with a line to follow (0 = left of line, 1 = on line, 2 = right of line)
        self.generate_track()

        # Initial robot position (index in the track)
        self.robot_position = 0

    def generate_track(self):
        # Generate a simple track with some curves
        # We'll simulate the line position as a sin wave
        x = np.linspace(0, 4 * np.pi, self.num_steps)
        line_position = np.sin(x) * 0.5

        # Add some straight sections
        line_position[:20] = 0
        line_position[-20:] = 0

        self.line_positions = line_position

    def read_sensors(self, position_idx):
        # Simulate reading from left and right line sensors
        # Returns (LEFT, RIGHT) boolean values

        # Get the current line position
        current_position = self.line_positions[position_idx]

        # Determine sensor readings based on position
        if abs(current_position) < 0.1:  # Robot is centered over the line
            return (True, True)  # Both sensors detect the line
        elif current_position < 0:  # Line is to the left
            return (True, False)  # Only left sensor detects the line
        else:  # Line is to the right
            return (False, True)  # Only right sensor detects the line

    def update_error(self, position_idx):
        # Read sensors
        LEFT, RIGHT = self.read_sensors(position_idx)

        # Determine error state based on sensor readings
        if LEFT and RIGHT:
            self.error = 1  # FORWARD
        elif LEFT:
            self.error = 0  # LEFT
        elif RIGHT:
            self.error = 2  # RIGHT

        return self.error

    def calculate_pid(self):
        # Calculate PID control values
        # P term
        p_term = self.KP * (self.error - 1)  # Adjust error to be -1, 0, 1

        # I term (accumulating error)
        self.integral += self.error - 1
        i_term = self.KI * self.integral

        # D term (rate of change)
        d_term = self.KD * ((self.error - 1) - (self.last_error - 1))

        # Calculate motor adjustment
        adjustment = p_term + i_term + d_term

        # Calculate motor speeds
        left_speed = max(0, min(255, self.BASE_SPEED - adjustment))
        right_speed = max(0, min(255, self.BASE_SPEED + adjustment))

        # Update last error
        self.last_error = self.error

        return left_speed, right_speed

    def determine_direction(self, left_speed, right_speed):
        if left_speed > right_speed:
            return "RIGHT"
        elif right_speed > left_speed:
            return "LEFT"
        else:
            return "FORWARD"

    def run_simulation(self):
        # Run the simulation
        robot_position = 0

        for i in range(self.num_steps):
            # Update time
            self.time = i * self.time_step

            # Store the current line position
            current_line_pos = self.line_positions[i]
            self.track_data.append(current_line_pos)

            # Update error based on sensors
            error = self.update_error(i)
            self.errors.append(error - 1)  # Store normalized error (-1, 0, 1)

            # Calculate PID and motor speeds
            left_speed, right_speed = self.calculate_pid()
            self.left_motor_speeds.append(left_speed)
            self.right_motor_speeds.append(right_speed)

            # Determine robot's movement direction
            direction = self.determine_direction(left_speed, right_speed)
            self.direction_labels.append(direction)

            # Update robot position (following the line with some error)
            # Simple model: the robot follows the line with a delay based on PID response
            if i > 0:
                # Robot position approaches the line position based on motor speeds
                speed_diff = right_speed - left_speed
                robot_position = 0.7 * robot_position + 0.3 * (current_line_pos - 0.05 * speed_diff)

            self.robot_positions.append(robot_position)

        return {
            'time': np.arange(0, self.simulation_length, self.time_step),
            'line_positions': self.line_positions,
            'robot_positions': self.robot_positions,
            'errors': self.errors,
            'left_motor_speeds': self.left_motor_speeds,
            'right_motor_speeds': self.right_motor_speeds,
            'direction_labels': self.direction_labels
        }

    def visualize_results(self, results):
        # Create a figure with multiple subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

        time_data = results['time']

        # Plot 1: Line position and robot position
        ax1.plot(time_data, results['line_positions'], 'b-', label='Line Position')
        ax1.plot(time_data, results['robot_positions'], 'r-', label='Robot Position')
        ax1.set_ylabel('Position')
        ax1.set_title('Line and Robot Positions')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Error
        ax2.plot(time_data, results['errors'], 'g-')
        ax2.set_ylabel('Error (-1=LEFT, 0=CENTER, 1=RIGHT)')
        ax2.set_title('Error Signal')
        ax2.grid(True)

        # Plot 3: Motor speeds
        ax3.plot(time_data, results['left_motor_speeds'], 'b-', label='Left Motor')
        ax3.plot(time_data, results['right_motor_speeds'], 'r-', label='Right Motor')
        ax3.set_ylabel('Motor Speed')
        ax3.set_title('Motor Speeds')
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Direction labels
        directions = np.array([d == "LEFT" for d in results['direction_labels']], dtype=int) * (-1) + \
                     np.array([d == "RIGHT" for d in results['direction_labels']], dtype=int) * 1
        ax4.plot(time_data, directions, 'k-')
        ax4.set_ylabel('Direction')
        ax4.set_title('Robot Direction (-1=LEFT, 0=FORWARD, 1=RIGHT)')
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['LEFT', 'FORWARD', 'RIGHT'])
        ax4.grid(True)

        ax4.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

        # Create an animation of the robot following the line
        self.create_animation(results)

    def create_animation(self, results):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get data
        time_data = results['time']
        line_positions = results['line_positions']
        robot_positions = results['robot_positions']

        # Plot the entire line path
        ax.plot(time_data, line_positions, 'b-', alpha=0.3, label='Line Path')

        # Create robot marker
        robot, = ax.plot([], [], 'ro', markersize=10, label='Robot')

        # Create text annotation for direction and speeds
        direction_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        # Set limits
        ax.set_xlim(0, self.simulation_length)
        ax.set_ylim(-1.5, 1.5)

        # Add grid and labels
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position')
        ax.set_title('Line Following Robot Simulation')
        ax.legend()

        def init():
            robot.set_data([], [])
            direction_text.set_text('')
            return robot, direction_text

        def update(frame):
            robot.set_data(time_data[frame], robot_positions[frame])

            # Update text annotation
            direction = results['direction_labels'][frame]
            left_speed = results['left_motor_speeds'][frame]
            right_speed = results['right_motor_speeds'][frame]
            error = results['errors'][frame]

            info_text = f"Direction: {direction}\nLeft: {left_speed:.1f}, Right: {right_speed:.1f}\nError: {error}"
            direction_text.set_text(info_text)

            return robot, direction_text

        # Create animation
        ani = FuncAnimation(fig, update, frames=range(0, len(time_data), 2),
                            init_func=init, blit=True, interval=50)

        plt.tight_layout()
        plt.show()


def analyze_pid_performance(simulation_results):
    # Calculate some metrics to analyze PID performance

    # Extract data
    line_positions = np.array(simulation_results['line_positions'])
    robot_positions = np.array(simulation_results['robot_positions'])
    errors = np.array(simulation_results['errors'])

    # Calculate mean absolute error
    mae = np.mean(np.abs(line_positions - robot_positions))

    # Calculate root mean square error
    rmse = np.sqrt(np.mean((line_positions - robot_positions) ** 2))

    # Calculate recovery time (time to get back on track after perturbation)
    # Define "on track" as having an error less than 0.1
    on_track = np.abs(line_positions - robot_positions) < 0.1
    recovery_times = []
    in_recovery = False
    recovery_start = 0

    for i in range(1, len(on_track)):
        if not on_track[i - 1] and not on_track[i]:
            if not in_recovery:
                in_recovery = True
                recovery_start = i
        elif on_track[i] and in_recovery:
            recovery_time = i - recovery_start
            recovery_times.append(recovery_time)
            in_recovery = False

    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0

    # Count direction changes (measure of oscillation)
    direction_changes = 0
    directions = simulation_results['direction_labels']
    for i in range(1, len(directions)):
        if directions[i] != directions[i - 1]:
            direction_changes += 1

    print(f"PID Performance Analysis:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Average Recovery Time: {avg_recovery_time:.2f} steps")
    print(f"Direction Changes: {direction_changes}")

    return {
        'mae': mae,
        'rmse': rmse,
        'avg_recovery_time': avg_recovery_time,
        'direction_changes': direction_changes
    }


def tune_pid_parameters():
    """
    Run simulations with different PID parameters to find better values
    """
    # Define parameter ranges to test
    kp_values = [0.5, 1.0, 1.7, 2.5, 3.0]
    ki_values = [0, 0.05, 0.1, 0.2]
    kd_values = [0, 0.5, 1.0, 1.5]

    best_params = {'kp': 0, 'ki': 0, 'kd': 0}
    best_rmse = float('inf')

    results = []

    for kp in kp_values:
        for ki in ki_values:
            for kd in kd_values:
                # Create simulation with these parameters
                sim = LineFollowerSimulation()
                sim.KP = kp
                sim.KI = ki
                sim.KD = kd

                # Run simulation
                simulation_results = sim.run_simulation()

                # Analyze performance
                metrics = analyze_pid_performance(simulation_results)
                rmse = metrics['rmse']

                results.append({
                    'kp': kp,
                    'ki': ki,
                    'kd': kd,
                    'rmse': rmse,
                    'mae': metrics['mae'],
                    'direction_changes': metrics['direction_changes']
                })

                # Check if this is the best so far
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'kp': kp, 'ki': ki, 'kd': kd}

    # Print best parameters
    print(f"Best PID parameters found:")
    print(f"KP: {best_params['kp']}")
    print(f"KI: {best_params['ki']}")
    print(f"KD: {best_params['kd']}")
    print(f"RMSE: {best_rmse:.4f}")

    # Plot some comparison results
    plot_pid_comparison(results)

    return best_params


def plot_pid_comparison(results):
    # Sort results by RMSE
    sorted_results = sorted(results, key=lambda x: x['rmse'])
    top_5 = sorted_results[:5]

    # Create a comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    index = np.arange(len(top_5))

    # Create bars for top 5 parameter sets
    for i, result in enumerate(top_5):
        label = f"KP={result['kp']}, KI={result['ki']}, KD={result['kd']}"
        ax.bar(index[i], result['rmse'], bar_width, label=label)

    ax.set_xlabel('Parameter Set')
    ax.set_ylabel('RMSE')
    ax.set_title('Top 5 PID Parameter Sets by RMSE')
    ax.set_xticks(index)
    ax.set_xticklabels([f"Set {i + 1}" for i in range(len(top_5))])
    ax.legend()

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Create and run the simulation with default parameters
    sim = LineFollowerSimulation()
    print("Running simulation with original parameters:")
    print(f"KP: {sim.KP}, KI: {sim.KI}, KD: {sim.KD}")

    # Run the simulation
    results = sim.run_simulation()

    # Analyze PID performance
    analyze_pid_performance(results)

    # Visualize the results
    sim.visualize_results(results)

    # Ask if the user wants to run parameter tuning
    user_input = input("Do you want to run PID parameter tuning? (y/n): ")
    if user_input.lower() == 'y':
        best_params = tune_pid_parameters()

        # Create a new simulation with the best parameters
        print("Running simulation with best parameters...")
        sim = LineFollowerSimulation()
        sim.KP = best_params['kp']
        sim.KI = best_params['ki']
        sim.KD = best_params['kd']

        # Run the simulation again
        results = sim.run_simulation()

        # Analyze PID performance
        analyze_pid_performance(results)

        # Visualize the results
        sim.visualize_results(results)