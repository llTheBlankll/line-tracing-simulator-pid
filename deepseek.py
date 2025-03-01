import pygame
import math
import sys

# PID Constants (from original code)
KP = 1.7
KD = 0.5
BASE_SPEED = 80

# Simulation Parameters
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
ROBOT_SIZE = 20
SENSOR_OFFSET = 25  # Distance from center to sensors
TRACK_WIDTH = 60  # Visual width of the track
SPEED_FACTOR = 0.5  # Overall simulation speed


class LineTrackingSim:
    def __init__(self):
        # Track parameters
        self.track_amplitude = 200
        self.track_frequency = 0.02

        # Robot state
        self.x = SCREEN_WIDTH // 2
        self.y = 50
        self.error = 1  # 0=left, 1=forward, 2=right
        self.last_error = 0
        self.motor_left = BASE_SPEED
        self.motor_right = BASE_SPEED

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)

    def get_track_center(self):
        """Calculate track center position at current Y coordinate"""
        return SCREEN_WIDTH // 2 + self.track_amplitude * math.sin(self.track_frequency * self.y)

    def update_sensors(self):
        """Check if sensors are over the track"""
        track_center = self.get_track_center()
        left_sensor = self.x - SENSOR_OFFSET
        right_sensor = self.x + SENSOR_OFFSET

        in_left = abs(left_sensor - track_center) < TRACK_WIDTH // 2
        in_right = abs(right_sensor - track_center) < TRACK_WIDTH // 2

        return in_left, in_right

    def update_control(self):
        """Update motor speeds using PD control"""
        motor_speed = KP * self.error + KD * (self.error - self.last_error)
        self.motor_left = BASE_SPEED + motor_speed
        self.motor_right = BASE_SPEED - motor_speed

        # Clamp motor speeds (0-255)
        self.motor_left = max(0, min(255, self.motor_left))
        self.motor_right = max(0, min(255, self.motor_right))

    def update_position(self):
        """Update robot position based on motor speeds"""
        # Vertical movement (constant forward motion)
        self.y += 2 * SPEED_FACTOR

        # Horizontal movement based on motor difference
        delta_x = (self.motor_left - self.motor_right) * 0.1 * SPEED_FACTOR
        self.x += delta_x

        # Keep robot within screen bounds
        self.x = max(ROBOT_SIZE, min(SCREEN_WIDTH - ROBOT_SIZE, self.x))

    def draw(self):
        """Draw everything on the screen"""
        self.screen.fill((255, 255, 255))  # White background

        # Draw track
        track_center = self.get_track_center()
        pygame.draw.rect(self.screen, (0, 0, 0),  # Black track
                         (track_center - TRACK_WIDTH // 2, self.y - 2,
                          TRACK_WIDTH, 4))

        # Draw robot
        pygame.draw.circle(self.screen, (255, 0, 0),  # Red robot
                           (int(self.x), int(self.y)), ROBOT_SIZE)

        # Draw sensors
        left_sensor_pos = (int(self.x - SENSOR_OFFSET), int(self.y))
        right_sensor_pos = (int(self.x + SENSOR_OFFSET), int(self.y))
        pygame.draw.circle(self.screen, (0, 0, 255), left_sensor_pos, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), right_sensor_pos, 5)

        # Display status
        status_text = [
            f"Error State: {['LEFT', 'FORWARD', 'RIGHT'][self.error]}",
            f"Motors: L={self.motor_left:.1f} R={self.motor_right:.1f}",
            f"Position: X={self.x:.1f} Y={self.y:.1f}"
        ]
        y_offset = 10
        for text in status_text:
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 30

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Update sensors and error state
            left, right = self.update_sensors()
            self.last_error = self.error
            if left and right:
                self.error = 1
            elif left:
                self.error = 0
            elif right:
                self.error = 2

            # Update control system and position
            self.update_control()
            self.update_position()

            # Draw everything
            self.draw()
            self.clock.tick(60)  # 60 FPS


if __name__ == "__main__":
    simulator = LineTrackingSim()
    simulator.run()