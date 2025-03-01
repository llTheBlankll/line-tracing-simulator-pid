"""
UI components for the line following robot simulation.
"""
import pygame
from src.utils.constants import BLACK, GRAY

# Hardcoded pygame event types
# These are the standard values in pygame, but we're using them directly to avoid linter errors
MOUSE_BUTTON_DOWN = 5  # pygame.MOUSEBUTTONDOWN
MOUSE_BUTTON_UP = 6    # pygame.MOUSEBUTTONUP  
MOUSE_MOTION = 4       # pygame.MOUSEMOTION

class PIDSlider:
    """
    A slider component for adjusting numerical values.
    Used for PID constants and other simulation parameters.
    """
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.handle_radius = 10
        self.handle_pos = self._value_to_pos(initial_val)

    def _value_to_pos(self, value):
        # Convert value to position
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + ratio * self.rect.width)

    def _pos_to_value(self, pos):
        # Convert position to value
        ratio = max(0, min(1, (pos - self.rect.x) / self.rect.width))
        return self.min_val + ratio * (self.max_val - self.min_val)

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