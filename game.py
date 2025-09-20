# game.py

class Game:
    def update(self, pose_data, dt):
        """Update game state based on pose data and time delta."""
        pass
    
    def render(self, frame):
        """Render game visuals on the frame."""
        pass
    
    def handle_input(self, pose_data):
        """Process pose data for game logic."""
        pass
    
    def reset(self):
        """Reset game state."""
        pass
    
    def is_menu(self):
        """Return True if this is a menu, False if a playable game."""
        return False