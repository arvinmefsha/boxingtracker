# main_menu.py

import cv2
import numpy as np

from game import Game

class MainMenu(Game):
    def __init__(self):
        self.options = ["Start Boxing Game", "Start Fruit Ninja Game", "Instructions", "Exit"]
        self.selected_option = 0
        self.instructions_mode = False

    def is_menu(self):
        return True

    def handle_input(self, key):
        """Handle keyboard input for menu navigation."""
        if self.instructions_mode:
            if key == ord('q') or key == 13:  # 'q' or Enter to exit instructions
                self.instructions_mode = False
            return
        
        if key == ord('w') or key == 82:  # Up arrow or 'w'
            self.selected_option = (self.selected_option - 1) % len(self.options)
        elif key == ord('s') or key == 84:  # Down arrow or 's'
            self.selected_option = (self.selected_option + 1) % len(self.options)
        elif key == 13:  # Enter key
            return self.options[self.selected_option]

    def render(self, frame):
        """Render the main menu or instructions."""
        # Resize frame to 1920x1080
        frame = cv2.resize(frame, (1920, 1080))
        # Dark background
        cv2.rectangle(frame, (0, 0), (1920, 1080), (20, 20, 20), -1)

        if self.instructions_mode:
            # Display instructions (updated for both games)
            instructions = [
                "Game Instructions:",
                "Boxing Game:",
                " - Two players on split screen.",
                " - Punch, kick, and knee with right side.",
                "Fruit Ninja Game:",
                " - Slice fruits with hand swipes (fast movements).",
                " - Avoid bombs or lose points.",
                " - Hands act as swords.",
                "Press 'q' or Enter to return to menu."
            ]
            for i, line in enumerate(instructions):
                cv2.putText(frame, line, (100, 100 + i * 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Display menu options
            cv2.putText(frame, "Body Game Platform", (600, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            for i, option in enumerate(self.options):
                color = (0, 255, 0) if i == self.selected_option else (255, 255, 255)
                cv2.putText(frame, option, (700, 400 + i * 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Use W/S or Up/Down arrows to navigate, Enter to select", 
                        (400, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

        return frame