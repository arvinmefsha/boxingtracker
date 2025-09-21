# main_menu.py

import cv2
import time
import os
import numpy as np
import sys

from game import Game
from helpers import overlay_transparent # You will need this helper function

class MainMenu(Game):
    def __init__(self):
        ## MODIFIED: Defined games with file paths first for easier loading and checking.
        self.games = [
            {
                "name": "Boxing",
                "thumbnail_path": os.path.join('assets', 'boxing_thumbnail.png'),
                "action": "Start Boxing Game"
            },
            {
                "name": "Fruit Ninja",
                "thumbnail_path": os.path.join('assets', 'fruit_ninja_thumbnail.png'),
                "action": "Start Fruit Ninja Game"
            },
            {
                "name": "Reaction Time",
                "thumbnail_path": os.path.join('assets', 'reaction_thumbnail.png'),
                "action": "Start Reaction Time Game"
            },
        ]
        
        # -- State and Assets --
        self.selected_game_index = 0
        
        # Load button and arrow images
        self.arrow_left = cv2.imread(os.path.join('assets', 'left_arrow.png'), cv2.IMREAD_UNCHANGED)
        self.arrow_right = cv2.imread(os.path.join('assets', 'right_arrow.png'), cv2.IMREAD_UNCHANGED)
        self.play_button = cv2.imread(os.path.join('assets', 'play_button.png'), cv2.IMREAD_UNCHANGED)

        # Safety check for loaded images
        if self.arrow_left is None or self.arrow_right is None or self.play_button is None:
            print("\n--- ERROR: Could not load menu assets (arrows or play button). Check asset folder. ---\n")
            sys.exit(1)
        
        # Define sizes
        new_button_size = (250, 100)
        new_arrow_size = (150, 150)
        thumbnail_size = (1000, 600)
        cursor_size = (60, 60)
        
        # Load, check, and resize all assets
        self.cursor_image = self.load_and_resize_asset(os.path.join('assets', 'hand.png'), cursor_size)
        for game in self.games:
            if game["thumbnail_path"]:
                game["thumbnail"] = self.load_and_resize_asset(game["thumbnail_path"], thumbnail_size)
            else:
                game["thumbnail"] = None

        self.play_button = cv2.resize(self.play_button, new_button_size, interpolation=cv2.INTER_AREA)
        self.arrow_right = cv2.resize(self.arrow_right, new_arrow_size, interpolation=cv2.INTER_AREA)
        self.arrow_left = cv2.resize(self.arrow_left, new_arrow_size, interpolation=cv2.INTER_AREA)

        # Create "disabled" versions of the arrows
        self.arrow_left_disabled = self.arrow_left.copy()
        self.arrow_left_disabled[:, :, 3] = self.arrow_left_disabled[:, :, 3] * 0.3
        self.arrow_right_disabled = self.arrow_right.copy()
        self.arrow_right_disabled[:, :, 3] = self.arrow_right_disabled[:, :, 3] * 0.3

        # -- Interaction --
        ## MODIFIED: Reduced hover time for a shorter loading bar
        self.hover_time = 0.7
        self.hover_timers = {'left': None, 'right': None, 'play': None}
        
        self.button_regions = {
            # Left Arrow is 150x150, centered at (360, 500)
            'left':  (360 - 75, 500 - 75, 360 + 75, 500 + 75),   # (285, 425, 435, 575)
            
            # Right Arrow is 150x150, centered at (1550, 500)
            'right': (1550 - 75, 500 - 75, 1550 + 75, 500 + 75), # (1475, 425, 1625, 575)
            
            # Play Button is 250x100, centered at (960, 980)
            'play':  (960 - 125, 980 - 50, 960 + 125, 980 + 50)  # (835, 930, 1085, 1030)
        }

        self.cursor_pos = (0, 0)
        self.hand_data = {}

    def load_and_resize_asset(self, path, size):
        """Helper function to load, check, and resize an image."""
        if not os.path.exists(path):
            print(f"\n--- FATAL ERROR: Asset file not found at path: {path} ---")
            sys.exit(1)
            
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"\n--- FATAL ERROR: Failed to load asset: {os.path.basename(path)} ---")
            print("The file might be corrupt or in an unsupported format.")
            sys.exit(1)
        
        if image.shape[2] < 4:
            print(f"\n--- FATAL ERROR: Asset '{os.path.basename(path)}' is not a transparent PNG. ---")
            print("Please re-export the image with a transparency (alpha) channel.")
            sys.exit(1)
            
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def reset(self):
        self.selected_game_index = 0
        self.hover_timers = {'left': None, 'right': None, 'play': None}

    def is_menu(self):
        return True

    def handle_input(self, hand_data):
        if hand_data['found']:
            self.cursor_pos = hand_data['cursor_pos']
            for name, (x1, y1, x2, y2) in self.button_regions.items():
                if x1 < self.cursor_pos[0] < x2 and y1 < self.cursor_pos[1] < y2:
                    ## MODIFIED: Check if the button is disabled before starting the timer
                    is_left_disabled = (name == 'left' and self.selected_game_index == 0)
                    is_right_disabled = (name == 'right' and self.selected_game_index == len(self.games) - 1)
                    
                    if is_left_disabled or is_right_disabled:
                        self.hover_timers[name] = None # Ensure timer is off for disabled buttons
                        continue # Skip to the next button

                    # If button is not disabled, proceed with hover logic
                    if self.hover_timers[name] is None:
                        self.hover_timers[name] = time.time()
                    elif time.time() - self.hover_timers[name] > self.hover_time:
                        return self.perform_action(name)
                else:
                    self.hover_timers[name] = None
        else:
            self.reset_timers()
        return None

    def perform_action(self, button_name):
        if button_name == 'left' and self.selected_game_index > 0:
            self.selected_game_index -= 1
        elif button_name == 'right' and self.selected_game_index < len(self.games) - 1:
            self.selected_game_index += 1
        elif button_name == 'play':
            self.reset_timers()
            return self.games[self.selected_game_index]["action"]
        self.reset_timers()
        return None
        
    def reset_timers(self):
        self.hover_timers = {'left': None, 'right': None, 'play': None}

    def render(self, frame):
        action_result = self.handle_input(self.hand_data)
        if action_result:
            return action_result

        frame[:] = (25, 25, 25) # Dark background

        # --- Draw all UI elements with corrected positions ---

        selected_item = self.games[self.selected_game_index]
        
        # Draw Title
        title_text = "Choose game"
        text_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, title_text, (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3, cv2.LINE_AA)

        # Draw Thumbnail or Instructions
        if selected_item["name"] == "Instructions":
            instructions = ["Game Instructions:", " ", "Boxing Game:", " - Punch, kick, and knee with your right side.", " ", "Fruit Ninja Game:", " - Slice fruits with fast hand swipes.", " ", "Reaction Time Game:", " - Wait for green bar, then raise hands fastest.", " "]
            for i, line in enumerate(instructions):
                cv2.putText(frame, line, (400, 300 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            if selected_item["thumbnail"] is not None:
                frame = overlay_transparent(frame, selected_item["thumbnail"], 960, 500)
        
        # Draw Item Name
        item_name = selected_item["name"]
        text_size, _ = cv2.getTextSize(item_name, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, item_name, (text_x, 900), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4, cv2.LINE_AA)
        
        # Draw Arrows (active or disabled)
        left_arrow_img = self.arrow_left if self.selected_game_index > 0 else self.arrow_left_disabled
        right_arrow_img = self.arrow_right if self.selected_game_index < len(self.games) - 1 else self.arrow_right_disabled
        if self.selected_game_index > 0:
            frame = overlay_transparent(frame, left_arrow_img, 360, 500)
        if self.selected_game_index < len(self.games) - 1:
            frame = overlay_transparent(frame, right_arrow_img, 1550, 500)

        # Draw Play Button
        frame = overlay_transparent(frame, self.play_button, 960, 980)
        
        # Draw Hover Progress Bars
        for name, start_time in self.hover_timers.items():
            if start_time is not None:
                # ... (progress calculation is the same) ...
                progress = min((time.time() - start_time) / self.hover_time, 1.0)
                (x1, y1, x2, y2) = self.button_regions[name]
                progress_y = y2 + 5
                bar_height = 5

                ## MODIFIED: Add a margin to control the bar's length
                # A larger margin makes the bar shorter.
                bar_margin = 10 # Makes the bar 20 pixels shorter on each side
                bar_yoff = 0

                # Calculate the new start and end points for the bar's length
                bar_x1 = x1 + bar_margin
                bar_x2 = x2 - bar_margin
                
                # Draw background bar using the new x-coordinates
                cv2.rectangle(frame, (bar_x1, progress_y+bar_yoff), (bar_x2, progress_y+bar_yoff + bar_height), (80, 80, 80), -1)
                
                # Draw progress fill using the new x-coordinates and new length
                fill_width = int((bar_x2 - bar_x1) * progress)
                cv2.rectangle(frame, (bar_x1, progress_y+bar_yoff), (bar_x1 + fill_width, progress_y+bar_yoff + bar_height), (0, 255, 0), -1)
        
        # Draw Cursor
        frame = overlay_transparent(frame, self.cursor_image, int(self.cursor_pos[0]), int(self.cursor_pos[1]))

        return frame