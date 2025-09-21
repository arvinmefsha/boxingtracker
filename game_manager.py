# game_manager.py

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

from main_menu import MainMenu
from fruit_ninja_game import FruitNinjaGame
# from boxing_game import BoxingGame # etc.

class GameManager:
    def __init__(self):
        # ... (video capture setup is the same) ...
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ## STABILITY FIX: Tell MediaPipe to look for up to 2 hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Increased from 1 to 2
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # -- Game State Setup --
        self.current_game = MainMenu()
        
        # -- Gesture Control State --
        self.pinch_was_active_last_frame = False
        self.pinch_threshold = 0.07
        ## STABILITY FIX: Add a variable to store the position of the locked-on hand
        self.locked_hand_pos = None

    def get_landmark_distance(self, landmark1, landmark2):
        return np.sqrt((landmark2.x - landmark1.x)**2 + (landmark2.y - landmark1.y)**2)

    def process_frame_for_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_data = {'found': False, 'cursor_pos': None, 'click': False, 'landmarks': None}

        if results.multi_hand_landmarks:
            tracked_hand = None

            ## STABILITY FIX: Logic to lock onto the closest hand
            if self.locked_hand_pos is None:
                # If no hand is locked, lock onto the first one found
                tracked_hand = results.multi_hand_landmarks[0]
            else:
                # Find the hand closest to our last known position
                min_dist = float('inf')
                for hand_landmarks in results.multi_hand_landmarks:
                    # Use the wrist landmark to find the hand's center
                    wrist_pos = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    dist = np.sqrt((wrist_pos.x - self.locked_hand_pos[0])**2 + (wrist_pos.y - self.locked_hand_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        tracked_hand = hand_landmarks

            if tracked_hand:
                hand_data['found'] = True
                hand_data['landmarks'] = tracked_hand.landmark
                
                # Update the locked position to the wrist of the tracked hand
                wrist = tracked_hand.landmark[self.mp_hands.HandLandmark.WRIST]
                self.locked_hand_pos = (wrist.x, wrist.y)

                # --- Cursor Position (from Ring Finger MCP) ---
                ring_finger_mcp = tracked_hand.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
                cursor_x = int(ring_finger_mcp.x * frame.shape[1])
                cursor_y = int(ring_finger_mcp.y * frame.shape[0])
                hand_data['cursor_pos'] = (cursor_x, cursor_y)

                # --- Click Detection (Pinch Gesture) ---
                index_tip = tracked_hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = tracked_hand.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                is_pinching_now = self.get_landmark_distance(index_tip, thumb_tip) < self.pinch_threshold
                if is_pinching_now and not self.pinch_was_active_last_frame:
                    hand_data['click'] = True
                self.pinch_was_active_last_frame = is_pinching_now
        else:
            # If no hands are found, reset the lock and pinch state
            self.pinch_was_active_last_frame = False
            self.locked_hand_pos = None # Lose the lock
            
        return hand_data

    # ... (The run method is unchanged) ...
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            hand_data = self.process_frame_for_hands(frame)
            self.current_game.hand_data = hand_data
            result = self.current_game.render(frame)
            if isinstance(result, str):
                selected_option = result
                if selected_option == "Start Fruit Ninja Game":
                    self.current_game = FruitNinjaGame()
                elif selected_option == "Exit":
                    break
            else: frame = result
            cv2.imshow('Body Game Platform', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m') and not isinstance(self.current_game, MainMenu):
                self.current_game = MainMenu()
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()