# game_manager.py

import cv2
import mediapipe as mp
import sys
import time

# Import ALL your game state classes
from main_menu import MainMenu
from fruit_ninja_game import FruitNinjaGame
from boxing_game import BoxingGame
from reaction_time_game import ReactionTimeGame ## NEW: Import ReactionTimeGame

class GameManager:
    def __init__(self):
        # -- Video Capture Setup --
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Initialize BOTH MediaPipe models
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # -- Game State Setup --
        self.current_game = MainMenu()
        self.prev_time = time.time()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1920, 1080))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Smartly choose which model and logic to run
            if isinstance(self.current_game, MainMenu):
                # Run the HANDS model for the menu
                results = self.hands.process(frame_rgb)
                hand_data = {'found': False, 'cursor_pos': self.current_game.cursor_pos}
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_data['found'] = True
                    cursor_landmark = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    hand_data['cursor_pos'] = (cursor_landmark.x * frame.shape[1], cursor_landmark.y * frame.shape[0])
                self.current_game.hand_data = hand_data

            ## MODIFIED: Combined logic for all Pose-based games
            elif isinstance(self.current_game, (BoxingGame, ReactionTimeGame)):
                # Run the POSE model, since both games need it
                height, width, _ = frame.shape
                half_width = width // 2
                p1_results = self.pose.process(frame_rgb[:, :half_width])
                p2_results = self.pose.process(frame_rgb[:, half_width:])
                pose_data = {'p1': p1_results, 'p2': p2_results}

                # Calculate delta time (dt)
                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time

                # Call the .update() method that both games expect
                self.current_game.update(pose_data, dt)
            
            # Let the current game render itself and check for menu commands
            result = self.current_game.render(frame)

            # Check for a command to switch games
            if isinstance(result, str):
                selected_option = result
                if selected_option == "Start Boxing Game":
                    self.current_game = BoxingGame()
                elif selected_option == "Start Fruit Ninja Game":
                    self.current_game = FruitNinjaGame()
                ## NEW: Add case for Reaction Time Game
                elif selected_option == "Start Reaction Time Game":
                    self.current_game = ReactionTimeGame()
                elif selected_option == "Exit":
                    break
            else:
                frame = result

            cv2.imshow('Body Game Platform', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            ## NEW: Pass key presses to the Reaction Time game if it's active
            if isinstance(self.current_game, ReactionTimeGame):
                self.current_game.handle_key(key)

            if key == ord('m') and not isinstance(self.current_game, MainMenu):
                print("Returning to menu...")
                self.current_game = MainMenu()

        # Final cleanup
        self.cap.release()
        self.hands.close()
        self.pose.close()
        cv2.destroyAllWindows()