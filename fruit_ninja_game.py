# fruit_ninja_game.py

import cv2
import mediapipe as mp
import time
import random

from game import Game
from helpers import calculate_velocity, line_circle_intersection

class FruitNinjaGame(Game):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_data = {'p1': None, 'p2': None}
        
        # Game parameters
        ## GAMEPLAY: Slower gravity for a floatier feel.
        self.gravity = .8
        self.spawn_interval = 0.5
        ## GAMEPLAY: Increased velocity threshold to require a faster swipe.
        self.slice_velocity_threshold = 800.0 # Pixels per second
        self.fruit_radius = 50
        self.bomb_radius = 50
        self.fruit_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        
        # Player 1 (Left side)
        self.p1_score = 0
        self.p1_fruits = []
        self.p1_bombs = []
        self.p1_feedback = ""
        self.p1_prev_left_wrist_screen = None
        self.p1_prev_right_wrist_screen = None
        self.p1_last_spawn = time.time()
        ## NEW FEATURE: Lists for visual effects
        self.p1_sliced_pieces = []
        self.p1_explosions = []
        
        # Player 2 (Right side)
        self.p2_score = 0
        self.p2_fruits = []
        self.p2_bombs = []
        self.p2_feedback = ""
        self.p2_prev_left_wrist_screen = None
        self.p2_prev_right_wrist_screen = None
        self.p2_last_spawn = time.time()
        ## NEW FEATURE: Lists for visual effects
        self.p2_sliced_pieces = []
        self.p2_explosions = []
        
        self.prev_time = time.time()

    def reset(self):
        """Reset game state for a new session."""
        self.__init__() # Re-initialize all variables

    def spawn_object(self, is_fruit=True):
        """Spawn a fruit or bomb."""
        ## GAMEPLAY: Slower initial velocity for easier slicing.
        vx = random.uniform(-0.1, 0.1)
        vy = random.uniform(-1, -.5)
        color = random.choice(self.fruit_colors) if is_fruit else (0, 0, 0)
        radius = self.fruit_radius if is_fruit else self.bomb_radius
        return {'x': random.uniform(0.2, 0.8), 'y': 1.0, 'vx': vx, 'vy': vy, 'color': color, 'radius': radius}

    def update_objects(self, objects, dt):
        """Update positions of fruits, bombs, and sliced fruit pieces."""
        new_objects = []
        for obj in objects:
            obj['x'] += obj['vx'] * dt
            obj['y'] += obj['vy'] * dt
            obj['vy'] += self.gravity * dt
            
            ## NEW FEATURE: Handle lifetime for fruit pieces
            if 'life' in obj:
                obj['life'] -= 1
                if obj['life'] <= 0:
                    continue  # This piece has expired

            if obj['y'] < 1.2:
                new_objects.append(obj)
        return new_objects
    
    ## NEW FEATURE: Update method for explosion effects
    def update_effects(self):
        """Update active explosion animations."""
        for explosions in [self.p1_explosions, self.p2_explosions]:
            next_explosions = []
            for exp in explosions:
                exp['radius'] += 4 # Expansion speed
                exp['life'] -= 1
                if exp['life'] > 0:
                    next_explosions.append(exp)
            if explosions is self.p1_explosions:
                self.p1_explosions = next_explosions
            else:
                self.p2_explosions = next_explosions

    def check_slice(self, prev_pos, curr_pos, objects, half_width, height):
        """Check if swipe slices any object."""
        sliced = []
        if prev_pos is None or curr_pos is None:
            return sliced
        start = (prev_pos[0], prev_pos[1])
        end = (curr_pos[0], curr_pos[1])
        for i, obj in enumerate(objects):
            center = (obj['x'] * half_width, obj['y'] * height)
            if line_circle_intersection(start, end, center, obj['radius']):
                sliced.append(i)
        return sliced

    def handle_input(self, pose_data, half_width, height):
        """Process pose data and detect swipes for both players."""
        self.pose_data = pose_data
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0: return
        self.prev_time = current_time

        # Spawning logic (unchanged)
        if current_time - self.p1_last_spawn > self.spawn_interval:
            self.p1_fruits.append(self.spawn_object(is_fruit=random.random() < 0.8))
            self.p1_last_spawn = current_time
        if current_time - self.p2_last_spawn > self.spawn_interval:
            self.p2_fruits.append(self.spawn_object(is_fruit=random.random() < 0.8))
            self.p2_last_spawn = current_time
        
        # Update all game objects and effects
        self.p1_fruits = self.update_objects(self.p1_fruits, dt)
        self.p1_bombs = self.update_objects(self.p1_bombs, dt)
        self.p1_sliced_pieces = self.update_objects(self.p1_sliced_pieces, dt)
        self.update_effects()
        
        self.p2_fruits = self.update_objects(self.p2_fruits, dt)
        self.p2_bombs = self.update_objects(self.p2_bombs, dt)
        self.p2_sliced_pieces = self.update_objects(self.p2_sliced_pieces, dt)
        
        # --- Process Player 1 Slicing ---
        if pose_data['p1'] and pose_data['p1'].pose_landmarks:
            landmarks = pose_data['p1'].pose_landmarks.landmark
            wrists = {
                'left': [landmarks[15].x * half_width, landmarks[15].y * height],
                'right': [landmarks[16].x * half_width, landmarks[16].y * height]
            }
            prev_wrists = {'left': self.p1_prev_left_wrist_screen, 'right': self.p1_prev_right_wrist_screen}

            for hand in ['left', 'right']:
                if prev_wrists[hand]:
                    velocity = calculate_velocity(wrists[hand], prev_wrists[hand], dt)
                    if velocity > self.slice_velocity_threshold:
                        # Check fruit slices
                        sliced_fruits = self.check_slice(prev_wrists[hand], wrists[hand], self.p1_fruits, half_width, height)
                        for i in sorted(sliced_fruits, reverse=True):
                            fruit = self.p1_fruits.pop(i)
                            self.p1_score += 1
                            self.p1_feedback = "Sliced!"
                            ## NEW FEATURE: Create two half-fruit pieces
                            for _ in range(2):
                                piece = {'x': fruit['x'], 'y': fruit['y'], 'vx': random.uniform(-1, 1), 'vy': fruit['vy'] - 0.5,
                                         'color': fruit['color'], 'radius': self.fruit_radius // 2, 'life': 20}
                                self.p1_sliced_pieces.append(piece)

                        # Check bomb slices
                        sliced_bombs = self.check_slice(prev_wrists[hand], wrists[hand], self.p1_bombs, half_width, height)
                        for i in sorted(sliced_bombs, reverse=True):
                            bomb = self.p1_bombs.pop(i)
                            self.p1_score -= 5
                            self.p1_feedback = "Boom!"
                            ## NEW FEATURE: Create an explosion effect
                            self.p1_explosions.append({'x': bomb['x'], 'y': bomb['y'], 'radius': 10, 'life': 15})

            self.p1_prev_left_wrist_screen = wrists['left']
            self.p1_prev_right_wrist_screen = wrists['right']

        # --- Process Player 2 Slicing (Identical logic) ---
        if pose_data['p2'] and pose_data['p2'].pose_landmarks:
            landmarks = pose_data['p2'].pose_landmarks.landmark
            wrists = {
                'left': [landmarks[15].x * half_width, landmarks[15].y * height],
                'right': [landmarks[16].x * half_width, landmarks[16].y * height]
            }
            prev_wrists = {'left': self.p2_prev_left_wrist_screen, 'right': self.p2_prev_right_wrist_screen}

            for hand in ['left', 'right']:
                if prev_wrists[hand]:
                    velocity = calculate_velocity(wrists[hand], prev_wrists[hand], dt)
                    if velocity > self.slice_velocity_threshold:
                        sliced_fruits = self.check_slice(prev_wrists[hand], wrists[hand], self.p2_fruits, half_width, height)
                        for i in sorted(sliced_fruits, reverse=True):
                            fruit = self.p2_fruits.pop(i)
                            self.p2_score += 1
                            self.p2_feedback = "Sliced!"
                            for _ in range(2):
                                self.p2_sliced_pieces.append({'x': fruit['x'], 'y': fruit['y'], 'vx': random.uniform(-1, 1), 'vy': fruit['vy'] - 0.5,
                                         'color': fruit['color'], 'radius': self.fruit_radius // 2, 'life': 20})
                        sliced_bombs = self.check_slice(prev_wrists[hand], wrists[hand], self.p2_bombs, half_width, height)
                        for i in sorted(sliced_bombs, reverse=True):
                            bomb = self.p2_bombs.pop(i)
                            self.p2_score -= 5
                            self.p2_feedback = "Boom!"
                            self.p2_explosions.append({'x': bomb['x'], 'y': bomb['y'], 'radius': 10, 'life': 15})
            
            self.p2_prev_left_wrist_screen = wrists['left']
            self.p2_prev_right_wrist_screen = wrists['right']


    def update(self, pose_data, dt):
        height, half_width = 1080, 960
        self.handle_input(pose_data, half_width, height)

    def render(self, frame):
        dt = time.time() - self.prev_time
        self.update(self.pose_data, dt)
        
        height, width, _ = frame.shape
        half_width = width // 2
        image_p1 = frame[:, :half_width]
        image_p2 = frame[:, half_width:]

        # --- Draw for Player 1 ---
        if self.pose_data['p1']:
            self.mp_drawing.draw_landmarks(image_p1, self.pose_data['p1'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        for obj_list in [self.p1_fruits, self.p1_bombs, self.p1_sliced_pieces]:
            for obj in obj_list:
                cv2.circle(image_p1, (int(obj['x'] * half_width), int(obj['y'] * height)), obj['radius'], obj['color'], -1)
        for exp in self.p1_explosions:
            cv2.circle(image_p1, (int(exp['x'] * half_width), int(exp['y'] * height)), int(exp['radius']), (255, 255, 255), 5)

        # --- Draw for Player 2 ---
        if self.pose_data['p2']:
            self.mp_drawing.draw_landmarks(image_p2, self.pose_data['p2'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        for obj_list in [self.p2_fruits, self.p2_bombs, self.p2_sliced_pieces]:
            for obj in obj_list:
                cv2.circle(image_p2, (int(obj['x'] * half_width), int(obj['y'] * height)), obj['radius'], obj['color'], -1)
        for exp in self.p2_explosions:
            cv2.circle(image_p2, (int(exp['x'] * half_width), int(exp['y'] * height)), int(exp['radius']), (255, 255, 255), 5)
        
        # Combine images and draw UI
        frame[:, :half_width] = image_p1
        frame[:, half_width:] = image_p2
        cv2.line(frame, (half_width, 0), (half_width, height), (255, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (width, 150), (20, 20, 20), -1)
        # (Score text rendering remains the same)
        # Player 1 Info
        cv2.putText(frame, 'PLAYER 1', (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p1_score}', (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p1_feedback, (30, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Player 2 Info
        cv2.putText(frame, 'PLAYER 2', (width - 300, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p2_score}', (width - 300, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p2_feedback, (width - 300, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame