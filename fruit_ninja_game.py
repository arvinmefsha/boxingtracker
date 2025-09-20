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
        self.pose_data = {'p1': None, 'p2': None}  # Store pose data
        
        # Game parameters
        self.gravity = 0.5  # Pixels per frame squared
        self.spawn_interval = 0.5  # Seconds between spawns
        self.slice_velocity_threshold = 1.0  # Normalized velocity for swipe
        self.fruit_radius = 30
        self.bomb_radius = 30
        self.fruit_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Blue, Red
        
        # Player 1 (Left side)
        self.p1_score = 0
        self.p1_fruits = []
        self.p1_bombs = []
        self.p1_feedback = ""
        self.p1_prev_left_wrist_screen = None
        self.p1_prev_right_wrist_screen = None
        self.p1_last_spawn = time.time()
        
        # Player 2 (Right side)
        self.p2_score = 0
        self.p2_fruits = []
        self.p2_bombs = []
        self.p2_feedback = ""
        self.p2_prev_left_wrist_screen = None
        self.p2_prev_right_wrist_screen = None
        self.p2_last_spawn = time.time()
        
        self.prev_time = time.time()

    def reset(self):
        """Reset game state for a new session."""
        self.p1_score = 0
        self.p1_fruits = []
        self.p1_bombs = []
        self.p1_feedback = ""
        self.p1_prev_left_wrist_screen = None
        self.p1_prev_right_wrist_screen = None
        self.p1_last_spawn = time.time()
        
        self.p2_score = 0
        self.p2_fruits = []
        self.p2_bombs = []
        self.p2_feedback = ""
        self.p2_prev_left_wrist_screen = None
        self.p2_prev_right_wrist_screen = None
        self.p2_last_spawn = time.time()
        
        self.prev_time = time.time()
        self.pose_data = {'p1': None, 'p2': None}

    def spawn_object(self, is_fruit=True):
        """Spawn a fruit or bomb."""
        vx = random.uniform(-5, 5)
        vy = random.uniform(-25, -15)  # Initial upward velocity
        color = random.choice(self.fruit_colors) if is_fruit else (0, 0, 0)
        radius = self.fruit_radius if is_fruit else self.bomb_radius
        return {'x': random.uniform(0.2, 0.8), 'y': 1.0, 'vx': vx, 'vy': vy, 'color': color, 'radius': radius}  # Normalized positions

    def update_objects(self, objects, dt, half_width, height):
        """Update positions of fruits or bombs."""
        new_objects = []
        for obj in objects:
            obj['x'] += obj['vx'] * dt
            obj['vy'] += self.gravity * dt
            obj['y'] += obj['vy'] * dt
            if obj['y'] < 1.5:  # Remove if off screen (normalized y >1 is bottom)
                new_objects.append(obj)
        return new_objects

    def check_slice(self, prev_pos, curr_pos, objects):
        """Check if swipe slices any object."""
        sliced = []
        if prev_pos is None or curr_pos is None:
            return sliced
        # Convert to pixels
        start = (prev_pos[0], prev_pos[1])
        end = (curr_pos[0], curr_pos[1])
        for i, obj in enumerate(objects):
            center = (obj['x'], obj['y'])
            if line_circle_intersection(start, end, center, obj['radius']):
                sliced.append(i)
        return sliced

    def handle_input(self, pose_data, half_width, height):
        """Process pose data and detect swipes for both players."""
        self.pose_data = pose_data  # Store for rendering
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        # Spawn for Player 1
        if current_time - self.p1_last_spawn > self.spawn_interval:
            if random.random() < 0.8:  # 80% chance fruit
                self.p1_fruits.append(self.spawn_object(is_fruit=True))
            else:
                self.p1_bombs.append(self.spawn_object(is_fruit=False))
            self.p1_last_spawn = current_time

        # Spawn for Player 2
        if current_time - self.p2_last_spawn > self.spawn_interval:
            if random.random() < 0.8:
                self.p2_fruits.append(self.spawn_object(is_fruit=True))
            else:
                self.p2_bombs.append(self.spawn_object(is_fruit=False))
            self.p2_last_spawn = current_time

        # Update objects (normalized to pixels for intersection)
        self.p1_fruits = self.update_objects(self.p1_fruits, dt, half_width, height)
        self.p1_bombs = self.update_objects(self.p1_bombs, dt, half_width, height)
        self.p2_fruits = self.update_objects(self.p2_fruits, dt, half_width, height)
        self.p2_bombs = self.update_objects(self.p2_bombs, dt, half_width, height)

        # Process Player 1
        if pose_data['p1'] and pose_data['p1'].pose_landmarks:
            landmarks = pose_data['p1'].pose_landmarks.landmark
            LEFT_WRIST = self.mp_pose.PoseLandmark.LEFT_WRIST.value
            RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

            p1_left_wrist = [landmarks[LEFT_WRIST].x * half_width, landmarks[LEFT_WRIST].y * height]
            p1_right_wrist = [landmarks[RIGHT_WRIST].x * half_width, landmarks[RIGHT_WRIST].y * height]

            # Left hand
            if self.p1_prev_left_wrist_screen:
                v_left = calculate_velocity(p1_left_wrist, self.p1_prev_left_wrist_screen, dt)
                if v_left > self.slice_velocity_threshold:
                    # Check fruits
                    sliced_fruits = self.check_slice(self.p1_prev_left_wrist_screen, p1_left_wrist, 
                                                     [{'x': f['x'] * half_width, 'y': f['y'] * height, 'radius': f['radius']} for f in self.p1_fruits])
                    for i in sorted(sliced_fruits, reverse=True):
                        del self.p1_fruits[i]
                        self.p1_score += 1
                        self.p1_feedback = "Sliced!"
                    # Check bombs
                    sliced_bombs = self.check_slice(self.p1_prev_left_wrist_screen, p1_left_wrist, 
                                                    [{'x': b['x'] * half_width, 'y': b['y'] * height, 'radius': b['radius']} for b in self.p1_bombs])
                    for i in sorted(sliced_bombs, reverse=True):
                        del self.p1_bombs[i]
                        self.p1_score -= 5
                        self.p1_feedback = "Boom!"

            # Right hand
            if self.p1_prev_right_wrist_screen:
                v_right = calculate_velocity(p1_right_wrist, self.p1_prev_right_wrist_screen, dt)
                if v_right > self.slice_velocity_threshold:
                    sliced_fruits = self.check_slice(self.p1_prev_right_wrist_screen, p1_right_wrist, 
                                                     [{'x': f['x'] * half_width, 'y': f['y'] * height, 'radius': f['radius']} for f in self.p1_fruits])
                    for i in sorted(sliced_fruits, reverse=True):
                        del self.p1_fruits[i]
                        self.p1_score += 1
                        self.p1_feedback = "Sliced!"
                    sliced_bombs = self.check_slice(self.p1_prev_right_wrist_screen, p1_right_wrist, 
                                                    [{'x': b['x'] * half_width, 'y': b['y'] * height, 'radius': b['radius']} for b in self.p1_bombs])
                    for i in sorted(sliced_bombs, reverse=True):
                        del self.p1_bombs[i]
                        self.p1_score -= 5
                        self.p1_feedback = "Boom!"

            self.p1_prev_left_wrist_screen = p1_left_wrist
            self.p1_prev_right_wrist_screen = p1_right_wrist

        # Process Player 2 (offset x by half_width in render)
        if pose_data['p2'] and pose_data['p2'].pose_landmarks:
            landmarks = pose_data['p2'].pose_landmarks.landmark
            LEFT_WRIST = self.mp_pose.PoseLandmark.LEFT_WRIST.value
            RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

            p2_left_wrist = [landmarks[LEFT_WRIST].x * half_width, landmarks[LEFT_WRIST].y * height]
            p2_right_wrist = [landmarks[RIGHT_WRIST].x * half_width, landmarks[RIGHT_WRIST].y * height]

            # Left hand
            if self.p2_prev_left_wrist_screen:
                v_left = calculate_velocity(p2_left_wrist, self.p2_prev_left_wrist_screen, dt)
                if v_left > self.slice_velocity_threshold:
                    sliced_fruits = self.check_slice(self.p2_prev_left_wrist_screen, p2_left_wrist, 
                                                     [{'x': f['x'] * half_width, 'y': f['y'] * height, 'radius': f['radius']} for f in self.p2_fruits])
                    for i in sorted(sliced_fruits, reverse=True):
                        del self.p2_fruits[i]
                        self.p2_score += 1
                        self.p2_feedback = "Sliced!"
                    sliced_bombs = self.check_slice(self.p2_prev_left_wrist_screen, p2_left_wrist, 
                                                    [{'x': b['x'] * half_width, 'y': b['y'] * height, 'radius': b['radius']} for b in self.p2_bombs])
                    for i in sorted(sliced_bombs, reverse=True):
                        del self.p2_bombs[i]
                        self.p2_score -= 5
                        self.p2_feedback = "Boom!"

            # Right hand
            if self.p2_prev_right_wrist_screen:
                v_right = calculate_velocity(p2_right_wrist, self.p2_prev_right_wrist_screen, dt)
                if v_right > self.slice_velocity_threshold:
                    sliced_fruits = self.check_slice(self.p2_prev_right_wrist_screen, p2_right_wrist, 
                                                     [{'x': f['x'] * half_width, 'y': f['y'] * height, 'radius': f['radius']} for f in self.p2_fruits])
                    for i in sorted(sliced_fruits, reverse=True):
                        del self.p2_fruits[i]
                        self.p2_score += 1
                        self.p2_feedback = "Sliced!"
                    sliced_bombs = self.check_slice(self.p2_prev_right_wrist_screen, p2_right_wrist, 
                                                    [{'x': b['x'] * half_width, 'y': b['y'] * height, 'radius': b['radius']} for b in self.p2_bombs])
                    for i in sorted(sliced_bombs, reverse=True):
                        del self.p2_bombs[i]
                        self.p2_score -= 5
                        self.p2_feedback = "Boom!"

            self.p2_prev_left_wrist_screen = p2_left_wrist
            self.p2_prev_right_wrist_screen = p2_right_wrist

    def update(self, pose_data, dt):
        """Update game state."""
        height = 1080
        half_width = 960  # 1920 / 2
        self.handle_input(pose_data, half_width, height)

    def render(self, frame):
        """Render the fruit ninja game visuals."""
        height, width, _ = frame.shape
        half_width = width // 2
        image_p1 = frame[:, :half_width]
        image_p2 = frame[:, half_width:]

        # Draw landmarks for Player 1
        if self.pose_data['p1'] and self.pose_data['p1'].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_p1, self.pose_data['p1'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(200, 100, 0), thickness=2, circle_radius=2)
            )

        # Draw landmarks for Player 2
        if self.pose_data['p2'] and self.pose_data['p2'].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_p2, self.pose_data['p2'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 100, 200), thickness=2, circle_radius=2)
            )

        # Draw objects for Player 1
        for fruit in self.p1_fruits:
            cx = int(fruit['x'] * half_width)
            cy = int(fruit['y'] * height)
            cv2.circle(image_p1, (cx, cy), fruit['radius'], fruit['color'], -1)
        for bomb in self.p1_bombs:
            cx = int(bomb['x'] * half_width)
            cy = int(bomb['y'] * height)
            cv2.circle(image_p1, (cx, cy), bomb['radius'], bomb['color'], -1)
            cv2.putText(image_p1, 'B', (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw objects for Player 2
        for fruit in self.p2_fruits:
            cx = int(fruit['x'] * half_width)
            cy = int(fruit['y'] * height)
            cv2.circle(image_p2, (cx, cy), fruit['radius'], fruit['color'], -1)
        for bomb in self.p2_bombs:
            cx = int(bomb['x'] * half_width)
            cy = int(bomb['y'] * height)
            cv2.circle(image_p2, (cx, cy), bomb['radius'], bomb['color'], -1)
            cv2.putText(image_p2, 'B', (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine images back
        frame[:, :half_width] = image_p1
        frame[:, half_width:] = image_p2

        # Resize to 1920x1080
        frame = cv2.resize(frame, (1920, 1080))

        # Draw divider line
        cv2.line(frame, (960, 0), (960, 1080), (255, 255, 255), 2)

        # Dashboard rectangle
        cv2.rectangle(frame, (0, 0), (1920, 150), (20, 20, 20), -1)

        # Player 1 Info
        cv2.putText(frame, 'PLAYER 1', (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p1_score}', (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p1_feedback, (30, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Player 2 Info
        cv2.putText(frame, 'PLAYER 2', (1920 - 300, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p2_score}', (1920 - 300, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p2_feedback, (1920 - 300, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return frame