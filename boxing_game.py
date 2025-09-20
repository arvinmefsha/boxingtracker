# boxing_game.py

import cv2
import mediapipe as mp
import time
import numpy as np

from game import Game
from helpers import calculate_distance, calculate_velocity, calculate_angle
from fighter_state import FighterState

class BoxingGame(Game):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_data = {'p1': None, 'p2': None}  # Store pose data
        
        self.p1_state = FighterState()
        self.p2_state = FighterState()
        
        self.prev_time = time.time()

    def reset(self):
        """Reset game state for a new session."""
        self.p1_state = FighterState()
        self.p2_state = FighterState()
        
        self.prev_time = time.time()
        self.pose_data = {'p1': None, 'p2': None}

    def process_fighter(self, landmarks, state, dt):
        if not landmarks:
            return
        
        # Landmark definitions
        RIGHT_SHOULDER = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RIGHT_ELBOW = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        RIGHT_WRIST = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RIGHT_HIP = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        RIGHT_KNEE = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        RIGHT_ANKLE = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
       
        # Get coordinates
        right_wrist_coords = [RIGHT_WRIST.x, RIGHT_WRIST.y]
        right_shoulder_coords = [RIGHT_SHOULDER.x, RIGHT_SHOULDER.y]
        right_elbow_coords = [RIGHT_ELBOW.x, RIGHT_ELBOW.y]
        right_ankle_coords = [RIGHT_ANKLE.x, RIGHT_ANKLE.y]
        right_knee_coords = [RIGHT_KNEE.x, RIGHT_KNEE.y]
        right_hip_coords = [RIGHT_HIP.x, RIGHT_HIP.y]
        
        # --- REFINED PUNCH DETECTION ---
        if state.prev_right_wrist:
            velocity = calculate_velocity(right_wrist_coords, state.prev_right_wrist, dt)
            shoulder_wrist_dist = calculate_distance(right_shoulder_coords, right_wrist_coords)
           
            # Condition to start punch: high velocity AND hand moving away from shoulder
            is_extending = shoulder_wrist_dist > state.prev_shoulder_wrist_dist
           
            if state.right_hand_state == 'IDLE' and velocity > 1.1 and is_extending:
                state.right_hand_state = 'PUNCHING'
            elif state.right_hand_state == 'PUNCHING':
                # Condition to end punch: velocity drops
                if velocity < 0.3:
                    state.right_punch_count += 1
                    state.right_hand_state = 'RETRACTING'
            elif state.right_hand_state == 'RETRACTING':
                # Condition to return to idle: hand is back near shoulder
                if shoulder_wrist_dist < state.prev_shoulder_wrist_dist or shoulder_wrist_dist < 0.2:
                    state.right_hand_state = 'IDLE'
        state.prev_right_wrist = right_wrist_coords
        state.prev_shoulder_wrist_dist = calculate_distance(right_shoulder_coords, right_wrist_coords)
        
        # --- REFINED KICK AND KNEE DETECTION ---
        if state.prev_right_ankle and state.prev_right_knee:
            ankle_velocity = calculate_velocity(right_ankle_coords, state.prev_right_ankle, dt)
            knee_vertical_velocity = (state.prev_right_knee[1] - right_knee_coords[1]) / dt
            leg_angle = calculate_angle(right_hip_coords, right_knee_coords, right_ankle_coords)
            
            # KNEE LOGIC: high upward knee velocity AND a bent leg
            is_knee_raised = right_knee_coords[1] < right_hip_coords[1]
            if state.right_leg_state == 'IDLE' and knee_vertical_velocity > 0.8 and is_knee_raised and leg_angle < 100:
                 state.right_leg_state = 'KNEEING'
            elif state.right_leg_state == 'KNEEING':
                # End knee when knee starts moving down
                if knee_vertical_velocity < 0:
                    state.right_knee_count += 1
                    state.right_leg_state = 'IDLE'
            
            # KICK LOGIC: high ankle velocity, then extension, then retraction
            if state.right_leg_state == 'IDLE' and ankle_velocity > 1.5:
                state.right_leg_state = 'KICKING'
            elif state.right_leg_state == 'KICKING':
                # Only count the kick if the leg extends and then velocity drops
                if leg_angle > 160 and ankle_velocity < 0.5:
                    state.right_kick_count += 1
                    state.right_leg_state = 'IDLE'
                # Timeout if kick is not completed
                elif ankle_velocity < 0.5:
                    state.right_leg_state = 'IDLE'
        state.prev_right_ankle = right_ankle_coords
        state.prev_right_knee = right_knee_coords

    def handle_input(self, pose_data):
        """Process pose data for both players."""
        self.pose_data = pose_data  # Store pose data for rendering
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt == 0:
            return
        
        self.prev_time = current_time

        # Process Player 1
        if pose_data['p1'] and pose_data['p1'].pose_landmarks:
            self.process_fighter(pose_data['p1'].pose_landmarks.landmark, self.p1_state, dt)

        # Process Player 2
        if pose_data['p2'] and pose_data['p2'].pose_landmarks:
            self.process_fighter(pose_data['p2'].pose_landmarks.landmark, self.p2_state, dt)

    def update(self, pose_data, dt):
        """Update game state (currently handled in handle_input for boxing)."""
        self.handle_input(pose_data)

    def render(self, frame):
        """Render the boxing game visuals (split-screen, landmarks, UI)."""
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

        # Combine images back
        frame[:, :half_width] = image_p1
        frame[:, half_width:] = image_p2

        # Resize to 1920x1080
        frame = cv2.resize(frame, (1920, 1080))

        # Draw divider line
        cv2.line(frame, (960, 0), (960, 1080), (255, 255, 255), 2)

        # Dashboard rectangle
        cv2.rectangle(frame, (0, 0), (1920, 140), (20, 20, 20), -1)

        # Player 1 Info
        cv2.putText(frame, 'FIGHTER 1 (BLUE)', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
        cv2.putText(frame, f'PUNCHES: {self.p1_state.right_punch_count}', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'KICKS: {self.p1_state.right_kick_count}', (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'KNEES: {self.p1_state.right_knee_count}', (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'ARM: {self.p1_state.right_hand_state}', (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'LEG: {self.p1_state.right_leg_state}', (250, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Player 2 Info
        p2_start_x = 1920 - 300
        cv2.putText(frame, 'FIGHTER 2 (RED)', (p2_start_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
        cv2.putText(frame, f'PUNCHES: {self.p2_state.right_punch_count}', (p2_start_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'KICKS: {self.p2_state.right_kick_count}', (p2_start_x, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'KNEES: {self.p2_state.right_knee_count}', (p2_start_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'ARM: {self.p2_state.right_hand_state}', (p2_start_x - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'LEG: {self.p2_state.right_leg_state}', (p2_start_x - 200, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame