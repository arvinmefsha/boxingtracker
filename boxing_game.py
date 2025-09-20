import cv2
import mediapipe as mp
import time

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

    def detect_punch(self, state, right_wrist, right_shoulder, dt, v_start=1.10, v_end=0.30, retract_dist=0.20):
        """
        Detects a right-hand straight punch based on wrist velocity and shoulder–wrist extension/retraction.
        Returns: (did_punch: bool, debug: dict)
        """
        debug = {}
        did_punch = False

        wrist_xy = [right_wrist.x, right_wrist.y]
        shoulder_xy = [right_shoulder.x, right_shoulder.y]

        if state.prev_right_wrist is not None:
            v = calculate_velocity(wrist_xy, state.prev_right_wrist, dt)
            d = calculate_distance(shoulder_xy, wrist_xy)
            is_extending = d > state.prev_shoulder_wrist_dist

            debug.update({"wrist_v": v, "shoulder_wrist_d": d, "is_extending": is_extending, "arm_state": state.right_hand_state})

            if state.right_hand_state == 'IDLE' and v > v_start and is_extending:
                state.right_hand_state = 'PUNCHING'
            elif state.right_hand_state == 'PUNCHING':
                if v < v_end:
                    state.right_punch_count += 1
                    did_punch = True
                    state.right_hand_state = 'RETRACTING'
            elif state.right_hand_state == 'RETRACTING':
                if d < state.prev_shoulder_wrist_dist or d < retract_dist:
                    state.right_hand_state = 'IDLE'

            state.prev_shoulder_wrist_dist = d

        state.prev_right_wrist = wrist_xy
        if state.prev_shoulder_wrist_dist == 0:
            state.prev_shoulder_wrist_dist = calculate_distance(shoulder_xy, wrist_xy)

        return did_punch, debug

    def detect_kick(self, state, right_hip, right_knee, right_ankle, dt, ankle_v_start=1.50, ankle_v_idle=0.50, min_extension_angle=160):
        """
        Detects a right-leg kick using ankle velocity and knee extension angle.
        Returns: (did_kick: bool, debug: dict)
        """
        debug = {}
        did_kick = False

        hip_xy = [right_hip.x, right_hip.y]
        knee_xy = [right_knee.x, right_knee.y]
        ankle_xy = [right_ankle.x, right_ankle.y]

        if state.prev_right_ankle is not None and state.prev_right_knee is not None:
            ankle_v = calculate_velocity(ankle_xy, state.prev_right_ankle, dt)
            leg_angle = calculate_angle(hip_xy, knee_xy, ankle_xy)

            debug.update({"ankle_v": ankle_v, "leg_angle": leg_angle, "leg_state": state.right_leg_state})

            if state.right_leg_state == 'IDLE' and ankle_v > ankle_v_start:
                state.right_leg_state = 'KICKING'
            elif state.right_leg_state == 'KICKING':
                if leg_angle > min_extension_angle and ankle_v < ankle_v_idle:
                    state.right_kick_count += 1
                    did_kick = True
                    state.right_leg_state = 'IDLE'
                elif ankle_v < ankle_v_idle:
                    state.right_leg_state = 'IDLE'

        state.prev_right_ankle = ankle_xy
        state.prev_right_knee = knee_xy
        return did_kick, debug

    def process_fighter(self, landmarks, state, dt):
        if not landmarks:
            return

        RS = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RW = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RH = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        RK = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        RA = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        self.detect_punch(state, RW, RS, dt)
        self.detect_kick(state, RH, RK, RA, dt)

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

        # Draw the black dashboard box FIRST
        cv2.rectangle(frame, (0, 0), (1920, 110), (20, 20, 20), -1)
        
        # Draw the white dividing line SECOND, so it appears on top of the box
        cv2.line(frame, (960, 0), (960, 1080), (255, 255, 255), 2)
        
        margin = 15

        # --- Player 1 Info (Left Panel) ---
        # Title and Counters are left-aligned to the screen edge
        cv2.putText(frame, 'FIGHTER 1 (BLUE)', (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
        cv2.putText(frame, f'PUNCHES: {self.p1_state.right_punch_count}', (margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'KICKS: {self.p1_state.right_kick_count}', (margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # States are right-aligned to the center divider
        p1_arm_text = f'ARM: {self.p1_state.right_hand_state}'
        p1_leg_text = f'LEG: {self.p1_state.right_leg_state}'
        p1_arm_text_size = cv2.getTextSize(p1_arm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        p1_leg_text_size = cv2.getTextSize(p1_leg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        cv2.putText(frame, p1_arm_text, (960 - p1_arm_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p1_leg_text, (960 - p1_leg_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # --- Player 2 Info (Right Panel) ---
        # States are left-aligned to the center divider
        cv2.putText(frame, f'ARM: {self.p2_state.right_hand_state}', (960 + margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'LEG: {self.p2_state.right_leg_state}', (960 + margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Title and Counters are right-aligned to the screen edge
        p2_title = 'FIGHTER 2 (RED)'
        p2_title_size = cv2.getTextSize(p2_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, p2_title, (1920 - p2_title_size[0] - margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)

        p2_punch_text = f'PUNCHES: {self.p2_state.right_punch_count}'
        p2_kick_text = f'KICKS: {self.p2_state.right_kick_count}'
        p2_punch_text_size = cv2.getTextSize(p2_punch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        p2_kick_text_size = cv2.getTextSize(p2_kick_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        cv2.putText(frame, p2_punch_text, (1920 - p2_punch_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p2_kick_text, (1920 - p2_kick_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame