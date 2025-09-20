import cv2
import mediapipe as mp
import time
import math  # NEW: for angle-from-vertical

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

    # --------- helpers to ensure dynamic fields exist (backward compatible) ---------
    def _ensure_hand_fields(self, state, side: str):
        """Ensure per-hand fields exist (stateful compatibility across versions)."""
        hand_state_attr = f'{side}_hand_state'
        punch_count_attr = f'{side}_punch_count'
        last_punch_ts_attr = f'last_{side}_punch_ts'
        if not hasattr(state, hand_state_attr):
            setattr(state, hand_state_attr, 'IDLE')
        if not hasattr(state, punch_count_attr):
            setattr(state, punch_count_attr, 0)
        if not hasattr(state, last_punch_ts_attr):
            setattr(state, last_punch_ts_attr, 0.0)

    def _ensure_leg_fields(self, state, side: str):
        leg_state_attr = f'{side}_leg_state'
        kick_count_attr = f'{side}_kick_count'
        if not hasattr(state, leg_state_attr):
            setattr(state, leg_state_attr, 'IDLE')
        if not hasattr(state, kick_count_attr):
            setattr(state, kick_count_attr, 0)

    def _ensure_duck_fields(self, state):
        if not hasattr(state, 'duck_state'):
            state.duck_state = 'IDLE'
        if not hasattr(state, 'duck_count'):
            state.duck_count = 0

    # --------- PUNCH (both arms) ---------
    def detect_punch(
        self,
        state,
        shoulder, elbow, wrist,       # joints for this arm
        other_wrist,                  # wrist of opposite arm
        hip,                          # hip on the same side as this arm
        waist_y,                      # waist line y
        side: str,
        angle_threshold: float = 150.0,
        shoulder_min_angle: float = 45.0,
        cooldown_s: float = 0.25
    ):
        """
        Count a punch for the given arm ('left' or 'right') when ALL are true:
          1) Arm extension: angle(shoulder–elbow–wrist) > angle_threshold (default 150°)
          2) Both hands above waist: wrist.y < waist_y and other_wrist.y < waist_y
          3) Shoulder openness: angle(hip–shoulder–elbow) >= shoulder_min_angle (default 45°)
          4) Per-arm cooldown passed (default 0.25 s)
        Uses a simple two-state machine per arm: IDLE -> PUNCHING -> IDLE.
        Returns: (did_punch: bool, debug: dict)
        """
        debug = {}
        did_punch = False

        # Ensure state fields
        self._ensure_hand_fields(state, side)

        hand_state_attr = f'{side}_hand_state'
        punch_count_attr = f'{side}_punch_count'
        last_punch_ts_attr = f'last_{side}_punch_ts'

        # Angles
        arm_angle = calculate_angle(
            [shoulder.x, shoulder.y],
            [elbow.x, elbow.y],
            [wrist.x, wrist.y]
        )
        shoulder_angle = calculate_angle(
            [hip.x, hip.y],
            [shoulder.x, shoulder.y],
            [elbow.x, elbow.y]
        )

        # Height check: both hands above waist
        both_hands_above_waist = (wrist.y < waist_y) and (other_wrist.y < waist_y)

        # Cooldown
        now = time.time()
        last_ts = getattr(state, last_punch_ts_attr)

        current_state = getattr(state, hand_state_attr)
        debug.update({
            f"{side}_arm_angle": arm_angle,
            f"{side}_shoulder_angle": shoulder_angle,
            "both_hands_above_waist": both_hands_above_waist,
            f"{side}_hand_state": current_state,
            f"{side}_since_last_punch": now - last_ts
        })

        can_count_now = (
            arm_angle > angle_threshold and
            shoulder_angle >= shoulder_min_angle and
            both_hands_above_waist and
            (now - last_ts) >= cooldown_s
        )

        # State transitions
        if current_state == "IDLE":
            if can_count_now:
                setattr(state, punch_count_attr, getattr(state, punch_count_attr) + 1)
                setattr(state, hand_state_attr, "PUNCHING")
                setattr(state, last_punch_ts_attr, now)
                did_punch = True

        elif current_state == "PUNCHING":
            if (arm_angle <= angle_threshold) or (shoulder_angle < shoulder_min_angle) or (not both_hands_above_waist):
                setattr(state, hand_state_attr, "IDLE")

        return did_punch, debug

    # --------- KICK (both legs) ---------
    def detect_kick(self, state, hip, ankle, side: str, idle_buffer=0.05):
        """
        Kick detection for a specified leg ('left' or 'right').
        Counts a kick when the ankle rises above the hip (y decreases).
        Uses a small buffer before resetting to IDLE to avoid jitter.
        Returns: (did_kick: bool, debug: dict)
        """
        debug = {}
        did_kick = False

        # Ensure required attributes exist on state for the chosen side
        self._ensure_leg_fields(state, side)
        leg_state_attr = f'{side}_leg_state'
        kick_count_attr = f'{side}_kick_count'

        hip_y = hip.y
        ankle_y = ankle.y

        current_leg_state = getattr(state, leg_state_attr)

        # Kick start: ankle above hip
        if ankle_y < hip_y:
            if current_leg_state == 'IDLE':
                setattr(state, kick_count_attr, getattr(state, kick_count_attr) + 1)
                did_kick = True
                setattr(state, leg_state_attr, 'KICKING')

        # Reset to IDLE with a buffer to prevent flicker
        elif ankle_y > hip_y + idle_buffer:
            setattr(state, leg_state_attr, 'IDLE')

        debug.update({
            f"{side}_hip_y": hip_y,
            f"{side}_ankle_y": ankle_y,
            f"{side}_leg_state": getattr(state, leg_state_attr)
        })

        return did_kick, debug

    # --------- DUCK (either side) ---------
    def detect_duck(self, state, LH, LS, RH, RS, angle_from_vertical_thresh: float = 45.0, hysteresis: float = 3.0):
        """
        Count a 'duck' when the torso line (hip->shoulder) tilts more than
        `angle_from_vertical_thresh` degrees away from the vertical on EITHER side.

        - left_tilt  = angle between vector (LS - LH) and vertical axis
        - right_tilt = angle between vector (RS - RH) and vertical axis
        If max(left_tilt, right_tilt) > threshold -> DUCK.
        Simple state machine with small hysteresis to avoid repeat counts.
        """
        debug = {}
        did_duck = False
        self._ensure_duck_fields(state)

        def tilt_from_vertical(hip, shoulder):
            # Vector hip->shoulder
            dx = shoulder.x - hip.x
            dy = shoulder.y - hip.y
            # Angle from vertical: arctan2(|dx|, |dy|)
            # (vertical is dy direction; larger horizontal component -> bigger tilt)
            angle_rad = math.atan2(abs(dx), abs(dy) + 1e-8)
            return math.degrees(angle_rad)

        left_tilt = tilt_from_vertical(LH, LS)
        right_tilt = tilt_from_vertical(RH, RS)
        max_tilt = max(left_tilt, right_tilt)

        debug.update({
            "left_tilt_from_vertical_deg": left_tilt,
            "right_tilt_from_vertical_deg": right_tilt,
            "max_tilt_deg": max_tilt,
            "duck_state": state.duck_state
        })

        tilting_now = (max_tilt > angle_from_vertical_thresh)

        if state.duck_state == 'IDLE':
            if tilting_now:
                state.duck_count = getattr(state, 'duck_count', 0) + 1
                state.duck_state = 'DUCKING'
                did_duck = True
        else:  # DUCKING
            # reset once tilt drops below (threshold - hysteresis)
            if max_tilt < (angle_from_vertical_thresh - hysteresis):
                state.duck_state = 'IDLE'

        return did_duck, debug

    # --------- per-frame processing ---------
    def process_fighter(self, landmarks, state, dt):
        if not landmarks:
            return

        # Right side landmarks
        RS = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RE = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        RW = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RH = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        RK = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        RA = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Left side landmarks
        LS = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        LE = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        LW = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        LH = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        LK = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        LA = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # Waist line: midpoint of both hips
        waist_y = (LH.y + RH.y) / 2.0

        # Bilateral punches: pass same-side HIP for shoulder-angle guard
        self.detect_punch(state, RS, RE, RW, LW, RH, waist_y, 'right')
        self.detect_punch(state, LS, LE, LW, RW, LH, waist_y, 'left')

        # Bilateral kicks (UNCHANGED)
        self.detect_kick(state, RH, RA, 'right')
        self.detect_kick(state, LH, LA, 'left')

        # Duck by torso tilt from vertical (either side)
        self.detect_duck(state, LH, LS, RH, RS)

    # --------- engine integration ---------
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

    # --------- rendering ---------
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

        # Increase dashboard height to fit Ducks line
        dashboard_h = 145
        # Draw the black dashboard box FIRST
        cv2.rectangle(frame, (0, 0), (1920, dashboard_h), (20, 20, 20), -1)
        
        # Draw the white dividing line SECOND, so it appears on top of the box
        cv2.line(frame, (960, 0), (960, 1080), (255, 255, 255), 2)
        
        margin = 15

        # --- Player 1 Info (Left Panel) ---
        cv2.putText(frame, 'FIGHTER 1 (BLUE)', (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
        
        # Totals for P1
        p1_total_punches = getattr(self.p1_state, 'right_punch_count', 0) + getattr(self.p1_state, 'left_punch_count', 0)
        p1_total_kicks = getattr(self.p1_state, 'right_kick_count', 0) + getattr(self.p1_state, 'left_kick_count', 0)
        p1_ducks = getattr(self.p1_state, 'duck_count', 0)
        p1_punch_text = f'PUNCHES: {p1_total_punches}'
        p1_kick_text = f'KICKS: {p1_total_kicks}'
        p1_duck_text = f'DUCKS: {p1_ducks}'
        cv2.putText(frame, p1_punch_text, (margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p1_kick_text, (margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p1_duck_text, (margin, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # States are right-aligned to the center divider
        # Combine arm state: show PUNCHING if either arm is punching
        p1_arm_combined = 'PUNCHING' if (getattr(self.p1_state, 'right_hand_state', 'IDLE') == 'PUNCHING' or getattr(self.p1_state, 'left_hand_state', 'IDLE') == 'PUNCHING') else 'IDLE'
        p1_arm_text = f'ARM: {p1_arm_combined}'
        # Combine leg state
        p1_leg_combined = 'KICKING' if (getattr(self.p1_state, 'right_leg_state', 'IDLE') == 'KICKING' or getattr(self.p1_state, 'left_leg_state', 'IDLE') == 'KICKING') else 'IDLE'
        p1_leg_text = f'LEG: {p1_leg_combined}'
        p1_arm_text_size = cv2.getTextSize(p1_arm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        p1_leg_text_size = cv2.getTextSize(p1_leg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        cv2.putText(frame, p1_arm_text, (960 - p1_arm_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p1_leg_text, (960 - p1_leg_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # --- Player 2 Info (Right Panel) ---
        # States are left-aligned to the center divider
        p2_arm_combined = 'PUNCHING' if (getattr(self.p2_state, 'right_hand_state', 'IDLE') == 'PUNCHING' or getattr(self.p2_state, 'left_hand_state', 'IDLE') == 'PUNCHING') else 'IDLE'
        p2_arm_text = f'ARM: {p2_arm_combined}'
        p2_leg_combined = 'KICKING' if (getattr(self.p2_state, 'right_leg_state', 'IDLE') == 'KICKING' or getattr(self.p2_state, 'left_leg_state', 'IDLE') == 'KICKING') else 'IDLE'
        p2_leg_text = f'LEG: {p2_leg_combined}'
        cv2.putText(frame, p2_arm_text, (960 + margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p2_leg_text, (960 + margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Title and Counters are right-aligned to the screen edge
        p2_title = 'FIGHTER 2 (RED)'
        p2_title_size = cv2.getTextSize(p2_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, p2_title, (1920 - p2_title_size[0] - margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)

        # Totals for P2
        p2_total_punches = getattr(self.p2_state, 'right_punch_count', 0) + getattr(self.p2_state, 'left_punch_count', 0)
        p2_total_kicks = getattr(self.p2_state, 'right_kick_count', 0) + getattr(self.p2_state, 'left_kick_count', 0)
        p2_ducks = getattr(self.p2_state, 'duck_count', 0)
        p2_punch_text = f'PUNCHES: {p2_total_punches}'
        p2_kick_text = f'KICKS: {p2_total_kicks}'
        p2_duck_text = f'DUCKS: {p2_ducks}'
        p2_punch_text_size = cv2.getTextSize(p2_punch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        p2_kick_text_size = cv2.getTextSize(p2_kick_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        p2_duck_text_size = cv2.getTextSize(p2_duck_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        cv2.putText(frame, p2_punch_text, (1920 - p2_punch_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p2_kick_text, (1920 - p2_kick_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, p2_duck_text, (1920 - p2_duck_text_size[0] - margin, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame
