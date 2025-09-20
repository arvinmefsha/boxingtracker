import cv2
import mediapipe as mp
import numpy as np
import time

# --- Helper Functions for Calculations ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_velocity(p1, p2, dt):
    """Calculates the velocity between two points over a time delta."""
    if dt == 0:
        return 0
    distance = calculate_distance(p1, p2)
    return distance / dt


# --- Fighter State Class ---
class FighterState:
    def __init__(self):
        # Right Hand
        self.right_hand_state = 'IDLE'
        self.right_punch_count = 0
        self.prev_right_wrist = None
        self.prev_shoulder_wrist_dist = 0
        
        # Right Leg
        self.right_leg_state = 'IDLE' 
        self.right_kick_count = 0
        self.prev_right_ankle = None
        self.prev_right_knee = None

# Functions for detection
# --- New, Modular Detection Functions ---

def detect_punch(state: FighterState,
                 right_wrist, right_shoulder,
                 dt,
                 v_start=1.10, v_end=0.30, retract_dist=0.20):
    """
    Detects a right-hand straight punch based on wrist velocity and
    shoulder–wrist extension/retraction.
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
            # count at peak/slowdown (extension completed)
            if v < v_end:
                state.right_punch_count += 1
                did_punch = True
                state.right_hand_state = 'RETRACTING'

        elif state.right_hand_state == 'RETRACTING':
            # back near shoulder or clearly reducing extension
            if d < state.prev_shoulder_wrist_dist or d < retract_dist:
                state.right_hand_state = 'IDLE'

        state.prev_shoulder_wrist_dist = d

    # book-keeping
    state.prev_right_wrist = wrist_xy
    if state.prev_shoulder_wrist_dist == 0:
        state.prev_shoulder_wrist_dist = calculate_distance(shoulder_xy, wrist_xy)

    return did_punch, debug


def detect_kick(state: FighterState,
                right_hip, right_knee, right_ankle,
                dt,
                ankle_v_start=1.50, ankle_v_idle=0.50, min_extension_angle=160):
    """
    Detects a right-leg kick using ankle velocity and knee extension angle.
    Returns: (did_kick: bool, debug: dict)
    """
    debug = {}
    did_kick = False

    hip_xy   = [right_hip.x, right_hip.y]
    knee_xy  = [right_knee.x, right_knee.y]
    ankle_xy = [right_ankle.x, right_ankle.y]

    if state.prev_right_ankle is not None and state.prev_right_knee is not None:
        ankle_v = calculate_velocity(ankle_xy, state.prev_right_ankle, dt)
        leg_angle = calculate_angle(hip_xy, knee_xy, ankle_xy)  # 180° = fully extended

        debug.update({"ankle_v": ankle_v, "leg_angle": leg_angle, "leg_state": state.right_leg_state})

        if state.right_leg_state == 'IDLE' and ankle_v > ankle_v_start:
            state.right_leg_state = 'KICKING'

        elif state.right_leg_state == 'KICKING':
            # count when extended and slows down (impact/peak)
            if leg_angle > min_extension_angle and ankle_v < ankle_v_idle:
                state.right_kick_count += 1
                did_kick = True
                state.right_leg_state = 'IDLE'
            # timeout/abort if it slows without extension
            elif ankle_v < ankle_v_idle:
                state.right_leg_state = 'IDLE'

    # book-keeping
    state.prev_right_ankle = ankle_xy
    state.prev_right_knee  = knee_xy
    return did_kick, debug


# --- Initialization ---

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

p1_state = FighterState()
p2_state = FighterState()

prev_time = time.time()

# --- Main Application Loop ---

while cap.isOpened():
    success, image = cap.read()
    # --- ROBUSTNESS FIX ---
    # If frame is not read successfully or is empty, skip to next iteration
    if not success or image is None:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    
    # Ensure the image has width before proceeding
    if image.shape[1] == 0:
        print("Warning: Frame has zero width.")
        continue

    height, width, _ = image.shape

    image_p1 = image[:, :width//2]
    image_p2 = image[:, width//2:]

    # --- FIX: Corrected typo from COLOR_BGR_RGB to COLOR_BGR2RGB ---
    image_rgb_p1 = cv2.cvtColor(image_p1, cv2.COLOR_BGR2RGB)
    image_rgb_p2 = cv2.cvtColor(image_p2, cv2.COLOR_BGR2RGB)
    
    results_p1 = pose.process(image_rgb_p1)
    results_p2 = pose.process(image_rgb_p2)

    current_time = time.time()
    dt = current_time - prev_time
    
    # Make sure dt is not zero to avoid division errors
    if dt == 0:
        continue

    # --- Processing Logic (Unified for both fighters) ---
    def process_fighter(landmarks, state, dt):
        if not landmarks:
            return

        RS = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RE = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]   # (kept if you want to add elbow checks later)
        RW = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RH = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        RK = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        RA = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 1) Punch
        _punch, punch_dbg = detect_punch(state, RW, RS, dt)

        # 2) Kick
        _kick, kick_dbg = detect_kick(state, RH, RK, RA, dt)


    # --- Apply logic to both players ---
    if results_p1.pose_landmarks:
        process_fighter(results_p1.pose_landmarks.landmark, p1_state, dt)
        mp_drawing.draw_landmarks(
            image_p1, results_p1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 100, 0), thickness=2, circle_radius=2))

    if results_p2.pose_landmarks:
        process_fighter(results_p2.pose_landmarks.landmark, p2_state, dt)
        mp_drawing.draw_landmarks(
            image_p2, results_p2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 100, 200), thickness=2, circle_radius=2))

    prev_time = current_time

    # --- Display Dashboard (FIXED LAYOUT) ---
    image[:, :width//2] = image_p1
    image[:, width//2:] = image_p2
    cv2.line(image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    
    cv2.rectangle(image, (0, 0), (width, 110), (20, 20, 20), -1)
    
    margin = 15

    # --- Player 1 Info (Left Panel) ---
    # Title and Counters are left-aligned to the screen edge
    cv2.putText(image, 'FIGHTER 1 (BLUE)', (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
    cv2.putText(image, f'PUNCHES: {p1_state.right_punch_count}', (margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'KICKS: {p1_state.right_kick_count}', (margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # States are right-aligned to the center divider
    p1_arm_text = f'ARM: {p1_state.right_hand_state}'
    p1_leg_text = f'LEG: {p1_state.right_leg_state}'
    p1_arm_text_size = cv2.getTextSize(p1_arm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    p1_leg_text_size = cv2.getTextSize(p1_leg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    
    cv2.putText(image, p1_arm_text, (width//2 - p1_arm_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, p1_leg_text, (width//2 - p1_leg_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # --- Player 2 Info (Right Panel) ---
    # States are left-aligned to the center divider
    cv2.putText(image, f'ARM: {p2_state.right_hand_state}', (width//2 + margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'LEG: {p2_state.right_leg_state}', (width//2 + margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Title and Counters are right-aligned to the screen edge
    p2_title = 'FIGHTER 2 (RED)'
    p2_title_size = cv2.getTextSize(p2_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(image, p2_title, (width - p2_title_size[0] - margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)

    p2_punch_text = f'PUNCHES: {p2_state.right_punch_count}'
    p2_kick_text = f'KICKS: {p2_state.right_kick_count}'
    p2_punch_text_size = cv2.getTextSize(p2_punch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    p2_kick_text_size = cv2.getTextSize(p2_kick_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    
    cv2.putText(image, p2_punch_text, (width - p2_punch_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, p2_kick_text, (width - p2_kick_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('AI Boxing Coach - Sparring Mode', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()