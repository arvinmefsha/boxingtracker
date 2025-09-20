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
        
        # --- MODIFICATION ---
        # Added state tracking for the left leg
        
        # Right Leg
        self.right_leg_state = 'IDLE' 
        self.right_kick_count = 0
        
        # Left Leg
        self.left_leg_state = 'IDLE'
        self.left_kick_count = 0


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


# --- FIX: Generalized detect_kick function ---
def detect_kick(state: FighterState,
                hip, ankle,
                side: str, # 'left' or 'right'
                idle_buffer=0.05):
    """
    Kick detection for a specified leg ('left' or 'right').
    - Counts a kick when the ankle goes above the hip.
    - Prevents jitter by requiring the ankle to go clearly *below* the hip + buffer before resetting to IDLE.
    """
    debug = {}
    did_kick = False

    hip_y = hip.y
    ankle_y = ankle.y

    # Dynamically get attributes based on the 'side' argument
    leg_state_attr = f'{side}_leg_state'
    kick_count_attr = f'{side}_kick_count'
    
    current_leg_state = getattr(state, leg_state_attr)

    # Kick start condition: ankle is above hip
    if ankle_y < hip_y:
        if current_leg_state == 'IDLE':
            # Increment the correct kick counter
            setattr(state, kick_count_attr, getattr(state, kick_count_attr) + 1)
            did_kick = True
            # Set the correct leg state to 'KICKING'
            setattr(state, leg_state_attr, 'KICKING')

    # Kick end condition with buffer to prevent flicker: ankle is clearly below hip
    elif ankle_y > hip_y + idle_buffer:  
        setattr(state, leg_state_attr, 'IDLE')
    
    debug.update({
        f"{side}_hip_y": hip_y,
        f"{side}_ankle_y": ankle_y,
        f"{side}_leg_state": getattr(state, leg_state_attr)
    })

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
    if not success or image is None:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    
    if image.shape[1] == 0:
        print("Warning: Frame has zero width.")
        continue

    height, width, _ = image.shape
    image_p1 = image[:, :width//2]
    image_p2 = image[:, width//2:]

    image_rgb_p1 = cv2.cvtColor(image_p1, cv2.COLOR_BGR2RGB)
    image_rgb_p2 = cv2.cvtColor(image_p2, cv2.COLOR_BGR2RGB)
    
    results_p1 = pose.process(image_rgb_p1)
    results_p2 = pose.process(image_rgb_p2)

    current_time = time.time()
    dt = current_time - prev_time
    
    if dt == 0:
        continue

    # --- Processing Logic (Unified for both fighters) ---
    def process_fighter(landmarks, state, dt):
        if not landmarks:
            return

        # Right Side Landmarks
        RS = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RW = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RH = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        RA = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # --- MODIFICATION: Get left side landmarks for kick detection ---
        LH = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        LA = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]


        # 1) Punch (remains right-hand only as per original logic)
        _punch, punch_dbg = detect_punch(state, RW, RS, dt)

        # 2) Kick (now detects both legs)
        # --- FIX: Calling the new generalized function for both legs ---
        _right_kick, right_kick_dbg = detect_kick(state, RH, RA, 'right')
        _left_kick, left_kick_dbg = detect_kick(state, LH, LA, 'left')


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

    # --- Display Dashboard ---
    image[:, :width//2] = image_p1
    image[:, width//2:] = image_p2
    
    cv2.rectangle(image, (0, 0), (width, 110), (20, 20, 20), -1)
    cv2.line(image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    
    margin = 15

    # --- MODIFICATION: Update UI to show total kicks and a combined leg state ---
    
    # Player 1 Info (Left Panel)
    total_kicks_p1 = p1_state.right_kick_count + p1_state.left_kick_count
    p1_leg_status = 'KICKING' if p1_state.right_leg_state == 'KICKING' or p1_state.left_leg_state == 'KICKING' else 'IDLE'

    cv2.putText(image, 'FIGHTER 1 (BLUE)', (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
    cv2.putText(image, f'PUNCHES: {p1_state.right_punch_count}', (margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'KICKS: {total_kicks_p1}', (margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    p1_arm_text = f'ARM: {p1_state.right_hand_state}'
    p1_leg_text = f'LEG: {p1_leg_status}'
    p1_arm_text_size = cv2.getTextSize(p1_arm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    p1_leg_text_size = cv2.getTextSize(p1_leg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    
    cv2.putText(image, p1_arm_text, (width//2 - p1_arm_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, p1_leg_text, (width//2 - p1_leg_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Player 2 Info (Right Panel)
    total_kicks_p2 = p2_state.right_kick_count + p2_state.left_kick_count
    p2_leg_status = 'KICKING' if p2_state.right_leg_state == 'KICKING' or p2_state.left_leg_state == 'KICKING' else 'IDLE'

    cv2.putText(image, f'ARM: {p2_state.right_hand_state}', (width//2 + margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'LEG: {p2_leg_status}', (width//2 + margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    p2_title = 'FIGHTER 2 (RED)'
    p2_title_size = cv2.getTextSize(p2_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(image, p2_title, (width - p2_title_size[0] - margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)

    p2_punch_text = f'PUNCHES: {p2_state.right_punch_count}'
    p2_kick_text = f'KICKS: {total_kicks_p2}'
    p2_punch_text_size = cv2.getTextSize(p2_punch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    p2_kick_text_size = cv2.getTextSize(p2_kick_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    
    cv2.putText(image, p2_punch_text, (width - p2_punch_text_size[0] - margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, p2_kick_text, (width - p2_kick_text_size[0] - margin, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('AI Boxing Coach - Sparring Mode', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
