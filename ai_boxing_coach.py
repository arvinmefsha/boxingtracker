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
        self.right_knee_count = 0
        self.prev_right_ankle = None
        self.prev_right_knee = None

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

        # Landmark definitions
        RIGHT_SHOULDER = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RIGHT_ELBOW = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        RIGHT_WRIST = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RIGHT_HIP = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        RIGHT_KNEE = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        RIGHT_ANKLE = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
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
    cv2.line(image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    
    cv2.rectangle(image, (0, 0), (width, 140), (20, 20, 20), -1)

    # Player 1 Info
    cv2.putText(image, 'FIGHTER 1 (BLUE)', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
    cv2.putText(image, f'PUNCHES: {p1_state.right_punch_count}', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'KICKS: {p1_state.right_kick_count}', (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'KNEES: {p1_state.right_knee_count}', (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'ARM: {p1_state.right_hand_state}', (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'LEG: {p1_state.right_leg_state}', (250, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Player 2 Info
    p2_start_x = width - 300
    cv2.putText(image, 'FIGHTER 2 (RED)', (p2_start_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
    cv2.putText(image, f'PUNCHES: {p2_state.right_punch_count}', (p2_start_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'KICKS: {p2_state.right_kick_count}', (p2_start_x, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'KNEES: {p2_state.right_knee_count}', (p2_start_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'ARM: {p2_state.right_hand_state}', (p2_start_x + 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'LEG: {p2_state.right_leg_state}', (p2_start_x + 200, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.imshow('AI Boxing Coach - Sparring Mode', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()