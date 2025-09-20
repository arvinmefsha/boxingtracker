# fighter_state.py

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