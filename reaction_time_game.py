# reaction_time_game.py

import cv2
import mediapipe as mp
import time
import random

from game import Game

class ReactionTimeGame(Game):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_data = {'p1': None, 'p2': None}  # Store pose data
        
        self.state = 'IDLE'  # IDLE, WAITING, GREEN, RESULTS
        self.start_time = 0
        self.green_time = 0
        self.delay = 0
        self.p1_time = None
        self.p2_time = None
        self.first_reaction_time = None  # Time when first player reacted
        self.reaction_timeout = 1  # 1 second after first reaction
        self.p1_cheated = False
        self.p2_cheated = False
        self.p1_wins = 0
        self.p2_wins = 0
        self.target_wins = 3  # First to 3 wins
        self.round_completed = False  # Flag to prevent re-incrementing wins

    def reset(self):
        """Reset game state."""
        self.state = 'IDLE'
        self.start_time = 0
        self.green_time = 0
        self.delay = 0
        self.p1_time = None
        self.p2_time = None
        self.first_reaction_time = None
        self.p1_cheated = False
        self.p2_cheated = False
        self.round_completed = False
        self.pose_data = {'p1': None, 'p2': None}

    def reset_match(self):
        """Reset the entire match (wins to 0)."""
        self.p1_wins = 0
        self.p2_wins = 0
        self.reset()

    def handle_key(self, key):
        """Handle keyboard input for the game."""
        if key == 32:  # Spacebar
            if self.state == 'IDLE':
                if self.p1_wins >= self.target_wins or self.p2_wins >= self.target_wins:
                    self.reset_match()  # Start new match if previous ended
                else:
                    self.p1_time = None
                    self.p2_time = None
                    self.first_reaction_time = None
                    self.p1_cheated = False
                    self.p2_cheated = False
                    self.round_completed = False
                    self.state = 'WAITING'
                    self.delay = random.uniform(3, 5)
                    self.start_time = time.time()
            elif self.state == 'RESULTS':
                if self.p1_wins >= self.target_wins or self.p2_wins >= self.target_wins:
                    self.reset_match()  # Start new match
                else:
                    # Start next round
                    self.p1_time = None
                    self.p2_time = None
                    self.first_reaction_time = None
                    self.p1_cheated = False
                    self.p2_cheated = False
                    self.round_completed = False
                    self.state = 'WAITING'
                    self.delay = random.uniform(3, 5)
                    self.start_time = time.time()
            elif self.state == 'GREEN':
                # Optional: reset if pressed during green
                self.reset()

    def determine_round_winner(self):
        """Determine winner of the current round and update win counts."""
        if self.round_completed:
            return None  # Already processed
        
        if self.p1_cheated and self.p2_cheated:
            self.round_completed = True
            return None  # Tie, no win
        elif self.p1_cheated:
            self.p2_wins += 1
            self.round_completed = True
            return 2
        elif self.p2_cheated:
            self.p1_wins += 1
            self.round_completed = True
            return 1
        elif self.p1_time is None and self.p2_time is None:
            self.round_completed = True
            return None  # Incomplete, no win
        elif self.p1_time is None:
            self.p2_wins += 1
            self.round_completed = True
            return 2
        elif self.p2_time is None:
            self.p1_wins += 1
            self.round_completed = True
            return 1
        else:
            if self.p1_time < self.p2_time:
                self.p1_wins += 1
                self.round_completed = True
                return 1
            elif self.p2_time < self.p1_time:
                self.p2_wins += 1
                self.round_completed = True
                return 2
            else:
                self.round_completed = True
                return None  # Tie, no win

    def is_raised(self, landmarks):
        """Check if both hands are raised above shoulders."""
        if not landmarks:
            return False
        
        lm = landmarks
        l_sh_y = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        r_sh_y = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        avg_sh_y = (l_sh_y + r_sh_y) / 2
        
        l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        if l_wr.visibility > 0.5 and r_wr.visibility > 0.5:
            if l_wr.y < avg_sh_y and r_wr.y < avg_sh_y:
                return True
        return False

    def update(self, pose_data, dt):
        """Update game state."""
        self.pose_data = pose_data
        current_time = time.time()
        
        if self.state == 'WAITING':
            # Check for cheating (hands raised during waiting)
            p1_raised = self.is_raised(pose_data['p1'].pose_landmarks.landmark if pose_data['p1'] and pose_data['p1'].pose_landmarks else None)
            p2_raised = self.is_raised(pose_data['p2'].pose_landmarks.landmark if pose_data['p2'] and pose_data['p2'].pose_landmarks else None)
            
            if p1_raised:
                self.p1_cheated = True
                self.state = 'RESULTS'
                return  # End immediately
            if p2_raised:
                self.p2_cheated = True
                self.state = 'RESULTS'
                return  # End immediately
            
            if current_time - self.start_time >= self.delay:
                self.state = 'GREEN'
                self.green_time = current_time
        
        elif self.state == 'GREEN':
            # Check Player 1
            if self.p1_time is None and self.is_raised(pose_data['p1'].pose_landmarks.landmark if pose_data['p1'] and pose_data['p1'].pose_landmarks else None):
                self.p1_time = current_time - self.green_time
                if self.first_reaction_time is None:
                    self.first_reaction_time = current_time
            
            # Check Player 2
            if self.p2_time is None and self.is_raised(pose_data['p2'].pose_landmarks.landmark if pose_data['p2'] and pose_data['p2'].pose_landmarks else None):
                self.p2_time = current_time - self.green_time
                if self.first_reaction_time is None:
                    self.first_reaction_time = current_time
            
            # End immediately if both have reacted
            if self.p1_time is not None and self.p2_time is not None:
                self.state = 'RESULTS'
            # Or timeout after first reaction if only one has reacted
            elif self.first_reaction_time is not None and current_time - self.first_reaction_time > self.reaction_timeout:
                self.state = 'RESULTS'
        
        elif self.state == 'RESULTS':
            # Determine winner only once per round
            self.determine_round_winner()

    def render(self, frame):
        """Render the reaction time game visuals."""
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

        # Draw status bar at top
        if self.state in ['IDLE', 'WAITING', 'RESULTS']:
            bar_color = (0, 0, 255)  # Red
        else:  # GREEN
            bar_color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (0, 0), (1920, 50), bar_color, -1)

        # Display instructions or results
        if self.state == 'IDLE':
            cv2.putText(frame, "Press Space to Start", (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        elif self.state == 'WAITING':
            # No countdown display, just red bar
            pass
        elif self.state == 'GREEN':
            # Green bar
            pass
        elif self.state == 'RESULTS':
            round_winner = self.determine_round_winner() if not self.round_completed else None
            match_over = self.p1_wins >= self.target_wins or self.p2_wins >= self.target_wins
            
            if match_over:
                if self.p1_wins >= self.target_wins:
                    final_text = "Player 1 Wins the Match!"
                elif self.p2_wins >= self.target_wins:
                    final_text = "Player 2 Wins the Match!"
                cv2.putText(frame, final_text, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Final Score: P1 {self.p1_wins} - P2 {self.p2_wins}", (650, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press Space for New Match", (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
            else:
                if round_winner == 1:
                    round_text = "Player 1 Wins Round!"
                elif round_winner == 2:
                    round_text = "Player 2 Wins Round!"
                else:
                    round_text = "Tie Round!"
                cv2.putText(frame, round_text, (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Score: P1 {self.p1_wins} - P2 {self.p2_wins}", (700, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if self.p1_cheated and self.p2_cheated:
                    p1_text = "Cheated"
                    p2_text = "Cheated"
                elif self.p1_cheated:
                    p1_text = "Cheated"
                    p2_text = f"Time: {self.p2_time:.2f}s" if self.p2_time is not None else "No Reaction"
                elif self.p2_cheated:
                    p1_text = f"Time: {self.p1_time:.2f}s" if self.p1_time is not None else "No Reaction"
                    p2_text = "Cheated"
                elif self.p1_time is None and self.p2_time is None:
                    p1_text = "No Reaction"
                    p2_text = "No Reaction"
                elif self.p1_time is None:
                    p1_text = "No Reaction"
                    p2_text = f"Time: {self.p2_time:.2f}s"
                elif self.p2_time is None:
                    p1_text = f"Time: {self.p1_time:.2f}s"
                    p2_text = "No Reaction"
                else:
                    p1_text = f"Time: {self.p1_time:.2f}s"
                    p2_text = f"Time: {self.p2_time:.2f}s"
                
                cv2.putText(frame, p1_text, (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, p2_text, (1300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press Space for Next Round", (700, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

        return frame