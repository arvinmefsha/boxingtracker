import cv2
import mediapipe as mp
import time
import math
import pygame
import os

from game import Game
from helpers import calculate_distance, calculate_velocity, calculate_angle
from fighter_state import FighterState

class BoxingGame(Game):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_data = {'p1': None, 'p2': None}

        self.p1_state = FighterState()
        self.p2_state = FighterState()

        self.p1_hearts = 12
        self.p2_hearts = 12

        self.prev_time = time.time()

        self.pending_attack = {'p1': None, 'p2': None}
        self.stun_until = {'p1': 0.0, 'p2': 0.0}
        self.last_stunned_by = {'p1': None, 'p2': None}
        self.last_attack_time = {'p1': 0.0, 'p2': 0.0}
        self.global_cooldown_s = 0.5

        self.game_over = False

        try:
            pygame.mixer.init()
            punch_path = os.path.join('assets', 'sounds', 'punch.mp3')
            kick_path = os.path.join('assets', 'sounds', 'kick.mp3')
            weave_path = os.path.join('assets', 'sounds', 'weave.mp3')
            music_path = os.path.join('assets', 'sounds', 'boxing.mp3')

            self.punch_sound = pygame.mixer.Sound(punch_path) if os.path.exists(punch_path) else None
            self.kick_sound = pygame.mixer.Sound(kick_path) if os.path.exists(kick_path) else None
            self.woosh_sound = pygame.mixer.Sound(weave_path) if os.path.exists(weave_path) else None

            if self.punch_sound: self.punch_sound.set_volume(1)
            if self.kick_sound: self.kick_sound.set_volume(1)
            if self.woosh_sound: self.woosh_sound.set_volume(1)

            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.25)
                pygame.mixer.music.play(-1)  

        except Exception as e:
            print(f"Sound init failed: {e}. Running without sound.")
            self.punch_sound = None
            self.kick_sound = None
            self.woosh_sound = None

    def reset(self):
        self.p1_state = FighterState()
        self.p2_state = FighterState()

        self.prev_time = time.time()
        self.pose_data = {'p1': None, 'p2': None}

        self.p1_hearts = 9
        self.p2_hearts = 9

        self.pending_attack = {'p1': None, 'p2': None}
        self.stun_until = {'p1': 0.0, 'p2': 0.0}
        self.last_stunned_by = {'p1': None, 'p2': None}
        self.last_attack_time = {'p1': 0.0, 'p2': 0.0}
        self.game_over = False

    def _ensure_hand_fields(self, state, side: str):
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

    def _ensure_weave_fields(self, state):
        if not hasattr(state, 'weave_state'):
            state.weave_state = 'IDLE'
        if not hasattr(state, 'weave_count'):
            state.weave_count = 0

    def _other(self, who: str) -> str:
        return 'p2' if who == 'p1' else 'p1'

    def _can_act(self, who: str) -> bool:
        return time.time() >= self.stun_until[who]

    def _on_cooldown(self, who: str) -> bool:
        return (time.time() - self.last_attack_time[who]) < self.global_cooldown_s

    def _mark_attack_time(self, who: str):
        self.last_attack_time[who] = time.time()

    def _start_attack(self, attacker: str, attack_type: str):
        now = time.time()
        defender = self._other(attacker)
        self.pending_attack[attacker] = {
            'type': attack_type,
            'time': now,
            'defender': defender
        }
        self._mark_attack_time(attacker)

    def _resolve_expired_attacks(self):
        now = time.time()
        for attacker in ('p1', 'p2'):
            pa = self.pending_attack[attacker]
            if pa is None:
                continue
            if now - pa['time'] >= 1.25:
                defender = pa['defender']
                dmg = 3 if pa['type'] == 'kick' else 1
                if defender == 'p1':
                    self.p1_hearts = max(0, self.p1_hearts - dmg)
                else:
                    self.p2_hearts = max(0, self.p2_hearts - dmg)
                if pa['type'] == 'kick' and self.kick_sound:
                    try:
                        self.kick_sound.play()
                    except Exception:
                        pass
                elif pa['type'] == 'punch' and self.punch_sound:
                    try:
                        self.punch_sound.play()
                    except Exception:
                        pass
                self.pending_attack[attacker] = None

    def _try_end_stun_early_on_counter(self, counterattacker: str):
        opponent = self._other(counterattacker)
        if self.last_stunned_by[opponent] == counterattacker:
            self.stun_until[opponent] = time.time()
            self.last_stunned_by[opponent] = None

    def _handle_weave_event(self, weaver: str):
        opponent = self._other(weaver)
        pa = self.pending_attack[opponent]
        if pa is None:
            return 
        dt = time.time() - pa['time']
        if dt < 0.75:
            self.stun_until[opponent] = time.time() + 0.5
            self.last_stunned_by[opponent] = weaver
            self.pending_attack[opponent] = None
            if self.woosh_sound:
                try:
                    self.woosh_sound.play()
                except Exception:
                    pass
        elif 0.75 <= dt <= 1.25:
            self.pending_attack[opponent] = None
        else:
            pass

    def detect_punch(
        self,
        state,
        shoulder, elbow, wrist,    
        other_wrist,              
        hip,                     
        waist_y,                 
        side: str,
        angle_threshold: float = 150.0,
        shoulder_min_angle: float = 45.0,
        cooldown_s: float = 0.25
    ):

        debug = {}
        did_punch = False

        self._ensure_hand_fields(state, side)

        hand_state_attr = f'{side}_hand_state'
        punch_count_attr = f'{side}_punch_count'
        last_punch_ts_attr = f'last_{side}_punch_ts'

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

        both_hands_above_waist = (wrist.y < waist_y) and (other_wrist.y < waist_y)

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

    def detect_kick(self, state, hip, ankle, side: str, idle_buffer=0.05):
        debug = {}
        did_kick = False

        self._ensure_leg_fields(state, side)
        leg_state_attr = f'{side}_leg_state'
        kick_count_attr = f'{side}_kick_count'

        hip_y = hip.y
        ankle_y = ankle.y

        current_leg_state = getattr(state, leg_state_attr)

        if ankle_y < hip_y:
            if current_leg_state == 'IDLE':
                setattr(state, kick_count_attr, getattr(state, kick_count_attr) + 1)
                did_kick = True
                setattr(state, leg_state_attr, 'KICKING')

        elif ankle_y > hip_y + idle_buffer:
            setattr(state, leg_state_attr, 'IDLE')

        debug.update({
            f"{side}_hip_y": hip_y,
            f"{side}_ankle_y": ankle_y,
            f"{side}_leg_state": getattr(state, leg_state_attr)
        })

        return did_kick, debug

    def detect_weave(self, state, LH, LS, RH, RS, angle_from_vertical_thresh: float = 45.0, hysteresis: float = 3.0):
        debug = {}
        did_weave = False
        self._ensure_weave_fields(state)

        def tilt_from_vertical(hip, shoulder):
            dx = shoulder.x - hip.x
            dy = shoulder.y - hip.y
            angle_rad = math.atan2(abs(dx), abs(dy) + 1e-8)
            return math.degrees(angle_rad)

        left_tilt = tilt_from_vertical(LH, LS)
        right_tilt = tilt_from_vertical(RH, RS)
        max_tilt = max(left_tilt, right_tilt)

        debug.update({
            "left_tilt_from_vertical_deg": left_tilt,
            "right_tilt_from_vertical_deg": right_tilt,
            "max_tilt_deg": max_tilt,
            "weave_state": state.weave_state
        })

        tilting_now = (max_tilt > angle_from_vertical_thresh)

        if state.weave_state == 'IDLE':
            if tilting_now:
                state.weave_count = getattr(state, 'weave_count', 0) + 1
                state.weave_state = 'WEAVING'
                did_weave = True
        else:
            if max_tilt < (angle_from_vertical_thresh - hysteresis):
                state.weave_state = 'IDLE'

        return did_weave, debug

    def process_fighter(self, who: str, landmarks, state, dt):
        if not landmarks or self.game_over:
            return

        RS = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        RE = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        RW = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        RH = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        RK = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        RA = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        LS = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        LE = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        LW = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        LH = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        LK = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        LA = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        waist_y = (LH.y + RH.y) / 2.0

        did_r_punch, _ = self.detect_punch(state, RS, RE, RW, LW, RH, waist_y, 'right')
        did_l_punch, _ = self.detect_punch(state, LS, LE, LW, RW, LH, waist_y, 'left')
        did_any_punch = did_r_punch or did_l_punch

        if did_any_punch and self._can_act(who) and not self._on_cooldown(who):
            self._start_attack(who, 'punch')
            self._try_end_stun_early_on_counter(who)

        did_r_kick, _ = self.detect_kick(state, RH, RA, 'right')
        did_l_kick, _ = self.detect_kick(state, LH, LA, 'left')
        did_any_kick = did_r_kick or did_l_kick

        if did_any_kick and self._can_act(who) and not self._on_cooldown(who):
            self._start_attack(who, 'kick')
            self._try_end_stun_early_on_counter(who)

        did_weave, _ = self.detect_weave(state, LH, LS, RH, RS)
        if did_weave and self._can_act(who):
            self._handle_weave_event(who)

    def handle_input(self, pose_data):
        self.pose_data = pose_data
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt == 0:
            return

        self.prev_time = current_time

        if pose_data['p1'] and pose_data['p1'].pose_landmarks:
            self.process_fighter('p1', pose_data['p1'].pose_landmarks.landmark, self.p1_state, dt)

        if pose_data['p2'] and pose_data['p2'].pose_landmarks:
            self.process_fighter('p2', pose_data['p2'].pose_landmarks.landmark, self.p2_state, dt)

    def update(self, pose_data, dt):
        if self.game_over:
            return

        self.handle_input(pose_data)

        self._resolve_expired_attacks()

        if self.p1_hearts <= 0 or self.p2_hearts <= 0:
            self.game_over = True

    def render(self, frame):
        height, width, _ = frame.shape
        half_width = width // 2
        image_p1 = frame[:, :half_width]
        image_p2 = frame[:, half_width:]

        if self.pose_data['p1'] and self.pose_data['p1'].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_p1, self.pose_data['p1'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(200, 100, 0), thickness=2, circle_radius=2)
            )

        if self.pose_data['p2'] and self.pose_data['p2'].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_p2, self.pose_data['p2'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 100, 200), thickness=2, circle_radius=2)
            )

        sx = width / 1920.0
        sy = height / 1080.0
        s = min(sx, sy)

        dashboard_h = max(100, int(190 * sy))

        margin = max(8, int(15 * sx))
        title_font_scale = 0.7 * s
        value_font_scale = 0.9 * s
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = max(1, int(2 * s))
        line_thick = max(2, int(2 * s))

        y_title = int(25 * sy)
        y_line1 = int(60 * sy)
        y_line2 = int(95 * sy)
        y_line3 = int(130 * sy)
        y_line4 = int(165 * sy)

        cv2.rectangle(frame, (0, 0), (width, dashboard_h), (20, 20, 20), -1)
        center_x = width // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), line_thick)

        cv2.putText(frame, 'FIGHTER 1 (BLUE)', (margin, y_title), font, title_font_scale, (255, 150, 150), thick)

        p1_total_punches = getattr(self.p1_state, 'right_punch_count', 0) + getattr(self.p1_state, 'left_punch_count', 0)
        p1_total_kicks = getattr(self.p1_state, 'right_kick_count', 0) + getattr(self.p1_state, 'left_kick_count', 0)
        p1_weaves = getattr(self.p1_state, 'weave_count', 0)

        cv2.putText(frame, f'PUNCHES: {p1_total_punches}', (margin, y_line1), font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, f'KICKS: {p1_total_kicks}', (margin, y_line2), font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, f'WEAVES: {p1_weaves}', (margin, y_line3), font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, f'HEARTS: {self.p1_hearts}', (margin, y_line4), font, value_font_scale, (255, 100, 100), thick)

        p1_arm_combined = 'PUNCHING' if (
            getattr(self.p1_state, 'right_hand_state', 'IDLE') == 'PUNCHING' or
            getattr(self.p1_state, 'left_hand_state',  'IDLE') == 'PUNCHING'
        ) else 'IDLE'
        p1_leg_combined = 'KICKING' if (
            getattr(self.p1_state, 'right_leg_state', 'IDLE') == 'KICKING' or
            getattr(self.p1_state, 'left_leg_state',  'IDLE') == 'KICKING'
        ) else 'IDLE'

        cv2.putText(frame, f'ARM: {p1_arm_combined}', (center_x - 150, y_line1), font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, f'LEG: {p1_leg_combined}', (center_x - 150, y_line2), font, value_font_scale, (255, 255, 255), thick)

        p2_arm_combined = 'PUNCHING' if (
            getattr(self.p2_state, 'right_hand_state', 'IDLE') == 'PUNCHING' or
            getattr(self.p2_state, 'left_hand_state',  'IDLE') == 'PUNCHING'
        ) else 'IDLE'
        p2_leg_combined = 'KICKING' if (
            getattr(self.p2_state, 'right_leg_state', 'IDLE') == 'KICKING' or
            getattr(self.p2_state, 'left_leg_state',  'IDLE') == 'KICKING'
        ) else 'IDLE'

        cv2.putText(frame, f'ARM: {p2_arm_combined}', (center_x + margin, y_line1),
                    font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, f'LEG: {p2_leg_combined}', (center_x + margin, y_line2),
                    font, value_font_scale, (255, 255, 255), thick)

        p2_title = 'FIGHTER 2 (RED)'
        p2_title_sz = cv2.getTextSize(p2_title, font, title_font_scale, thick)[0]
        cv2.putText(frame, p2_title, (width - p2_title_sz[0] - margin, y_title),
                    font, title_font_scale, (150, 150, 255), thick)

        p2_total_punches = getattr(self.p2_state, 'right_punch_count', 0) + getattr(self.p2_state, 'left_punch_count', 0)
        p2_total_kicks   = getattr(self.p2_state, 'right_kick_count', 0) + getattr(self.p2_state, 'left_kick_count', 0)
        p2_weaves        = getattr(self.p2_state, 'weave_count', 0)

        p2_punch_text = f'PUNCHES: {p2_total_punches}'
        p2_kick_text  = f'KICKS: {p2_total_kicks}'
        p2_weave_text = f'WEAVES: {p2_weaves}'
        p2_hearts_text = f'HEARTS: {self.p2_hearts}'

        p2_punch_sz = cv2.getTextSize(p2_punch_text, font, value_font_scale, thick)[0]
        p2_kick_sz  = cv2.getTextSize(p2_kick_text,  font, value_font_scale, thick)[0]
        p2_weave_sz = cv2.getTextSize(p2_weave_text,  font, value_font_scale, thick)[0]
        p2_hearts_sz = cv2.getTextSize(p2_hearts_text, font, value_font_scale, thick)[0]

        cv2.putText(frame, p2_punch_text, (width - p2_punch_sz[0] - margin, y_line1),
                    font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, p2_kick_text, (width - p2_kick_sz[0] - margin, y_line2),
                    font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, p2_weave_text, (width - p2_weave_sz[0] - margin, y_line3),
                    font, value_font_scale, (255, 255, 255), thick)
        cv2.putText(frame, p2_hearts_text, (width - p2_hearts_sz[0] - margin, y_line4),
                    font, value_font_scale, (255, 100, 100), thick)

        if self.game_over:
            go_txt = "GAME OVER"
            size = cv2.getTextSize(go_txt, font, 1.5 * s, max(2, int(4 * s)))[0]
            cv2.putText(frame, go_txt, (center_x - size[0] // 2, int(0.5 * height)),
                        font, 1.5 * s, (0, 255, 255), max(2, int(4 * s)))

        return frame
