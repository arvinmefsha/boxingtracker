# fruit_ninja_game.py

import cv2
import mediapipe as mp
import time
import random
import numpy as np
import os
import sys

import pygame
from collections import deque

from game import Game
from helpers import calculate_velocity, line_circle_intersection

def overlay_transparent(background, overlay, x, y):
    if overlay.shape[2] < 4: return background
    bg_h, bg_w, _ = background.shape
    img_h, img_w, _ = overlay.shape
    x_pos, y_pos = x - img_w // 2, y - img_h // 2
    x_start_bg, y_start_bg = max(x_pos, 0), max(y_pos, 0)
    x_end_bg, y_end_bg = min(x_pos + img_w, bg_w), min(y_pos + img_h, bg_h)
    x_start_overlay, y_start_overlay = max(0, -x_pos), max(0, -y_pos)
    x_end_overlay, y_end_overlay = x_start_overlay + (x_end_bg - x_start_bg), y_start_overlay + (y_end_bg - y_start_bg)
    if (x_end_bg <= x_start_bg) or (y_end_bg <= y_start_bg): return background
    roi = background[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
    overlay_clipped = overlay[y_start_overlay:y_end_overlay, x_start_overlay:x_end_overlay]
    alpha = overlay_clipped[:, :, 3] / 255.0
    mask = np.dstack([alpha, alpha, alpha])
    overlay_rgb = overlay_clipped[:, :, :3]
    background_part = roi * (1 - mask)
    overlay_part = overlay_rgb * mask
    background[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = background_part + overlay_part
    return background

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if image.shape[2] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3]
        rotated_rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        rotated_alpha = cv2.warpAffine(alpha, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.merge([rotated_rgb[:,:,0], rotated_rgb[:,:,1], rotated_rgb[:,:,2], rotated_alpha])
    else:
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

class FruitNinjaGame(Game):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_data = {'p1': None, 'p2': None}
        
        self.gravity = 1.0
        self.spawn_interval = 0.9
        self.slice_velocity_threshold = 1500.0
        
        self.image_size = (200, 200) 
        self.object_radius = self.image_size[0] // 2

        print("Loading assets...")
        self.fruit_images = self.load_fruit_images('assets')
        self.bomb_image = cv2.imread(os.path.join('assets', 'bomb.png'), cv2.IMREAD_UNCHANGED)
        
        if self.bomb_image is None:
            print("\n--- ERROR: Failed to load 'bomb.png'. Make sure it's in the 'assets' folder. ---\n")
            sys.exit(1)
        self.bomb_image = cv2.resize(self.bomb_image, self.image_size, interpolation=cv2.INTER_AREA)

        self.stopwatch_image = cv2.imread(os.path.join('assets', 'stopwatch.png'), cv2.IMREAD_UNCHANGED)
        if self.stopwatch_image is None:
            print("\n--- ERROR: Failed to load 'stopwatch.png'. Make sure it's in the 'assets' folder. ---\n")
            sys.exit(1)
        self.stopwatch_image = cv2.resize(self.stopwatch_image, (60, 60), interpolation=cv2.INTER_AREA)

        print("Assets loaded successfully.")

        print("Initializing sound...")
        pygame.mixer.init()
        try:
            self.theme_sound = pygame.mixer.Sound('assets/sounds/fruitninjatheme.mp3')
            self.slice_sound = pygame.mixer.Sound('assets/sounds/slice.mp3')
            self.bomb_sound = pygame.mixer.Sound('assets/sounds/explosion.mp3')
            self.theme_sound.set_volume(0.3)
            self.slice_sound.set_volume(0.3)
            self.bomb_sound.set_volume(0.7)
            print("Sound loaded successfully.")
        except pygame.error as e:
            print(f"\n--- Sound Error: {e} --- \nGame will run without sound.")
            self.theme_sound, self.slice_sound, self.bomb_sound = None, None, None

        self.game_duration = 30 # seconds
        self.game_start_time = None
        self.game_over = False
        self.time_remaining = self.game_duration


        if self.theme_sound:
            self.theme_sound.play(loops=-1) 
        self.reset()

    def load_fruit_images(self, path):
        images = {}
        fruit_types = ['apple', 'watermelon', 'banana']
        for fruit in fruit_types:
            images[fruit] = {}
            for part in ['whole', 'left', 'right']:
                suffix = '' if part == 'whole' else ('_l' if part == 'left' else '_r')
                file_path = os.path.join(path, f'{fruit}{suffix}.png')
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"\n--- ERROR: Failed to load image: {os.path.basename(file_path)} ---\n")
                    sys.exit(1)
                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                images[fruit][part] = img
        return images

    def reset(self):
        self.p1_score, self.p2_score = 0, 0
        self.p1_fruits, self.p2_fruits = [], []
        self.p1_bombs, self.p2_bombs = [], []
        self.p1_text_effects, self.p2_text_effects = [], []
        self.p1_sliced_pieces, self.p2_sliced_pieces = [], []
        self.p1_particles, self.p2_particles = [], []
        self.p1_feedback, self.p2_feedback = "", ""
        self.p1_prev_left_wrist_screen, self.p1_prev_right_wrist_screen = None, None
        self.p2_prev_left_wrist_screen, self.p2_prev_right_wrist_screen = None, None
        self.p1_last_spawn, self.p2_last_spawn = time.time(), time.time()
        self.p1_left_trail, self.p1_right_trail = deque(maxlen=20), deque(maxlen=20)
        self.p2_left_trail, self.p2_right_trail = deque(maxlen=20), deque(maxlen=20)
        self.prev_time = time.time()
        self.game_start_time = time.time()
        self.game_over = False

    def spawn_object(self, is_fruit=True):
        vx = random.uniform(-0.3, 0.3)
        vy = random.uniform(-1.2, -0.9) 
        obj = {'x': random.uniform(0.2, 0.8), 'y': 1.1, 'vx': vx, 'vy': vy, 'radius': self.object_radius}
        if is_fruit:
            fruit_type = random.choice(list(self.fruit_images.keys()))
            obj.update({'type': 'fruit', 'images': self.fruit_images[fruit_type]})
        else:
            obj.update({'type': 'bomb', 'image': self.bomb_image})
        return obj
    def update_objects(self, objects, dt):
        new_objects = []
        for obj in objects:
            obj['x'] += obj['vx'] * dt
            obj['y'] += obj['vy'] * dt
            obj['vy'] += self.gravity * dt
            if 'rotation_speed' in obj: obj['angle'] = (obj.get('angle', 0) + obj['rotation_speed'] * dt) % 360
            if 'life' in obj:
                obj['life'] -= 1
                if obj['life'] <= 0: continue
            if obj['y'] < 1.3: new_objects.append(obj)
        return new_objects
    def update_text_effects(self, text_effects):
        updated_effects = []
        for effect in text_effects:
            effect['life'] -= 1
            if effect['life'] > 0:
                half_life = effect['max_life'] / 2
                progress = (effect['max_life'] - effect['life']) / half_life if effect['life'] > half_life else effect['life'] / half_life
                effect['font_scale'] = effect['min_scale'] + (effect['max_scale'] - effect['min_scale']) * progress
                updated_effects.append(effect)
        return updated_effects
    def check_slice(self, prev_pos, curr_pos, objects, half_width, height):
        sliced = []
        if prev_pos is None or curr_pos is None: return sliced
        start, end = tuple(prev_pos), tuple(curr_pos)
        for i, obj in enumerate(objects):
            if not (0 <= obj['y'] <= 1.0): continue
            center = (obj['x'] * half_width, obj['y'] * height)
            if line_circle_intersection(start, end, center, obj['radius']): sliced.append(i)
        return sliced
    def process_slicing_for_player(self, player_data):
        pose, prev_wrists, fruits, bombs, sliced_pieces, particles, text_effects, trails, score, feedback_text = player_data
        half_width, height = 960, 1080
        landmarks = pose.pose_landmarks.landmark
        wrists = {'left': [landmarks[15].x * half_width, landmarks[15].y * height],
                  'right': [landmarks[16].x * half_width, landmarks[16].y * height]}
        trails['left'].append(wrists['left'])
        trails['right'].append(wrists['right'])
        for hand in ['left', 'right']:
            if prev_wrists[hand]:
                velocity = calculate_velocity(wrists[hand], prev_wrists[hand], self.dt)
                if velocity > self.slice_velocity_threshold:
                    slice_angle_rad = np.arctan2(-(wrists[hand][1] - prev_wrists[hand][1]), wrists[hand][0] - prev_wrists[hand][0])
                    slice_angle_deg = np.degrees(slice_angle_rad)
                    sliced_fruits = self.check_slice(prev_wrists[hand],  wrists[hand], fruits, half_width, height)
                    if sliced_fruits and self.slice_sound: self.slice_sound.play()
                    for i in sorted(sliced_fruits, reverse=True):
                        fruit = fruits.pop(i)
                        score += 1
                        feedback_text = "Sliced!"
                        text_effects.append({'text': '+1', 'x': fruit['x'], 'y': fruit['y'], 'life': 30, 'max_life': 30, 'font_scale': 0, 'min_scale': 1, 'max_scale': 2, 'color': (255, 255, 255)})
                        separation_speed = 0.5 
                        left_vx, left_vy = separation_speed * np.cos(slice_angle_rad + np.pi/2), separation_speed * np.sin(slice_angle_rad + np.pi/2)
                        right_vx, right_vy = separation_speed * np.cos(slice_angle_rad - np.pi/2), separation_speed * np.sin(slice_angle_rad - np.pi/2)
                        piece1 = {'x': fruit['x'], 'y': fruit['y'], 'vx': fruit['vx'] + left_vx, 'vy': fruit['vy'] + left_vy, 'life': 60, 'image': rotate_image(fruit['images']['left'], slice_angle_deg), 'rotation_speed': random.uniform(-100, 100)}
                        piece2 = {'x': fruit['x'], 'y': fruit['y'], 'vx': fruit['vx'] + right_vx, 'vy': fruit['vy'] + right_vy, 'life': 60, 'image': rotate_image(fruit['images']['right'], slice_angle_deg), 'rotation_speed': random.uniform(-100, 100)}
                        sliced_pieces.extend([piece1, piece2])
                    bomb_sliced = self.check_slice(prev_wrists[hand], wrists[hand], bombs, half_width, height)
                    if bomb_sliced and self.bomb_sound: self.bomb_sound.play()
                    for i in sorted(bomb_sliced, reverse=True):
                        bomb = bombs.pop(i)
                        score -= 5
                        feedback_text = "Boom!"
                        text_effects.append({'text': '-5', 'x': bomb['x'], 'y': bomb['y'], 'life': 30, 'max_life': 30, 'font_scale': 0, 'min_scale': 1.5, 'max_scale': 3, 'color': (8, 15, 207)})
                        for _ in range(30):
                            angle, speed = random.uniform(0, 2*np.pi), random.uniform(1, 4)
                            particles.append({'x': bomb['x'], 'y': bomb['y'], 'vx': np.cos(angle)*speed, 'vy': np.sin(angle)*speed, 'life': 30, 'radius': random.randint(3,8), 'color': random.choice([(0,0,255), (0,165,255), (0,255,255)])})
        return wrists, score, feedback_text
    
    def handle_input(self, pose_data):
        current_time = time.time()
        self.dt = current_time - self.prev_time
        if self.dt == 0: return
        self.prev_time = current_time

        if not self.game_over:
            elapsed_time = current_time - self.game_start_time
            self.time_remaining = self.game_duration - elapsed_time
            if self.time_remaining <= 0:
                self.time_remaining = 0
                self.game_over = True
                print("Game Over!")

        if not self.game_over:
            if current_time - self.p1_last_spawn > self.spawn_interval:
                obj = self.spawn_object(random.random() < 0.85)
                (self.p1_fruits if obj['type'] == 'fruit' else self.p1_bombs).append(obj)
                self.p1_last_spawn = current_time
            if current_time - self.p2_last_spawn > self.spawn_interval:
                obj = self.spawn_object(random.random() < 0.85)
                (self.p2_fruits if obj['type'] == 'fruit' else self.p2_bombs).append(obj)
                self.p2_last_spawn = current_time
            
            if pose_data['p1'] and pose_data['p1'].pose_landmarks:
                p1_trails = {'left': self.p1_left_trail, 'right': self.p1_right_trail}
                wrists, self.p1_score, self.p1_feedback = self.process_slicing_for_player([pose_data['p1'], {'left': self.p1_prev_left_wrist_screen, 'right': self.p1_prev_right_wrist_screen}, self.p1_fruits, self.p1_bombs, self.p1_sliced_pieces, self.p1_particles, self.p1_text_effects, p1_trails, self.p1_score, self.p1_feedback])
                self.p1_prev_left_wrist_screen, self.p1_prev_right_wrist_screen = wrists['left'], wrists['right']
            if pose_data['p2'] and pose_data['p2'].pose_landmarks:
                p2_trails = {'left': self.p2_left_trail, 'right': self.p2_right_trail}
                wrists, self.p2_score, self.p2_feedback = self.process_slicing_for_player([pose_data['p2'], {'left': self.p2_prev_left_wrist_screen, 'right': self.p2_prev_right_wrist_screen}, self.p2_fruits, self.p2_bombs, self.p2_sliced_pieces, self.p2_particles, self.p2_text_effects, p2_trails, self.p2_score, self.p2_feedback])
                self.p2_prev_left_wrist_screen, self.p2_prev_right_wrist_screen = wrists['left'], wrists['right']

        for lst in ['p1_fruits', 'p1_bombs', 'p1_sliced_pieces', 'p1_particles', 'p2_fruits', 'p2_bombs', 'p2_sliced_pieces', 'p2_particles']:
            setattr(self, lst, self.update_objects(getattr(self, lst), self.dt))
        self.p1_text_effects = self.update_text_effects(self.p1_text_effects)
        self.p2_text_effects = self.update_text_effects(self.p2_text_effects)

    def update(self, pose_data, dt):
        self.pose_data = pose_data
        self.handle_input(pose_data)

    def render(self, frame):
        
        
        height, width, _ = frame.shape
        half_width = width // 2
        image_p1, image_p2 = frame[:, :half_width], frame[:, half_width:]

        for trail in [self.p1_left_trail, self.p1_right_trail]:
            for i, point in enumerate(trail):
                alpha = i / len(trail)
                radius = int(2 + alpha * 15)
                color = (0, int(200 * alpha), int(255 * alpha))
                if i != len(trail)-1:
                    nextPoint = trail[i+1]
                    for a_d in range(0, 100, 1):
                        a = a_d / 100.0
                        interPoint = (int(point[0]*(1-a) + nextPoint[0]*a), int(point[1]*(1-a) + nextPoint[1]*a))
                        cv2.circle(image_p1, interPoint, radius, color, -1)
                cv2.circle(image_p1, (int(point[0]), int(point[1])), radius, color, -1)
        for obj_list in [self.p1_fruits, self.p1_bombs, self.p1_sliced_pieces]:
            for obj in obj_list:
                img_to_draw = obj['image'] if 'image' in obj else obj['images']['whole']
                if 'angle' in obj: img_to_draw = rotate_image(img_to_draw, obj['angle'])
                overlay_transparent(image_p1, img_to_draw, int(obj['x'] * half_width), int(obj['y'] * height))
        for p in self.p1_particles: cv2.circle(image_p1, (int(p['x'] * half_width), int(p['y'] * height)), p['radius'], p['color'], -1)
        for effect in self.p1_text_effects:
            pos = (int(effect['x'] * half_width), int(effect['y'] * height))
            cv2.putText(image_p1, effect['text'], pos, cv2.FONT_HERSHEY_SIMPLEX, effect['font_scale'], effect['color'], 3, cv2.LINE_AA)
        for trail in [self.p2_left_trail, self.p2_right_trail]:
            for i, point in enumerate(trail):
                alpha = i / len(trail)
                radius = int(2 + alpha * 15)
                color = (int(255 * alpha), int(150 * alpha), 0)
                if i != len(trail)-1:
                    nextPoint = trail[i+1]
                    for a_d in range(0, 100, 1):
                        a = a_d / 100.0
                        interPoint = (int(point[0]*(1-a) + nextPoint[0]*a), int(point[1]*(1-a) + nextPoint[1]*a))
                        cv2.circle(image_p2, interPoint, radius, color, -1)
                cv2.circle(image_p2, (int(point[0]), int(point[1])), radius, color, -1)
        for obj_list in [self.p2_fruits, self.p2_bombs, self.p2_sliced_pieces]:
            for obj in obj_list:
                img_to_draw = obj['image'] if 'image' in obj else obj['images']['whole']
                if 'angle' in obj: img_to_draw = rotate_image(img_to_draw, obj['angle'])
                overlay_transparent(image_p2, img_to_draw, int(obj['x'] * half_width), int(obj['y'] * height))
        for p in self.p2_particles: cv2.circle(image_p2, (int(p['x'] * half_width), int(p['y'] * height)), p['radius'], p['color'], -1)
        for effect in self.p2_text_effects:
            pos = (int(effect['x'] * half_width), int(effect['y'] * height))
            cv2.putText(image_p2, effect['text'], pos, cv2.FONT_HERSHEY_SIMPLEX, effect['font_scale'], effect['color'], 3, cv2.LINE_AA)

        frame[:, :half_width], frame[:, half_width:] = image_p1, image_p2
        
        cv2.line(frame, (half_width, 0), (half_width, height), (255, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (width, 150), (20, 20, 20), -1)

        cv2.putText(frame, 'PLAYER 1', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p1_score}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p1_feedback, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'PLAYER 2', (width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p2_score}', (width - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p2_feedback, (width - 300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        stopwatch_x = width // 2
        stopwatch_y = 75
        overlay_transparent(frame, self.stopwatch_image, stopwatch_x, stopwatch_y)

        time_text = str(int(self.time_remaining))
        font_scale = 1.5
        font_thickness = 3
        text_size, _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = stopwatch_x - text_size[0] // 2
        text_y = stopwatch_y + 55
        cv2.putText(frame, time_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        if self.game_over:
            game_over_text = "Game Over"
            font_scale_go = 3
            font_thickness_go = 5
            text_size_go, _ = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_go, font_thickness_go)
            text_x_go = (width - text_size_go[0]) // 2
            text_y_go = (height + text_size_go[1]) // 2
            cv2.putText(frame, game_over_text, (text_x_go, text_y_go), cv2.FONT_HERSHEY_SIMPLEX, font_scale_go, (0, 215, 255), font_thickness_go, cv2.LINE_AA)

        return frame