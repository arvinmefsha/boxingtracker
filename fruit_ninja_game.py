# fruit_ninja_game.py

import cv2
import mediapipe as mp
import time
import random
import numpy as np
import os
import sys

import pygame

from game import Game
from helpers import calculate_velocity, line_circle_intersection

# ... (overlay_transparent function remains the same) ...
def overlay_transparent(background, overlay, x, y):
    """
    Overlays a transparent PNG on a background image, correctly handling edges.
    """
    if overlay.shape[2] < 4: return background
    bg_h, bg_w, _ = background.shape
    img_h, img_w, _ = overlay.shape
    x_pos = x - img_w // 2
    y_pos = y - img_h // 2
    x_start_bg, y_start_bg = max(x_pos, 0), max(y_pos, 0)
    x_end_bg, y_end_bg = min(x_pos + img_w, bg_w), min(y_pos + img_h, bg_h)
    x_start_overlay, y_start_overlay = max(0, -x_pos), max(0, -y_pos)
    x_end_overlay = x_start_overlay + (x_end_bg - x_start_bg)
    y_end_overlay = y_start_overlay + (y_end_bg - y_start_bg)
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

## NEW HELPER FUNCTION: Rotates an image around its center
## FIXED HELPER FUNCTION: Rotates an image around its center, preserving alpha channel.
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # If the image has an alpha channel (4 channels)
    if image.shape[2] == 4:
        # Split into RGB and Alpha
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]

        # Rotate RGB channels
        rotated_rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)) # Fill black for background
        # Rotate Alpha channel (as grayscale)
        rotated_alpha = cv2.warpAffine(alpha, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) # Fill 0 for transparent

        # Combine rotated RGB and Alpha back into a 4-channel image
        rotated_image = cv2.merge([rotated_rgb[:,:,0], rotated_rgb[:,:,1], rotated_rgb[:,:,2], rotated_alpha])
    else:
        # If no alpha channel, just rotate normally (border will be filled with black)
        rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return rotated_image

class FruitNinjaGame(Game):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_data = {'p1': None, 'p2': None}
        
        self.gravity = 1.0
        self.spawn_interval = 0.9
        self.slice_velocity_threshold = 1500.0
        
        ## KEPT: Image size at 200x200
        self.image_size = (200, 200) 
        self.object_radius = self.image_size[0] // 2

        print("Loading assets...")
        self.fruit_images = self.load_fruit_images('assets')
        self.bomb_image = cv2.imread(os.path.join('assets', 'bomb.png'), cv2.IMREAD_UNCHANGED)
        if self.bomb_image is None:
            print("\n--- ERROR: Failed to load 'bomb.png'. Make sure it's in the 'assets' folder. ---\n")
            sys.exit(1)
        self.bomb_image = cv2.resize(self.bomb_image, self.image_size, interpolation=cv2.INTER_AREA)
        print("Assets loaded successfully.")


        print("Initializing sound...")
        pygame.mixer.init()
        try:
            self.theme_sound = pygame.mixer.Sound('assets/sounds/fruitninjatheme.mp3')
            self.slice_sound = pygame.mixer.Sound('assets/sounds/slice.mp3')
            self.bomb_sound = pygame.mixer.Sound('assets/sounds/explosion.mp3')
            # You can set volume for each sound (0.0 to 1.0)
            self.theme_sound.set_volume(0.3)
            self.slice_sound.set_volume(0.3)
            self.bomb_sound.set_volume(0.7)
            print("Sound loaded successfully.")
        except pygame.error as e:
            print(f"\n--- Sound Error: {e} ---")
            print("Could not load sound files. Make sure 'slice.wav' and 'bomb.wav' are in 'assets/sound/'.")
            print("Game will run without sound.")
            self.theme_sound = None
            self.slice_sound = None
            self.bomb_sound = None

        
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
        self.p1_sliced_pieces, self.p2_sliced_pieces = [], []
        self.p1_particles, self.p2_particles = [], []
        self.p1_feedback, self.p2_feedback = "", ""
        self.p1_prev_left_wrist_screen, self.p1_prev_right_wrist_screen = None, None
        self.p2_prev_left_wrist_screen, self.p2_prev_right_wrist_screen = None, None
        self.p1_last_spawn, self.p2_last_spawn = time.time(), time.time()
        self.prev_time = time.time()

    def spawn_object(self, is_fruit=True):
        vx = random.uniform(-0.3, 0.3)
        ## KEPT: Modified initial vertical velocity
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
            # Optional: Add some rotation to sliced pieces for extra effect
            if 'rotation_speed' in obj:
                obj['angle'] = (obj.get('angle', 0) + obj['rotation_speed'] * dt) % 360

            if 'life' in obj:
                obj['life'] -= 1
                if obj['life'] <= 0: continue
            if obj['y'] < 1.3: 
                new_objects.append(obj)
        return new_objects

    def check_slice(self, prev_pos, curr_pos, objects, half_width, height):
        sliced = []
        if prev_pos is None or curr_pos is None: return sliced
        start, end = tuple(prev_pos), tuple(curr_pos)
        for i, obj in enumerate(objects):
            if not (0 <= obj['y'] <= 1.0): continue # Only slice visible objects
            center = (obj['x'] * half_width, obj['y'] * height)
            if line_circle_intersection(start, end, center, obj['radius']):
                sliced.append(i)
        return sliced
        
    def process_slicing_for_player(self, player_data):
        pose, prev_wrists, fruits, bombs, sliced_pieces, particles, score, feedback_text = player_data
        
        half_width, height = 960, 1080
        landmarks = pose.pose_landmarks.landmark
        wrists = {'left': [landmarks[15].x * half_width, landmarks[15].y * height],
                  'right': [landmarks[16].x * half_width, landmarks[16].y * height]}

        for hand in ['left', 'right']:
            if prev_wrists[hand]:
                velocity = calculate_velocity(wrists[hand], prev_wrists[hand], self.dt)
                if velocity > self.slice_velocity_threshold:
                    # Calculate slice angle
                    dx = wrists[hand][0] - prev_wrists[hand][0]
                    dy = wrists[hand][1] - prev_wrists[hand][1]
                    # Angle in degrees, 0 is horizontal right, positive is counter-clockwise
                    slice_angle_rad = np.arctan2(-dy, dx) # -dy because screen Y is inverted
                    slice_angle_deg = np.degrees(slice_angle_rad) 

                    sliced_fruits = self.check_slice(prev_wrists[hand],  wrists[hand], fruits, half_width, height)
                    
                    if sliced_fruits and self.slice_sound:
                        self.slice_sound.play()

                    for i in sorted(sliced_fruits, reverse=True):
                        fruit = fruits.pop(i)
                        score += 1
                        feedback_text = "Sliced!"
                        
                        # Push pieces perpendicular to the slice direction
                        separation_speed = 0.5 
                        left_vx = separation_speed * np.cos(slice_angle_rad + np.pi/2) 
                        left_vy = separation_speed * np.sin(slice_angle_rad + np.pi/2)
                        right_vx = separation_speed * np.cos(slice_angle_rad - np.pi/2)
                        right_vy = separation_speed * np.sin(slice_angle_rad - np.pi/2)

                        # Create rotated pieces
                        piece1 = {
                            'x': fruit['x'], 'y': fruit['y'], 
                            'vx': fruit['vx'] + left_vx, 'vy': fruit['vy'] + left_vy, 
                            'life': 60, 
                            'image': rotate_image(fruit['images']['left'], slice_angle_deg),
                            'rotation_speed': random.uniform(-100, 100) # Add some rotational spin
                        }
                        piece2 = {
                            'x': fruit['x'], 'y': fruit['y'], 
                            'vx': fruit['vx'] + right_vx, 'vy': fruit['vy'] + right_vy, 
                            'life': 60, 
                            'image': rotate_image(fruit['images']['right'], slice_angle_deg),
                            'rotation_speed': random.uniform(-100, 100)
                        }
                        ## FIXED: Changed self.p1_sliced_pieces to the generic 'sliced_pieces' list
                        sliced_pieces.extend([piece1, piece2])
                    
                    bomb_sliced = self.check_slice(prev_wrists[hand], wrists[hand], bombs, half_width, height)
                    if bomb_sliced and self.bomb_sound:
                        self.bomb_sound.play()
                    # Bomb Slicing
                    for i in sorted(bomb_sliced, reverse=True):
                        bomb = bombs.pop(i)
                        score -= 5
                        feedback_text = "Boom!"
                        for _ in range(30):
                            angle, speed = random.uniform(0, 2*np.pi), random.uniform(1, 4)
                            particles.append({'x': bomb['x'], 'y': bomb['y'], 'vx': np.cos(angle)*speed, 'vy': np.sin(angle)*speed, 'life': 30, 'radius': random.randint(3,8), 'color': random.choice([(0,0,255), (0,165,255), (0,255,255)])})
        return wrists, score, feedback_text

    def handle_input(self, pose_data):
        current_time = time.time()
        self.dt = current_time - self.prev_time
        if self.dt == 0: return
        self.prev_time = current_time

        if current_time - self.p1_last_spawn > self.spawn_interval:
            obj = self.spawn_object(random.random() < 0.85)
            (self.p1_fruits if obj['type'] == 'fruit' else self.p1_bombs).append(obj)
            self.p1_last_spawn = current_time
        if current_time - self.p2_last_spawn > self.spawn_interval:
            obj = self.spawn_object(random.random() < 0.85)
            (self.p2_fruits if obj['type'] == 'fruit' else self.p2_bombs).append(obj)
            self.p2_last_spawn = current_time
        
        for lst in ['p1_fruits', 'p1_bombs', 'p1_sliced_pieces', 'p1_particles', 'p2_fruits', 'p2_bombs', 'p2_sliced_pieces', 'p2_particles']:
            setattr(self, lst, self.update_objects(getattr(self, lst), self.dt))

        if pose_data['p1'] and pose_data['p1'].pose_landmarks:
            wrists, self.p1_score, self.p1_feedback = self.process_slicing_for_player([pose_data['p1'], {'left': self.p1_prev_left_wrist_screen, 'right': self.p1_prev_right_wrist_screen}, self.p1_fruits, self.p1_bombs, self.p1_sliced_pieces, self.p1_particles, self.p1_score, self.p1_feedback])
            self.p1_prev_left_wrist_screen, self.p1_prev_right_wrist_screen = wrists['left'], wrists['right']
        if pose_data['p2'] and pose_data['p2'].pose_landmarks:
            wrists, self.p2_score, self.p2_feedback = self.process_slicing_for_player([pose_data['p2'], {'left': self.p2_prev_left_wrist_screen, 'right': self.p2_prev_right_wrist_screen}, self.p2_fruits, self.p2_bombs, self.p2_sliced_pieces, self.p2_particles, self.p2_score, self.p2_feedback])
            self.p2_prev_left_wrist_screen, self.p2_prev_right_wrist_screen = wrists['left'], wrists['right']

    def update(self, pose_data):
        self.pose_data = pose_data
        self.handle_input(pose_data)

    def render(self, frame):
        self.update(self.pose_data)
        
        height, width, _ = frame.shape
        half_width = width // 2
        image_p1 = frame[:, :half_width]
        image_p2 = frame[:, half_width:]

        # Drawing loop for p1_sliced_pieces needs to handle rotation
        for obj_list in [self.p1_fruits, self.p1_bombs, self.p1_sliced_pieces]:
            for obj in obj_list:
                img_to_draw = obj['image'] if 'image' in obj else obj['images']['whole']
                if 'angle' in obj: # If piece has an angle, rotate it before drawing
                    img_to_draw = rotate_image(img_to_draw, obj['angle'])
                overlay_transparent(image_p1, img_to_draw, int(obj['x'] * half_width), int(obj['y'] * height))

        for p in self.p1_particles: cv2.circle(image_p1, (int(p['x'] * half_width), int(p['y'] * height)), p['radius'], p['color'], -1)
        if self.pose_data['p1']: self.mp_drawing.draw_landmarks(image_p1, self.pose_data['p1'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
        # Drawing loop for p2_sliced_pieces needs to handle rotation
        for obj_list in [self.p2_fruits, self.p2_bombs, self.p2_sliced_pieces]:
            for obj in obj_list:
                img_to_draw = obj['image'] if 'image' in obj else obj['images']['whole']
                if 'angle' in obj: # If piece has an angle, rotate it before drawing
                    img_to_draw = rotate_image(img_to_draw, obj['angle'])
                overlay_transparent(image_p2, img_to_draw, int(obj['x'] * half_width), int(obj['y'] * height))

        for p in self.p2_particles: cv2.circle(image_p2, (int(p['x'] * half_width), int(p['y'] * height)), p['radius'], p['color'], -1)
        if self.pose_data['p2']: self.mp_drawing.draw_landmarks(image_p2, self.pose_data['p2'].pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        frame[:, :half_width] = image_p1
        frame[:, half_width:] = image_p2
        
        cv2.line(frame, (half_width, 0), (half_width, height), (255, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (width, 150), (20, 20, 20), -1)

        cv2.putText(frame, 'PLAYER 1', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p1_score}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p1_feedback, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, 'PLAYER 2', (width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'SCORE: {self.p2_score}', (width - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self.p2_feedback, (width - 300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame