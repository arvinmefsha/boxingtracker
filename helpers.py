# helpers.py

import numpy as np

def overlay_transparent(background, overlay, x, y):
    if overlay.shape[2] < 4:
        return background

    bg_h, bg_w, _ = background.shape
    img_h, img_w, _ = overlay.shape

    x_pos = x - img_w // 2
    y_pos = y - img_h // 2

    x_start_bg = max(x_pos, 0)
    y_start_bg = max(y_pos, 0)
    x_end_bg = min(x_pos + img_w, bg_w)
    y_end_bg = min(y_pos + img_h, bg_h)

    x_start_overlay = max(0, -x_pos)
    y_start_overlay = max(0, -y_pos)
    x_end_overlay = x_start_overlay + (x_end_bg - x_start_bg)
    y_end_overlay = y_start_overlay + (y_end_bg - y_start_bg)

    if (x_end_bg <= x_start_bg) or (y_end_bg <= y_start_bg):
        return background

    roi = background[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
    overlay_clipped = overlay[y_start_overlay:y_end_overlay, x_start_overlay:x_end_overlay]

    alpha = overlay_clipped[:, :, 3] / 255.0
    mask = np.dstack([alpha, alpha, alpha])
    
    overlay_rgb = overlay_clipped[:, :, :3]
    background_part = roi * (1 - mask)
    overlay_part = overlay_rgb * mask

    background[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = background_part + overlay_part
    return background

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
   
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
   
    if angle > 180.0:
        angle = 360 - angle
       
    return angle

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_velocity(p1, p2, dt):
    if dt == 0:
        return 0
    distance = calculate_distance(p1, p2)
    return distance / dt

def line_circle_intersection(start, end, center, radius):
    x1, y1 = start
    x2, y2 = end
    cx, cy = center
    
    dx = x2 - x1
    dy = y2 - y1
    a = dx**2 + dy**2
    if a == 0:
        return False
    b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    c = (x1 - cx)**2 + (y1 - cy)**2 - radius**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return False
    
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    return False