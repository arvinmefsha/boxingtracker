# helpers.py

import numpy as np

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

def line_circle_intersection(start, end, center, radius):
    x1, y1 = start
    x2, y2 = end
    cx, cy = center
    
    dx = x2 - x1
    dy = y2 - y1
    a = dx**2 + dy**2
    if a == 0:
        return False  # No movement
    b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    c = (x1 - cx)**2 + (y1 - cy)**2 - radius**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return False
    
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    
    # Check if either t is between 0 and 1
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    return False