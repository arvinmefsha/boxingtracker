import cv2
import mediapipe as mp

class InputModule:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            exit()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_frame(self):
        success, frame = self.cap.read()
        if not success or frame is None:
            print("Ignoring empty camera frame.")
            return None
        if frame.shape[1] == 0:
            print("Warning: Frame has zero width.")
            return None
        return cv2.flip(frame, 1)

    def process_pose(self, frame):
        if frame is None:
            return {'p1': None, 'p2': None}
        
        height, width, _ = frame.shape
        image_p1 = frame[:, :width//2]
        image_p2 = frame[:, width//2:]
        
        image_rgb_p1 = cv2.cvtColor(image_p1, cv2.COLOR_BGR2RGB)
        image_rgb_p2 = cv2.cvtColor(image_p2, cv2.COLOR_BGR2RGB)
        
        results_p1 = self.pose.process(image_rgb_p1)
        results_p2 = self.pose.process(image_rgb_p2)
        
        return {'p1': results_p1, 'p2': results_p2}

    def release(self):
        self.cap.release()
        self.pose.close()