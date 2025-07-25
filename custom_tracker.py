import cv2
import numpy as np

class CustomTracker:
    def __init__(self):
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )
        self.bbox = None
        self.prev_gray = None
        self.tracks = []

    def init(self, frame, bbox):
        self.bbox = bbox
        self.tracks = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()

        x, y, w, h = bbox
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        
        corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        if corners is not None:
            for corner in corners:
                self.tracks.append([corner])
        
        print(f"Initialized with {len(self.tracks)} feature points.")
        return len(self.tracks) > 0

    def update(self, frame):
        if not self.tracks:
            return False, None
        
        return True, self.bbox