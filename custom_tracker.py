import cv2
import numpy as np

class CustomTracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=8, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.bbox = None
        self.prev_gray = None
        self.tracks = []
        self.lost_count = 0
        self.max_lost_frames = 15
        
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

    def init(self, frame, bbox):
        self.bbox = bbox
        self.lost_count = 0
        self.tracks = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()

        x, y, w, h = bbox
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)

        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if corners is not None:
            for corner in corners: self.tracks.append([corner])
        return len(self.tracks) > 0

    def update(self, frame):
        if self.prev_gray is None or len(self.tracks) == 0: return False, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        self.prev_gray = gray.copy()

        if p1 is None or st is None:
            self.lost_count += 1
            return self.lost_count <= self.max_lost_frames, self.bbox

        good_new = p1[st == 1]
        new_tracks = []
        for i, tr in enumerate(self.tracks):
            if st[i] == 1:
                tr.append(p1[i])
                if len(tr) > 10: del tr[0]
                new_tracks.append(tr)
        self.tracks = new_tracks

        if len(good_new) < 8:
            self.lost_count += 1
            return self.lost_count <= self.max_lost_frames, self.bbox
        
        self.lost_count = 0
        x_min, y_min, new_w, new_h = cv2.boundingRect(good_new)
        
        self.kalman.predict()
        measurement = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
        self.kalman.correct(measurement)
        state = self.kalman.statePost.flatten()
        
        smooth_x, smooth_y = max(0, int(state[0])), max(0, int(state[1]))
        smooth_w, smooth_h = max(15, int(state[2])), max(20, int(state[3]))
        
        self.bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
        return True, self.bbox