import cv2
import numpy as np

class CustomTracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=8, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.bbox = None
        self.prev_gray = None
        self.tracks = []
        self.track_len = 10
        self.detect_interval = 5
        self.frame_idx = 0
        self.lost_count = 0
        self.max_lost_frames = 15
        
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        self.tracking_mode = "normal"
        self.mode_configs = {
            "smooth": {"detect_interval": 8, "max_corners": 50, "quality_level": 0.03, "min_distance": 12, "win_size": (19, 19), "max_level": 2, "max_lost_frames": 20},
            "normal": {"detect_interval": 5, "max_corners": 80, "quality_level": 0.02, "min_distance": 10, "win_size": (15, 15), "max_level": 2, "max_lost_frames": 15},
            "high_motion": {"detect_interval": 3, "max_corners": 120, "quality_level": 0.015, "min_distance": 8, "win_size": (13, 13), "max_level": 3, "max_lost_frames": 10}
        }

    def init(self, frame, bbox):
        self.bbox = bbox
        self.lost_count = 0
        self.frame_idx = 0
        self.tracks = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()
        
        x, y, w, h = bbox
        regions = [(x + w//4, y + h//4, w//2, h//2), (x, y, w, h)]
        total_features = 0
        for region in regions:
            rx, ry, rw, rh = region
            mask = np.zeros_like(gray)
            mask[ry:ry+rh, rx:rx+rw] = 255
            region_params = self.feature_params.copy()
            region_params['maxCorners'] = 50
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **region_params)
            if corners is not None:
                for corner in corners: self.tracks.append([corner])
                total_features += len(corners)

        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        return total_features > 0
    
    def set_tracking_mode(self, mode):
        if mode in self.mode_configs:
            self.tracking_mode = mode
            config = self.mode_configs[mode]
            self.detect_interval = config["detect_interval"]
            self.max_lost_frames = config["max_lost_frames"]
            self.feature_params.update({'maxCorners': config["max_corners"], 'qualityLevel': config["quality_level"], 'minDistance': config["min_distance"]})
            self.lk_params.update({'winSize': config["win_size"], 'maxLevel': config["max_level"]})
            print(f"Tracking mode set to: {mode}")
            return True
        return False

    def update(self, frame):
        if self.prev_gray is None: return False, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_idx += 1
        
        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            
            good_new = p1[st == 1]
            new_tracks = []
            for i, tr in enumerate(self.tracks):
                if st[i] == 1:
                    tr.append(p1[i])
                    if len(tr) > self.track_len: del tr[0]
                    new_tracks.append(tr)
            self.tracks = new_tracks
            
            if len(good_new) >= 8:
                current_points = good_new.reshape(-1, 2)
                x_coords = current_points[:, 0]
                y_coords = current_points[:, 1]
                
                x_min = np.percentile(x_coords, 5)
                x_max = np.percentile(x_coords, 95)
                y_min = np.percentile(y_coords, 5)
                y_max = np.percentile(y_coords, 95)

                padding_w = max(10, (x_max - x_min) * 0.15)
                padding_h = max(10, (y_max - y_min) * 0.15)

                x_min = max(0, x_min - padding_w)
                y_min = max(0, y_min - padding_h)
                x_max = min(gray.shape[1], x_max + padding_w)
                y_max = min(gray.shape[0], y_max + padding_h)

                new_w = x_max - x_min
                new_h = y_max - y_min

                self.lost_count = 0
                self.kalman.predict()
                measurement = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
                self.kalman.correct(measurement)
                state = self.kalman.statePost.flatten()
                self.bbox = (int(state[0]), int(state[1]), int(state[2]), int(state[3]))
            else:
                self.lost_count += 1
        else:
            self.lost_count += 1
        
        self.prev_gray = gray.copy()
        success = self.lost_count <= self.max_lost_frames
        return success, self.bbox