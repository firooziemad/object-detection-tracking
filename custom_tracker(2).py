import cv2
import numpy as np

class CT:
    def __init__(self):
        self.fp = dict(maxCorners=100, qualityLevel=0.01, minDistance=8, blockSize=7)
        self.lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.bb = None
        self.pg = None
        self.tr = []
        self.tl = 10
        self.di = 5
        self.fi = 0
        self.lc = 0
        self.mlf = 15
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.bbh = []
        self.mh = 3
        self.ssf = 0.25
        self.pbs = None
        self.tm = "normal"
        self.mc = {
            "smooth": {"detect_interval": 8, "max_corners": 50, "quality_level": 0.03, "min_distance": 12, "win_size": (19, 19), "max_level": 2, "max_lost_frames": 20},
            "normal": {"detect_interval": 5, "max_corners": 80, "quality_level": 0.02, "min_distance": 10, "win_size": (15, 15), "max_level": 2, "max_lost_frames": 15, "size_smoothing": 0.08},
            "high_motion": {"detect_interval": 3, "max_corners": 120, "quality_level": 0.015, "min_distance": 8, "win_size": (13, 13), "max_level": 3, "max_lost_frames": 10}
        }

    def init(self, frame, bbox):
        self.bb = bbox
        self.lc = 0
        self.fi = 0
        self.tr = []
        self.bbh = [bbox]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.pg = gray.copy()

        x, y, w, h = bbox
        regions = [(x + w//4, y + h//4, w//2, h//2), (x, y, w, h)]
        total_features = 0
        for region in regions:
            rx, ry, rw, rh = region
            mask = np.zeros_like(gray)
            mask[ry:ry+rh, rx:rx+rw] = 255
            region_params = self.fp.copy()
            region_params['maxCorners'] = 50
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **region_params)
            if corners is not None:
                for corner in corners: self.tr.append([corner])
                total_features += len(corners)

        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        return total_features > 0

    def set_tracking_mode(self, mode):
        if mode in self.mc:
            self.tm = mode
            config = self.mc[mode]
            if "size_smoothing" in config: self.ssf = config["size_smoothing"]
            self.mlf = config["max_lost_frames"]
            self.di = config["detect_interval"]
            self.fp.update({'maxCorners': config["max_corners"], 'qualityLevel': config["quality_level"], 'minDistance': config["min_distance"]})
            self.lk.update({'winSize': config["win_size"], 'maxLevel': config["max_level"]})
            print(f"Tracking mode set to: {mode}")
            return True
        return False

    def _is_lost(self, tracked_points, frame_shape):
        if len(tracked_points) < 5:
            return True

        h, w = frame_shape[:2]
        x_coords = tracked_points[:, 0, 0]
        y_coords = tracked_points[:, 0, 1]

        edge_threshold = 20
        at_edge = np.sum((x_coords < edge_threshold) | (x_coords > w - edge_threshold) | \
                         (y_coords < edge_threshold) | (y_coords > h - edge_threshold))
        if at_edge / len(tracked_points) > 0.7:
            return True

        spread_x = np.max(x_coords) - np.min(x_coords)
        spread_y = np.max(y_coords) - np.min(y_coords)
        if spread_x < 30 or spread_y < 40:
            return True

        return False

    def update(self, frame):
        if self.pg is None: return False, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.fi += 1

        if len(self.tr) > 0:
            p0 = np.float32([t[-1] for t in self.tr]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.pg, gray, p0, None, **self.lk)

            good_new = p1[st == 1]
            self.tr = [t for i, t in enumerate(self.tr) if st[i] == 1]

            if not self._is_lost(good_new, gray.shape):
                x_coords = good_new[:, 0, 0]
                y_coords = good_new[:, 0, 1]
                x_min, x_max = np.percentile(x_coords, 5), np.percentile(x_coords, 95)
                y_min, y_max = np.percentile(y_coords, 5), np.percentile(y_coords, 95)
                detected_w, detected_h = x_max - x_min, y_max - y_min
                new_w, new_h = detected_w, detected_h
                if self.pbs is not None:
                    prev_w, prev_h = self.pbs
                    new_w = prev_w * (1-self.ssf) + detected_w * self.ssf
                    new_h = prev_h * (1-self.ssf) + detected_h * self.ssf
                self.pbs = (new_w, new_h)
                self.lc = 0
                self.kf.predict()
                measurement = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
                self.kf.correct(measurement)
                self.bb = tuple(map(int, self.kf.statePost.flatten()[:4]))
            else:
                self.lc += 3
        else:
            self.lc += 1

        if (self.fi % self.di == 0 or len(self.tr) < 20) and self.bb is not None:
            x, y, w, h = self.bb
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            for t in self.tr:
                cv2.circle(mask, tuple(map(int, t[-1][0])), 5, 0, -1)

            new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.fp)
            if new_corners is not None:
                for corner in new_corners:
                    self.tr.append([corner])

        self.pg = gray.copy()
        success = self.lc <= self.mlf
        return success, self.bb
		
		
		git add custom_tracker.py main_runner.py
git commit -m "Refactor tracker state variables for conciseness"