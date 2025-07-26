import cv2
import numpy as np

class CT: # Renamed
    def __init__(self):
        # Renamed variables
        self.fp = dict(maxCorners=100, qualityLevel=0.01, minDistance=8, blockSize=7)
        self.lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.bbox = None
        self.prev_gray = None
        self.tracks = []
        self.lost_count = 0
        self.max_lost_frames = 15
        self.frame_idx = 0

        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        self.size_smoothing_factor = 0.25
        self.prev_bbox_size = None
        self.tracking_mode = "normal"
        self.mode_configs = {
            "smooth": {"detect_interval": 8, "max_corners": 50, "quality_level": 0.03, "min_distance": 12, "win_size": (19, 19), "max_level": 2, "max_lost_frames": 20},
            "normal": {"detect_interval": 5, "max_corners": 80, "quality_level": 0.02, "min_distance": 10, "win_size": (15, 15), "max_level": 2, "max_lost_frames": 15, "size_smoothing": 0.08},
            "high_motion": {"detect_interval": 3, "max_corners": 120, "quality_level": 0.015, "min_distance": 8, "win_size": (13, 13), "max_level": 3, "max_lost_frames": 10}
        }
        self.detect_interval = self.mode_configs["normal"]["detect_interval"]

    def init(self, frame, bbox):
        self.bbox = bbox
        self.lost_count = 0
        self.tracks = []
        self.prev_bbox_size = (bbox[2], bbox[3])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()

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
                for corner in corners: self.tracks.append([corner])
                total_features += len(corners)

        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        return total_features > 0

    def set_tracking_mode(self, mode):
        if mode in self.mode_configs:
            self.tracking_mode = mode
            config = self.mode_configs[mode]
            if "size_smoothing" in config: self.size_smoothing_factor = config["size_smoothing"]
            self.max_lost_frames = config["max_lost_frames"]
            self.detect_interval = config["detect_interval"]
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
        if self.prev_gray is None: return False, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_idx += 1

        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk)

            good_new = p1[st == 1]
            self.tracks = [tr for i, tr in enumerate(self.tracks) if st[i] == 1]

            if not self._is_lost(good_new, gray.shape):
                x_coords = good_new[:, 0, 0]
                y_coords = good_new[:, 0, 1]
                x_min, x_max = np.percentile(x_coords, 5), np.percentile(x_coords, 95)
                y_min, y_max = np.percentile(y_coords, 5), np.percentile(y_coords, 95)
                detected_w, detected_h = x_max - x_min, y_max - y_min
                new_w, new_h = detected_w, detected_h
                if self.prev_bbox_size is not None:
                    prev_w, prev_h = self.prev_bbox_size
                    new_w = prev_w * (1-self.size_smoothing_factor) + detected_w * self.size_smoothing_factor
                    new_h = prev_h * (1-self.size_smoothing_factor) + detected_h * self.size_smoothing_factor
                self.prev_bbox_size = (new_w, new_h)
                self.lost_count = 0
                self.kalman.predict()
                measurement = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
                self.kalman.correct(measurement)
                self.bbox = tuple(map(int, self.kalman.statePost.flatten()[:4]))
            else:
                self.lost_count += 3
        else:
            self.lost_count += 1

        if (self.frame_idx % self.detect_interval == 0 or len(self.tracks) < 20) and self.bbox is not None:
            x, y, w, h = self.bbox
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            for tr in self.tracks:
                cv2.circle(mask, tuple(map(int, tr[-1][0])), 5, 0, -1)

            new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.fp)
            if new_corners is not None:
                for corner in new_corners:
                    self.tracks.append([corner])

        self.prev_gray = gray.copy()
        success = self.lost_count <= self.max_lost_frames
        return success, self.bbox
		
		git add custom_tracker.py
git commit -m "refactor(tracker): Rename class to CT and core parameter variables"