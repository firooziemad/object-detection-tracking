import cv2
import numpy as np

class CustomTracker:
    def __init__(self):
        # Feature tracking parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.bbox = None
        self.tracks = []
        self.track_len = 10
        self.detect_interval = 5
        self.frame_idx = 0
        self.lost_count = 0
        self.max_lost_frames = 15
        
        # Enhanced Kalman Filter for ultra-smooth bounding box
        self.kalman = cv2.KalmanFilter(8, 4)  # [x, y, w, h, vx, vy, vw, vh]
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0.95, 0, 0, 0],  # Slightly damped velocities
            [0, 0, 0, 0, 0, 0.95, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.9, 0],   # More damped size velocities
            [0, 0, 0, 0, 0, 0, 0, 0.9]
        ], np.float32)
        
        # Enhanced noise parameters for smoother tracking
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.005
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.15
        
        self.bbox_history = []
        self.max_history = 5
        self.size_smoothing_factor = 0.12
        self.prev_bbox_size = None
        
        self.bbox_buffer = []
        self.buffer_size = 3
        self.confidence_threshold = 0.7
        
        # Tracking mode configurations
        self.tracking_mode = "normal"
        self.mode_configs = {
            "smooth": {"detect_interval": 8, "max_corners": 60, "quality_level": 0.025, "min_distance": 15, "win_size": (25, 25), "max_level": 2, "max_lost_frames": 25, "size_smoothing": 0.08, "position_smoothing": 0.8, "kalman_process_noise": 0.003, "kalman_measurement_noise": 0.2},
            "normal": {"detect_interval": 5, "max_corners": 80, "quality_level": 0.02, "min_distance": 10, "win_size": (19, 19), "max_level": 2, "max_lost_frames": 15, "size_smoothing": 0.12, "position_smoothing": 0.6, "kalman_process_noise": 0.005, "kalman_measurement_noise": 0.15},
            "high_motion": {"detect_interval": 3, "max_corners": 120, "quality_level": 0.015, "min_distance": 8, "win_size": (15, 15), "max_level": 3, "max_lost_frames": 10, "size_smoothing": 0.18, "position_smoothing": 0.4, "kalman_process_noise": 0.01, "kalman_measurement_noise": 0.1}
        }
        
        self.position_history = []
        self.position_history_size = 10
        self.velocity_history = []
        self.velocity_history_size = 5

        # --- GPU ACCELERATION SETUP ---
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            print("✅ OpenCV CUDA is available. Initializing GPU tracker components.")
            try:
                # Create GPU-accelerated detectors and trackers
                self.gpu_lk_flow = cv2.cuda.createSparsePyrLKOpticalFlow(
                    self.lk_params['winSize'], self.lk_params['maxLevel'], 30
                )
                # GpuMat objects to hold frames on the GPU
                self.prev_gpu_gray = cv2.cuda_GpuMat()
                self.gpu_gray = cv2.cuda_GpuMat()
                self.gpu_frame = cv2.cuda_GpuMat()
            except cv2.error as e:
                print(f"❌ Failed to initialize OpenCV CUDA components: {e}")
                self.gpu_available = False
        else:
            print("ℹ️ OpenCV CUDA not found. Falling back to CPU.")
            self.prev_gray = None

    def init(self, frame, bbox):
        self.bbox = bbox
        self.lost_count = 0
        self.frame_idx = 0
        self.tracks = []
        self.bbox_history = [bbox]
        self.bbox_buffer = [bbox]
        self.position_history = [bbox[:2]]
        self.velocity_history = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.gpu_available:
            self.prev_gpu_gray.upload(gray)
        else:
            self.prev_gray = gray.copy()
        
        x, y, w, h = bbox
        regions = [
            (x + w//6, y + h//6, 2*w//3, 2*h//3),
            (x + w//8, y + h//8, 3*w//4, 3*h//4),
            (x + w//12, y + h//12, 5*w//6, 5*h//6),
            (x, y, w, h)
        ]
        
        total_features = 0
        for i, region in enumerate(regions):
            rx, ry, rw, rh = region
            mask = np.zeros_like(gray)
            mask[ry:ry+rh, rx:rx+rw] = 255
            
            region_params = self.feature_params.copy()
            region_params['maxCorners'] = [40, 30, 25, 20][i]
            region_params['qualityLevel'] = [0.03, 0.025, 0.02, 0.015][i]
            
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **region_params)
            
            if corners is not None:
                for corner in corners:
                    self.tracks.append([corner])
                total_features += len(corners)
        
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.prev_bbox_size = (w, h)
        
        print(f"Enhanced initialization: {total_features} features across {len(regions)} regions")
        return total_features > 0

    def set_tracking_mode(self, mode):
        if mode in self.mode_configs:
            self.tracking_mode = mode
            config = self.mode_configs[mode]
            
            self.detect_interval = config["detect_interval"]
            self.max_lost_frames = config["max_lost_frames"]
            self.size_smoothing_factor = config["size_smoothing"]
            
            self.feature_params.update({'maxCorners': config["max_corners"], 'qualityLevel': config["quality_level"], 'minDistance': config["min_distance"]})
            self.lk_params.update({'winSize': config["win_size"], 'maxLevel': config["max_level"]})
            
            if self.gpu_available:
                self.gpu_lk_flow.setWinSize(config["win_size"])
                self.gpu_lk_flow.setMaxLevel(config["max_level"])

            self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * config["kalman_process_noise"]
            self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * config["kalman_measurement_noise"]
            
            print(f"Enhanced tracking mode set to: {mode}")
            return True
        return False

    def detect_outliers(self, points):
        if len(points) < 8:
            return points
        center = np.median(points, axis=0)
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        Q1, Q3 = np.percentile(distances, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR
        valid_mask = (distances >= lower_bound) & (distances <= upper_bound)
        return points[valid_mask]

    def predict_motion(self):
        if len(self.velocity_history) < 2:
            return None
        avg_velocity = np.mean(self.velocity_history[-3:], axis=0)
        if self.bbox is not None:
            x, y, w, h = self.bbox
            return (x + avg_velocity[0], y + avg_velocity[1], w, h)
        return None

    def smooth_bbox_transition(self, new_bbox, confidence=1.0):
        if self.bbox is None:
            return new_bbox
        current_x, current_y, current_w, current_h = self.bbox
        new_x, new_y, new_w, new_h = new_bbox
        config = self.mode_configs[self.tracking_mode]
        position_smoothing = config.get("position_smoothing", 0.5)
        motion_magnitude = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
        adaptive_factor = min(1.0, motion_magnitude / 50.0)
        smooth_factor = position_smoothing * confidence * (1.0 - adaptive_factor * 0.3)
        smooth_x = current_x + (new_x - current_x) * smooth_factor
        smooth_y = current_y + (new_y - current_y) * smooth_factor
        smooth_w = current_w + (new_w - current_w) * smooth_factor * 0.7
        smooth_h = current_h + (new_h - current_h) * smooth_factor * 0.7
        return (int(smooth_x), int(smooth_y), int(smooth_w), int(smooth_h))

    def update(self, frame):
        """Main update dispatcher. Calls GPU or CPU version based on availability."""
        if self.gpu_available:
            return self.update_gpu(frame)
        else:
            return self.update_cpu(frame)

    def _process_tracked_points(self, good_new_points, gray_shape):
        """Helper function to process points and calculate bounding box. CPU-based."""
        if len(good_new_points) >= 8:
            current_points = np.array(good_new_points).reshape(-1, 2)
            filtered_points = self.detect_outliers(current_points)
            
            if len(filtered_points) >= 6:
                x_coords, y_coords = filtered_points[:, 0], filtered_points[:, 1]
                x_min, x_max = np.percentile(x_coords, [8, 92])
                y_min, y_max = np.percentile(y_coords, [8, 92])
                
                detected_w, detected_h = x_max - x_min, y_max - y_min
                
                # ... (rest of the detailed bbox logic from original update)
                if self.tracking_mode == "smooth":
                    padding_w, padding_h = max(20, detected_w * 0.35), max(25, detected_h * 0.40)
                elif self.tracking_mode == "normal":
                    padding_w, padding_h = max(15, detected_w * 0.25), max(20, detected_h * 0.30)
                else: # high_motion
                    padding_w, padding_h = max(10, detected_w * 0.20), max(15, detected_h * 0.25)

                x_min, y_min = max(0, x_min - padding_w), max(0, y_min - padding_h)
                x_max, y_max = min(gray_shape[1], x_max + padding_w), min(gray_shape[0], y_max + padding_h)
                new_w, new_h = x_max - x_min, y_max - y_min

                if self.prev_bbox_size is not None:
                    prev_w, prev_h = self.prev_bbox_size
                    size_change = abs(new_w - prev_w) + abs(new_h - prev_h)
                    adaptive_smoothing = min(self.size_smoothing_factor * (1 + size_change / 100.0), 0.3)
                    max_w_change, max_h_change = max(8, prev_w * adaptive_smoothing), max(8, prev_h * adaptive_smoothing)
                    if abs(new_w - prev_w) > max_w_change: new_w = prev_w + np.sign(new_w - prev_w) * max_w_change
                    if abs(new_h - prev_h) > max_h_change: new_h = prev_h + np.sign(new_h - prev_h) * max_h_change
                
                self.prev_bbox_size = (new_w, new_h)

                if new_w > 20 and new_h > 25:
                    self.kalman.predict()
                    self.kalman.correct(np.array([x_min, y_min, new_w, new_h], dtype=np.float32))
                    state = self.kalman.statePost.flatten()
                    kalman_x, kalman_y = max(0, int(state[0])), max(0, int(state[1]))
                    kalman_w, kalman_h = max(20, int(state[2])), max(25, int(state[3]))
                    kalman_w, kalman_h = min(kalman_w, gray_shape[1] - kalman_x), min(kalman_h, gray_shape[0] - kalman_y)
                    
                    new_bbox = (kalman_x, kalman_y, kalman_w, kalman_h)
                    confidence = min(1.0, len(filtered_points) / 50.0)
                    self.bbox = self.smooth_bbox_transition(new_bbox, confidence)

                    self.bbox_history.append(self.bbox)
                    if len(self.bbox_history) > self.max_history: self.bbox_history.pop(0)
                    
                    if len(self.position_history) > 0:
                        velocity = (self.bbox[0] - self.position_history[-1][0], self.bbox[1] - self.position_history[-1][1])
                        self.velocity_history.append(velocity)
                        if len(self.velocity_history) > self.velocity_history_size: self.velocity_history.pop(0)

                    self.position_history.append(self.bbox[:2])
                    if len(self.position_history) > self.position_history_size: self.position_history.pop(0)
                    
                    self.lost_count = 0
                    return
        # If any check fails, increment lost count
        self.lost_count += 1

    def update_gpu(self, frame):
        """Enhanced update method with superior smoothness, accelerated with CUDA."""
        self.gpu_frame.upload(frame)
        cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2GRAY, self.gpu_gray)
        self.frame_idx += 1
        
        if len(self.tracks) > 0:
            p0_cpu = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p0_gpu = cv2.cuda_GpuMat(p0_cpu)
            
            p1_gpu, st_gpu, err_gpu = self.gpu_lk_flow.calc(self.prev_gpu_gray, self.gpu_gray, p0_gpu, None)
            
            p1 = p1_gpu.download()
            status = st_gpu.download().ravel()
            
            good_new_points = p1[status == 1]
            
            new_tracks = []
            survived_tracks = [self.tracks[i] for i, s in enumerate(status) if s == 1]
            for track, new_point in zip(survived_tracks, good_new_points):
                track.append([new_point[0]])
                if len(track) > self.track_len:
                    del track[0]
                new_tracks.append(track)
            self.tracks = new_tracks
            
            self._process_tracked_points(good_new_points, self.gpu_gray.size())
            
        if (self.frame_idx % self.detect_interval == 0 or len(self.tracks) < 25) and self.bbox is not None:
            gray_cpu = self.gpu_gray.download()
            mask = np.zeros_like(gray_cpu)
            x, y, w, h = self.bbox
            inner_margin = min(w//8, h//8, 15)
            mask[y+inner_margin:y+h-inner_margin, x+inner_margin:x+w-inner_margin] = 255
            for tr in self.tracks:
                cv2.circle(mask, tuple(map(int, tr[-1][0])), 12, 0, -1)
            
            new_feature_params = self.feature_params.copy()
            new_feature_params['maxCorners'] = 30
            new_feature_params['qualityLevel'] *= 1.2
            new_corners = cv2.goodFeaturesToTrack(gray_cpu, mask=mask, **new_feature_params)
            
            if new_corners is not None:
                for corner in new_corners:
                    self.tracks.append([corner])

        self.prev_gpu_gray, self.gpu_gray = self.gpu_gray, self.prev_gpu_gray
        
        success = (self.lost_count <= self.max_lost_frames and len(self.tracks) > 8 and self.bbox is not None)
        return success, self.bbox

    def update_cpu(self, frame):
        """The original CPU-based update method."""
        if self.prev_gray is None:
            return False, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.frame_idx += 1
        
        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            
            good_indices, good_new_points = [], []
            for i, (new_point, status, error) in enumerate(zip(p1, _st, _err)):
                if status == 1 and error < 50:
                    good_indices.append(i)
                    good_new_points.append(new_point)
            
            new_tracks = []
            for i, new_point in zip(good_indices, good_new_points):
                track = self.tracks[i]
                track.append([new_point[0]])
                if len(track) > self.track_len:
                    del track[0]
                new_tracks.append(track)
            self.tracks = new_tracks
            
            self._process_tracked_points(good_new_points, gray.shape)
            
        if (self.frame_idx % self.detect_interval == 0 or len(self.tracks) < 25) and self.bbox is not None:
            mask = np.zeros_like(gray)
            x, y, w, h = self.bbox
            inner_margin = min(w//8, h//8, 15)
            mask[y+inner_margin:y+h-inner_margin, x+inner_margin:x+w-inner_margin] = 255
            for tr in self.tracks:
                cv2.circle(mask, tuple(map(int, tr[-1][0])), 12, 0, -1)
            
            new_feature_params = self.feature_params.copy()
            new_feature_params['maxCorners'] = 30
            new_feature_params['qualityLevel'] *= 1.2
            new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **new_feature_params)
            
            if new_corners is not None:
                for corner in new_corners:
                    self.tracks.append([corner])
        
        self.prev_gray = gray.copy()
        success = (self.lost_count <= self.max_lost_frames and len(self.tracks) > 8 and self.bbox is not None)
        return success, self.bbox

    def draw_tracks(self, frame):
        """Enhanced visualization with better track display"""
        if self.bbox is not None:
            x, y, w, h = self.bbox
            thickness = max(2, min(4, len(self.tracks) // 20))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            
            quality = min(100, len(self.tracks))
            quality_color = (0, int(255 * quality / 100), int(255 * (1 - quality / 100)))
            cv2.rectangle(frame, (x, y-25), (x + int(w * quality / 100), y-15), quality_color, -1)
            
            mode_text = f"{self.tracking_mode.upper()}: {len(self.tracks)} pts"
            cv2.putText(frame, mode_text, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for tr in self.tracks:
            track_age = len(tr)
            if track_age > 7: color = (0, 255, 0)
            elif track_age > 3: color = (0, 255, 255)
            else: color = (0, 128, 255)
            
            cv2.circle(frame, tuple(map(int, tr[-1][0])), 3, color, -1)
            
            if len(tr) > 3:
                points = np.array([np.int32(p[0]) for p in tr[-5:]]).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], False, color, 1)
        
        return frame