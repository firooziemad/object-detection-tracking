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
        self.prev_gray = None
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
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.005  # Reduced process noise
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.15  # Slightly increased measurement noise
        
        # For enhanced bounding box estimation
        self.bbox_history = []
        self.max_history = 5
        self.size_smoothing_factor = 0.12  # Even smoother size changes
        self.prev_bbox_size = None
        
        # Enhanced smooth transition system
        self.bbox_buffer = []
        self.buffer_size = 3
        self.confidence_threshold = 0.7
        
        # Tracking mode configurations with enhanced smoothness
        self.tracking_mode = "normal"
        self.mode_configs = {
                    "smooth": {
                        "detect_interval": 8,
                        "max_corners": 60,
                        "quality_level": 0.025,
                        "min_distance": 15,
                        "win_size": (25, 25),
                        "max_level": 2,
                        "max_lost_frames": 25,
                        "size_smoothing": 0.08,  # Increased from 0.06
                        "position_smoothing": 0.8,
                        "kalman_process_noise": 0.003,
                        "kalman_measurement_noise": 0.2
                    },
                    "normal": {
                        "detect_interval": 5,
                        "max_corners": 80,
                        "quality_level": 0.02,
                        "min_distance": 10,
                        "win_size": (19, 19),
                        "max_level": 2,
                        "max_lost_frames": 15,
                        "size_smoothing": 0.12, # Increased from 0.08
                        "position_smoothing": 0.6,
                        "kalman_process_noise": 0.005,
                        "kalman_measurement_noise": 0.15
                    },
                    "high_motion": {
                        "detect_interval": 3,
                        "max_corners": 120,
                        "quality_level": 0.015,
                        "min_distance": 8,
                        "win_size": (15, 15),
                        "max_level": 3,
                        "max_lost_frames": 10,
                        "size_smoothing": 0.18, # Increased from 0.15
                        "position_smoothing": 0.4,
                        "kalman_process_noise": 0.01,
                        "kalman_measurement_noise": 0.1
                    }
                }
        
        
        # Outlier detection for robust tracking
        self.position_history = []
        self.position_history_size = 10
        
        # Motion prediction for enhanced tracking
        self.velocity_history = []
        self.velocity_history_size = 5

    def init(self, frame, bbox):
        """Initialize tracker with first frame and bounding box"""
        self.bbox = bbox
        self.lost_count = 0
        self.frame_idx = 0
        self.tracks = []
        self.bbox_history = [bbox]
        self.bbox_buffer = [bbox]
        self.position_history = [bbox[:2]]
        self.velocity_history = []
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self.prev_gray = gray.copy()
        
        # Enhanced feature extraction with better distribution
        x, y, w, h = bbox
        
        # Create strategic regions for feature detection
        regions = [
            (x + w//6, y + h//6, 2*w//3, 2*h//3),  # Central region
            (x + w//8, y + h//8, 3*w//4, 3*h//4),  # Expanded center
            (x + w//12, y + h//12, 5*w//6, 5*h//6), # Near-full region
            (x, y, w, h)  # Full bbox
        ]
        
        total_features = 0
        for i, region in enumerate(regions):
            rx, ry, rw, rh = region
            mask = np.zeros_like(gray)
            mask[ry:ry+rh, rx:rx+rw] = 255
            
            # Adjust feature parameters for each region
            region_params = self.feature_params.copy()
            region_params['maxCorners'] = [40, 30, 25, 20][i]  # Decreasing corners per region
            region_params['qualityLevel'] = [0.03, 0.025, 0.02, 0.015][i]  # Higher quality for inner regions
            
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **region_params)
            
            if corners is not None:
                for corner in corners:
                    self.tracks.append([corner])
                total_features += len(corners)
        
        # Initialize enhanced Kalman filter
        x, y, w, h = bbox
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.prev_bbox_size = (w, h)
        
        print(f"Enhanced initialization: {total_features} features across {len(regions)} regions")
        return total_features > 0

    def set_tracking_mode(self, mode):
        """Set tracking mode with enhanced parameters"""
        if mode in self.mode_configs:
            self.tracking_mode = mode
            config = self.mode_configs[mode]
            
            # Update parameters
            self.detect_interval = config["detect_interval"]
            self.max_lost_frames = config["max_lost_frames"]
            self.size_smoothing_factor = config["size_smoothing"]
            
            # Update feature detection parameters
            self.feature_params.update({
                'maxCorners': config["max_corners"],
                'qualityLevel': config["quality_level"],
                'minDistance': config["min_distance"]
            })
            
            # Update optical flow parameters
            self.lk_params.update({
                'winSize': config["win_size"],
                'maxLevel': config["max_level"]
            })
            
            # Update Kalman filter noise parameters
            self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * config["kalman_process_noise"]
            self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * config["kalman_measurement_noise"]
            
            print(f"Enhanced tracking mode set to: {mode}")
            return True
        return False

    def detect_outliers(self, points):
        """Detect and remove outlier points using statistical methods"""
        if len(points) < 8:
            return points
        
        # Calculate center and distances
        center = np.median(points, axis=0)
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        
        # Use interquartile range to detect outliers
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        
        # Define outlier threshold (more conservative)
        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR
        
        # Filter outliers
        valid_mask = (distances >= lower_bound) & (distances <= upper_bound)
        return points[valid_mask]

    def predict_motion(self):
        """Predict next position based on velocity history"""
        if len(self.velocity_history) < 2:
            return None
        
        # Calculate average velocity
        avg_velocity = np.mean(self.velocity_history[-3:], axis=0)
        
        # Predict next position
        if self.bbox is not None:
            x, y, w, h = self.bbox
            predicted_x = x + avg_velocity[0]
            predicted_y = y + avg_velocity[1]
            return (predicted_x, predicted_y, w, h)
        
        return None

    def smooth_bbox_transition(self, new_bbox, confidence=1.0):
        """Apply enhanced smoothing to bounding box transitions"""
        if self.bbox is None:
            return new_bbox
        
        current_x, current_y, current_w, current_h = self.bbox
        new_x, new_y, new_w, new_h = new_bbox
        
        # Get smoothing parameters
        config = self.mode_configs[self.tracking_mode]
        position_smoothing = config.get("position_smoothing", 0.5)
        
        # Adaptive smoothing based on confidence and motion
        motion_magnitude = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
        adaptive_factor = min(1.0, motion_magnitude / 50.0)  # Normalize motion
        
        smooth_factor = position_smoothing * confidence * (1.0 - adaptive_factor * 0.3)
        
        # Apply smoothing
        smooth_x = current_x + (new_x - current_x) * smooth_factor
        smooth_y = current_y + (new_y - current_y) * smooth_factor
        smooth_w = current_w + (new_w - current_w) * smooth_factor * 0.7  # Slower size changes
        smooth_h = current_h + (new_h - current_h) * smooth_factor * 0.7
        
        return (int(smooth_x), int(smooth_y), int(smooth_w), int(smooth_h))

    def update(self, frame):
        """Enhanced update method with superior smoothness"""
        if self.prev_gray is None:
            return False, None
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self.frame_idx += 1
        
        # Track existing features with enhanced robustness
        if len(self.tracks) > 0:
            # Get current positions of all tracks
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            
            # Calculate optical flow with error checking
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None, **self.lk_params
            )
            
            # Enhanced point filtering
            good_indices = []
            good_new_points = []
            good_old_points = []
            
            for i, (track, new_point, status, error) in enumerate(zip(self.tracks, p1, _st, _err)):
                if status == 1 and error < 50:  # Error threshold
                    good_indices.append(i)
                    good_new_points.append(new_point)
                    good_old_points.append(p0[i])
            
            # Update tracks with good points only
            new_tracks = []
            for i, new_point in zip(good_indices, good_new_points):
                track = self.tracks[i]
                track.append([new_point[0]])
                if len(track) > self.track_len:
                    del track[0]
                new_tracks.append(track)
            
            self.tracks = new_tracks
            
            # Enhanced bounding box calculation
            if len(good_new_points) >= 8:
                current_points = np.array(good_new_points).reshape(-1, 2)
                
                # Remove outliers using enhanced method
                filtered_points = self.detect_outliers(current_points)
                
                if len(filtered_points) >= 6:
                    # Calculate robust bounding box using percentiles
                    x_coords = filtered_points[:, 0]
                    y_coords = filtered_points[:, 1]
                    
                    # Use percentiles for robust bounds
                    x_min = np.percentile(x_coords, 8)   # More conservative percentiles
                    x_max = np.percentile(x_coords, 92)
                    y_min = np.percentile(y_coords, 8)
                    y_max = np.percentile(y_coords, 92)
                    
                    # Calculate dimensions
                    detected_w = x_max - x_min
                    detected_h = y_max - y_min
                    
                    # Enhanced adaptive padding based on tracking mode
                    if self.tracking_mode == "smooth":
                        padding_w = max(20, detected_w * 0.35)
                        padding_h = max(25, detected_h * 0.40)
                    elif self.tracking_mode == "normal":
                        padding_w = max(15, detected_w * 0.25)
                        padding_h = max(20, detected_h * 0.30)
                    else:  # high_motion
                        padding_w = max(10, detected_w * 0.20)
                        padding_h = max(15, detected_h * 0.25)
                    
                    # Apply padding
                    x_min = max(0, x_min - padding_w)
                    y_min = max(0, y_min - padding_h)
                    x_max = min(gray.shape[1], x_max + padding_w)
                    y_max = min(gray.shape[0], y_max + padding_h)
                    
                    new_w = x_max - x_min
                    new_h = y_max - y_min
                    
                    # Enhanced size smoothing
                    if self.prev_bbox_size is not None:
                        prev_w, prev_h = self.prev_bbox_size
                        
                        # Dynamic smoothing based on size change magnitude
                        size_change = abs(new_w - prev_w) + abs(new_h - prev_h)
                        adaptive_smoothing = self.size_smoothing_factor * (1 + size_change / 100.0)
                        adaptive_smoothing = min(adaptive_smoothing, 0.3)  # Cap the smoothing
                        
                        # Apply size constraints
                        max_w_change = max(8, prev_w * adaptive_smoothing)
                        max_h_change = max(8, prev_h * adaptive_smoothing)
                        
                        # Smooth size transitions
                        if abs(new_w - prev_w) > max_w_change:
                            new_w = prev_w + np.sign(new_w - prev_w) * max_w_change
                        
                        if abs(new_h - prev_h) > max_h_change:
                            new_h = prev_h + np.sign(new_h - prev_h) * max_h_change
                    
                    # Store current size for next frame
                    self.prev_bbox_size = (new_w, new_h)
                    
                    # Ensure reasonable size constraints
                    if new_w > 20 and new_h > 25:
                        # Enhanced Kalman filtering
                        self.kalman.predict()
                        measurement = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
                        self.kalman.correct(measurement)
                        
                        # Get smoothed values with enhanced processing
                        state = self.kalman.statePost.flatten()
                        kalman_x = max(0, int(state[0]))
                        kalman_y = max(0, int(state[1]))
                        kalman_w = max(20, int(state[2]))
                        kalman_h = max(25, int(state[3]))
                        
                        # Ensure within frame bounds
                        kalman_w = min(kalman_w, gray.shape[1] - kalman_x)
                        kalman_h = min(kalman_h, gray.shape[0] - kalman_y)
                        
                        new_bbox = (kalman_x, kalman_y, kalman_w, kalman_h)
                        
                        # Apply additional smoothing transition
                        confidence = min(1.0, len(filtered_points) / 50.0)
                        self.bbox = self.smooth_bbox_transition(new_bbox, confidence)
                        
                        # Update history and velocity
                        self.bbox_history.append(self.bbox)
                        if len(self.bbox_history) > self.max_history:
                            self.bbox_history.pop(0)
                        
                        # Update velocity history
                        if len(self.position_history) > 0:
                            prev_pos = self.position_history[-1]
                            current_pos = self.bbox[:2]
                            velocity = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
                            self.velocity_history.append(velocity)
                            if len(self.velocity_history) > self.velocity_history_size:
                                self.velocity_history.pop(0)
                        
                        # Update position history
                        self.position_history.append(self.bbox[:2])
                        if len(self.position_history) > self.position_history_size:
                            self.position_history.pop(0)
                        
                        self.lost_count = 0
                        
                        if self.frame_idx % 30 == 0:  # Reduced logging frequency
                            print(f"Enhanced tracking: {len(filtered_points)} features, bbox: {kalman_w}x{kalman_h}")
                    else:
                        self.lost_count += 1
                else:
                    self.lost_count += 1
            else:
                self.lost_count += 1
                if len(good_new_points) > 0:
                    print(f"Insufficient features: {len(good_new_points)}")
        
        # Enhanced feature detection with strategic placement
        if (self.frame_idx % self.detect_interval == 0 or len(self.tracks) < 25) and self.bbox is not None:
            x, y, w, h = self.bbox
            
            # Create enhanced mask for feature detection
            mask = np.zeros_like(gray)
            
            # Focus on central regions for stability
            inner_margin = min(w//8, h//8, 15)
            mask[y+inner_margin:y+h-inner_margin, x+inner_margin:x+w-inner_margin] = 255
            
            # Mask out existing features with larger radius for better distribution
            for tr in self.tracks:
                cv2.circle(mask, tuple(map(int, tr[-1][0])), 12, 0, -1)
            
            # Detect new features with enhanced parameters
            new_feature_params = self.feature_params.copy()
            new_feature_params['maxCorners'] = 30
            new_feature_params['qualityLevel'] *= 1.2  # Higher quality threshold
            
            new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **new_feature_params)
            
            if new_corners is not None:
                for corner in new_corners:
                    self.tracks.append([corner])
                if self.frame_idx % 60 == 0:  # Reduced logging
                    print(f"Enhanced feature addition: {len(new_corners)} new features")
        
        self.prev_gray = gray.copy()
        
        # Enhanced success criteria
        success = (self.lost_count <= self.max_lost_frames and 
                  len(self.tracks) > 8 and 
                  self.bbox is not None)
        
        return success, self.bbox

    def draw_tracks(self, frame):
        """Enhanced visualization with better track display"""
        if self.bbox is not None:
            x, y, w, h = self.bbox
            
            # Draw main bounding box with thickness based on tracking quality
            thickness = max(2, min(4, len(self.tracks) // 20))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            
            # Draw tracking quality indicator
            quality = min(100, len(self.tracks))
            quality_color = (0, int(255 * quality / 100), int(255 * (1 - quality / 100)))
            cv2.rectangle(frame, (x, y-25), (x + int(w * quality / 100), y-15), quality_color, -1)
            
            # Add tracking mode indicator
            mode_text = f"{self.tracking_mode.upper()}: {len(self.tracks)} pts"
            cv2.putText(frame, mode_text, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw enhanced feature points
        for i, tr in enumerate(self.tracks):
            # Color based on track age
            track_age = len(tr)
            if track_age > 7:
                color = (0, 255, 0)  # Green for stable tracks
            elif track_age > 3:
                color = (0, 255, 255)  # Yellow for medium tracks
            else:
                color = (0, 128, 255)  # Orange for new tracks
            
            cv2.circle(frame, tuple(map(int, tr[-1][0])), 3, color, -1)
            
            # Draw track history for stable tracks
            if len(tr) > 3:
                points = np.array([np.int32(p[0]) for p in tr[-5:]]).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], False, color, 1)
        
        return frame