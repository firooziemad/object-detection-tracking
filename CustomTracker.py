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
        
        # Kalman Filter for smooth bounding box
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
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # For bounding box estimation
        self.bbox_history = []
        self.max_history = 3
        self.size_smoothing_factor = 0.25  # How much change is allowed per frame
        self.prev_bbox_size = None
        
                ###############################################
        # Add these in __init__ method after existing variables:
        self.tracking_mode = "normal"  # "smooth", "normal", "high_motion"
        self.mode_configs = {
            "smooth": {
                "detect_interval": 8,
                "max_corners": 50,
                "quality_level": 0.03,
                "min_distance": 12,
                "win_size": (19, 19),
                "max_level": 2,
                "max_lost_frames": 20
            },
            "normal": {
                "detect_interval": 5,
                "max_corners": 80,
                "quality_level": 0.02,
                "min_distance": 10,
                "win_size": (15, 15),
                "max_level": 2,
                "max_lost_frames": 15,
                "size_smoothing": 0.08  # Much smoother size changes
            },
            "high_motion": {
                "detect_interval": 3,
                "max_corners": 120,
                "quality_level": 0.015,
                "min_distance": 8,
                "win_size": (13, 13),
                "max_level": 3,
                "max_lost_frames": 10
                
            }
        }
        
        #############################################

    def init(self, frame, bbox):
        """Initialize tracker with first frame and bounding box"""
        self.bbox = bbox
        self.lost_count = 0
        self.frame_idx = 0
        self.tracks = []
        self.bbox_history = [bbox]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self.prev_gray = gray.copy()
        
        # Extract features from the bounding box region with better coverage
        x, y, w, h = bbox
        
        # Create multiple regions within the bbox for better feature distribution
        regions = [
            (x + w//4, y + h//4, w//2, h//2),  # Center region
            (x + w//8, y + h//8, 3*w//4, 3*h//4),  # Larger center
            (x, y, w, h)  # Full bbox
        ]
        
        total_features = 0
        for region in regions:
            rx, ry, rw, rh = region
            mask = np.zeros_like(gray)
            mask[ry:ry+rh, rx:rx+rw] = 255
            
            # Reduce max corners for each region
            region_params = self.feature_params.copy()
            region_params['maxCorners'] = 30
            
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **region_params)
            
            if corners is not None:
                for corner in corners:
                    self.tracks.append([corner])
                total_features += len(corners)
        
        # Initialize Kalman filter
        x, y, w, h = bbox
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        print(f"Initialized with {total_features} feature points in {len(regions)} regions")
        return total_features > 0

##############################################
    def set_tracking_mode(self, mode):
        """Set tracking mode: 'smooth', 'normal', or 'high_motion'"""
        if mode in self.mode_configs:
            self.tracking_mode = mode
            config = self.mode_configs[mode]
            
            # Update parameters
            self.detect_interval = config["detect_interval"]
            self.max_lost_frames = config["max_lost_frames"]
            if "size_smoothing" in config:
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
            
            print(f"Tracking mode set to: {mode}")
            return True
        return False


##############################################
    def update(self, frame):
        """Update tracker with new frame"""
        if self.prev_gray is None:
            return False, None
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self.frame_idx += 1
        
        # Track existing features
        if len(self.tracks) > 0:
            # Get current positions of all tracks
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            
            # Calculate optical flow
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None, **self.lk_params
            )
            
            # Select good points
            good_new = p1[_st == 1]
            good_old = p0[_st == 1]
            
            # Update tracks with good points
            new_tracks = []
            for tr, (x, y) in zip(self.tracks, p1.reshape(-1, 2)):
                if _st[len(new_tracks)] == 1:  # Good track
                    tr.append([(x, y)])
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
            
            self.tracks = new_tracks
            
            # Calculate precise bounding box from tracked points
            if len(good_new) >= 8:  # Need sufficient points
                # Get current positions
                current_points = good_new.reshape(-1, 2)
                
                # Remove outliers using clustering approach
                if len(current_points) > 15:
                    # Calculate distances from median center
                    center = np.median(current_points, axis=0)
                    distances = np.sqrt(np.sum((current_points - center)**2, axis=1))
                    
                    # Keep points within 1.5 * median distance
                    threshold = np.median(distances) * 2.0
                    valid_mask = distances <= threshold
                    current_points = current_points[valid_mask]
                
                if len(current_points) >= 5:
                    # Calculate tight bounding box
                    x_coords = current_points[:, 0]
                    y_coords = current_points[:, 1]
                    
                    # Use percentiles for more robust bounds (removes extreme outliers)
                    x_min = np.percentile(x_coords, 5)
                    x_max = np.percentile(x_coords, 95)
                    y_min = np.percentile(y_coords, 5)
                    y_max = np.percentile(y_coords, 95)
                    
                    # Calculate dimensions
                    detected_w = x_max - x_min
                    detected_h = y_max - y_min
                    
                    # Adaptive padding based on object size
                    padding_w = max(15, detected_w * 0.25)  # Increased from 0.15 to 0.25
                    padding_h = max(20, detected_h * 0.30)  # Increased from 0.20 to 0.30
                    
                    # Apply padding
                    x_min = max(0, x_min - padding_w)
                    y_min = max(0, y_min - padding_h)
                    x_max = min(gray.shape[1], x_max + padding_w)
                    y_max = min(gray.shape[0], y_max + padding_h)
                    
                    new_w = x_max - x_min
                    new_h = y_max - y_min
                    ###########################
                    # After calculating new_w and new_h, add this smoothing:
                    if self.prev_bbox_size is not None:
                        prev_w, prev_h = self.prev_bbox_size
                        
                        # Limit size changes to smooth transitions
                        max_w_change = max(5, prev_w * self.size_smoothing_factor)
                        max_h_change = max(5, prev_h * self.size_smoothing_factor)
                        
                        # Clamp the size changes
                        if abs(new_w - prev_w) > max_w_change:
                            if new_w > prev_w:
                                new_w = prev_w + max_w_change
                            else:
                                new_w = prev_w - max_w_change
                        
                        if abs(new_h - prev_h) > max_h_change:
                            if new_h > prev_h:
                                new_h = prev_h + max_h_change
                            else:
                                new_h = prev_h - max_h_change

                    # Store current size for next frame
                    self.prev_bbox_size = (new_w, new_h)
                    ###########################
                    
                    # Ensure reasonable size
                    if new_w > 15 and new_h > 20:
                        # Use Kalman filter for smooth bounding box
                        self.kalman.predict()
                        measurement = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
                        self.kalman.correct(measurement)
                        
                        # Get smoothed values
                        state = self.kalman.statePost.flatten()
                        smooth_x = max(0, int(state[0]))
                        smooth_y = max(0, int(state[1]))
                        smooth_w = max(15, int(state[2]))
                        smooth_h = max(20, int(state[3]))
                        
                        # Ensure within frame bounds
                        smooth_w = min(smooth_w, gray.shape[1] - smooth_x)
                        smooth_h = min(smooth_h, gray.shape[0] - smooth_y)
                        
                        self.bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
                        self.bbox_history.append(self.bbox)
                        if len(self.bbox_history) > self.max_history:
                            self.bbox_history.pop(0)
                        self.lost_count = 0
                        
                        print(f"Tracking {len(good_new)} features, bbox: {smooth_w}x{smooth_h}")
                    else:
                        self.lost_count += 1
                else:
                    self.lost_count += 1
            else:
                self.lost_count += 1
                print(f"Too few features: {len(good_new) if 'good_new' in locals() else 0}")
        
        # Add new features periodically or when we have too few
        if (self.frame_idx % self.detect_interval == 0 or len(self.tracks) < 20) and self.bbox is not None:
            x, y, w, h = self.bbox
            
            # Create mask to avoid detecting features too close to existing ones
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            
            # Mask out existing features
            for tr in self.tracks:
                cv2.circle(mask, tuple(map(int, tr[-1][0])), 8, 0, -1)
            
            # Detect new features
            new_feature_params = self.feature_params.copy()
            new_feature_params['maxCorners'] = 25
            new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **new_feature_params)
            
            if new_corners is not None:
                for corner in new_corners:
                    self.tracks.append([corner])
                print(f"Added {len(new_corners)} new features")
        
        self.prev_gray = gray.copy()
        
        # Check if tracking is successful
        success = self.lost_count <= self.max_lost_frames and len(self.tracks) > 5
        
        return success, self.bbox

    def draw_tracks(self, frame):
        """Draw tracking visualization"""
        if self.bbox is not None:
            x, y, w, h = self.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw feature points and tracks
        for tr in self.tracks:
            cv2.circle(frame, tuple(map(int, tr[-1][0])), 2, (0, 255, 0), -1)
            
            # Draw track history
            if len(tr) > 1:
                points = np.array([np.int32(p[0]) for p in tr]).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], False, (0, 255, 0), 1)
        
        return frame