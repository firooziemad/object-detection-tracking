import cv2
import numpy as np
from ultralytics import YOLO
from CustomTracker import CustomTracker  # Assuming CustomTracker is in a separate file
import sys
import time
import argparse

# --- MANUAL CONFIGURATION ---
VIDEO_PATH = "person3.mp4"
TARGET_CLASS_NAME = "motorcycle"
MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.4 # Confidence for initial detection
MIN_AREA_INITIAL = 50 # Minimum pixel area to consider an object for tracking

# YOLO class mapping
YOLO_CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34,
    "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38,
    "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43,
    "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48,
    "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
    "donut": 54, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58,
    "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63,
    "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78,
    "toothbrush": 79
}

class MultiObjectTracker:
    """A class to manage tracking multiple objects simultaneously."""
    def __init__(self, tracking_mode='normal'):
        self.trackers = {}
        self.next_tracker_id = 0
        self.tracking_mode = tracking_mode
        self.colors = {}

    def get_color_for_id(self, tracker_id):
        """Generate a consistent, random color for a given tracker ID."""
        if tracker_id not in self.colors:
            np.random.seed(tracker_id)
            self.colors[tracker_id] = (np.random.randint(50, 255), 
                                       np.random.randint(50, 255), 
                                       np.random.randint(50, 255))
        return self.colors[tracker_id]

    def initialize_trackers(self, frame, initial_bboxes):
        """Initializes a new tracker for each bounding box."""
        for bbox in initial_bboxes:
            tracker = CustomTracker()
            tracker.set_tracking_mode(self.tracking_mode)
            
            if tracker.init(frame, bbox):
                self.trackers[self.next_tracker_id] = tracker
                print(f"Initialized tracker ID: {self.next_tracker_id} at {bbox}")
                self.next_tracker_id += 1
        
        print(f"\nSuccessfully initialized {len(self.trackers)} trackers.")

    def update_all(self, frame):
        """Update all active trackers with the new frame."""
        updated_bboxes = {}
        lost_trackers = []
        
        for tracker_id, tracker in self.trackers.items():
            success, box = tracker.update(frame)
            if success:
                updated_bboxes[tracker_id] = box
            else:
                lost_trackers.append(tracker_id)
        
        # Remove trackers that have been lost for too long
        for tracker_id in lost_trackers:
            print(f"Tracker {tracker_id} lost.")
            del self.trackers[tracker_id]
            
        return updated_bboxes

    def draw_all_tracks(self, frame):
        """Draw bounding boxes and IDs for all tracked objects."""
        for tracker_id, tracker in self.trackers.items():
            if tracker.bbox is not None:
                x, y, w, h = tracker.bbox
                color = self.get_color_for_id(tracker_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                label = f"ID: {tracker_id}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def get_all_detections(boxes, min_area, target_class_id):
    """Get all valid detections from YOLO results that meet criteria."""
    valid_detections = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
        if class_id == target_class_id:
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            area = width * height
            
            if area > min_area:
                bbox = (int(xyxy[0]), int(xyxy[1]), int(width), int(height))
                valid_detections.append(bbox)
                
    return valid_detections

def main():
    # --- CONFIGURATION ---
    parser = argparse.ArgumentParser(description='Multi-Object Tracking using YOLO and CustomTracker')
    parser.add_argument('--object', '-o', type=str, default=TARGET_CLASS_NAME, help='Object to track')
    parser.add_argument('--video', '-v', type=str, default=VIDEO_PATH, help='Path to video file')
    parser.add_argument('--confidence', '-c', type=float, default=YOLO_CONFIDENCE_THRESHOLD, help='YOLO confidence threshold')
    parser.add_argument('--mode', '-m', type=str, default='high_motion', choices=['smooth', 'normal', 'high_motion'], help='Tracking mode')
    args = parser.parse_args()

    target_class_name = args.object
    if target_class_name not in YOLO_CLASSES:
        print(f"Error: Object '{target_class_name}' not in YOLO_CLASSES.")
        return
    target_class_id = YOLO_CLASSES[target_class_name]

    print("Configuration:")
    print(f"  - Video: {args.video}")
    print(f"  - Target: {target_class_name} (ID: {target_class_id})")
    print(f"  - Confidence: {args.confidence}")
    print(f"  - Tracking Mode: {args.mode}")
    print(f"  - Min Initial Area: {MIN_AREA_INITIAL}")

    # --- INITIALIZATION ---
    print("\nLoading YOLO model...")
    model = YOLO(MODEL_NAME)
    model.overrides['verbose'] = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        return

    success, frame = cap.read()
    if not success:
        print("Error reading the first frame.")
        cap.release()
        return

    # --- INITIAL DETECTION ---
    print(f"\nDetecting all '{target_class_name}' objects in the first frame...")
    results = model(frame, verbose=False, classes=[target_class_id], conf=args.confidence)
    
    initial_bboxes = get_all_detections(results[0].boxes, MIN_AREA_INITIAL, target_class_id)
    
    if not initial_bboxes:
        print(f"Could not find any '{target_class_name}' objects meeting the criteria in the first frame.")
        print("Try lowering the --confidence or checking the video content.")
        cap.release()
        return
        
    print(f"Found {len(initial_bboxes)} initial objects to track.")
    
    # --- INITIALIZE MULTI-TRACKER ---
    multi_tracker = MultiObjectTracker(tracking_mode=args.mode)
    multi_tracker.initialize_trackers(frame, initial_bboxes)

    # --- TRACKING LOOP ---
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                print("\nEnd of video.")
                break
            frame_count += 1
        
        # Update all trackers
        if not paused:
            multi_tracker.update_all(frame)

        # --- VISUALIZATION ---
        display_frame = frame.copy()
        display_frame = multi_tracker.draw_all_tracks(display_frame)
        
        # Information overlay
        info_text = f"Frame: {frame_count} | Tracking {len(multi_tracker.trackers)} Objects | Mode: {args.mode.upper()}"
        cv2.putText(display_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if paused:
            cv2.putText(display_frame, "PAUSED", (10, display_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f"Multi-Object Tracking - {target_class_name}", display_frame)

        # --- CONTROLS ---
        key = cv2.waitKey(200) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' ') and paused: # Step frame-by-frame when paused
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            multi_tracker.update_all(frame)

    # --- CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cv2.destroyAllWindows()