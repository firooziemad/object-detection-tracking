import cv2
import numpy as np
from ultralytics import YOLO
from CustomTracker import CustomTracker
import sys
import time
import argparse

# --- MANUAL CONFIGURATION (Edit these directly) ---
VIDEO_PATH = "carr3.mp4"        # Change your video path here
TARGET_CLASS_NAME = "car"      # Change object: "person", "car", "dog", "cat", "bicycle", etc.
MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.2  # Lower this for more detections (e.g., 0.3, 0.4)

# YOLO class mapping for easy object switching
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

# Object-specific configurations
OBJECT_PRESETS = {
    "person": {"min_area": 5000, "redetect_min_area": 2000, "tracking_mode": "normal"},
    "car": {"min_area": 8000, "redetect_min_area": 3000, "tracking_mode": "high_motion"},
    "dog": {"min_area": 3000, "redetect_min_area": 1500, "tracking_mode": "high_motion"},
    "cat": {"min_area": 2000, "redetect_min_area": 1000, "tracking_mode": "smooth"},
    "bicycle": {"min_area": 6000, "redetect_min_area": 2500, "tracking_mode": "normal"},
    "motorcycle": {"min_area": 7000, "redetect_min_area": 3000, "tracking_mode": "high_motion"},
    "bottle": {"min_area": 1000, "redetect_min_area": 500, "tracking_mode": "smooth"},
    "laptop": {"min_area": 4000, "redetect_min_area": 2000, "tracking_mode": "smooth"}
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Multi-Object Tracking System')
    parser.add_argument('--object', '-o', type=str, help='Object to track (overrides manual config)')
    parser.add_argument('--video', '-v', type=str, help='Video path (overrides manual config)')
    parser.add_argument('--confidence', '-c', type=float, help='YOLO confidence threshold (0.1-1.0)')
    parser.add_argument('--ad', '--auto-detect', action='store_true', help='Enable auto re-detection mode')
    parser.add_argument('--hd', '--history-detect', action='store_true', help='Enable history re-detection mode')
    parser.add_argument('--mode', '-m', type=str, choices=['smooth', 'normal', 'high_motion'], help='Tracking mode')
    parser.add_argument('--list', '-l', action='store_true', help='List available objects and exit')
    
    return parser.parse_args()

def get_bbox_center(bbox):
    """Get center point of bounding box"""
    x, y, w, h = bbox
    return (x + w/2, y + h/2)

def get_bbox_distance(bbox1, bbox2):
    """Calculate distance between two bounding box centers"""
    center1 = get_bbox_center(bbox1)
    center2 = get_bbox_center(bbox2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def get_bbox_size_similarity(bbox1, bbox2):
    """Calculate size similarity between two bounding boxes (0-1, 1=identical)"""
    _, _, w1, h1 = bbox1
    _, _, w2, h2 = bbox2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    if area1 == 0 or area2 == 0:
        return 0
    
    ratio = min(area1, area2) / max(area1, area2)
    return ratio

def find_best_matching_detection(current_bbox, detections, bbox_history):
    """Find the detection that best matches current tracking using history"""
    if not detections:
        return None
    
    best_detection = None
    best_score = -1
    
    for detection in detections:
        xyxy = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf[0].cpu().numpy()
        detection_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
        
        # Distance score
        distance = get_bbox_distance(current_bbox, detection_bbox)
        max_distance = 200
        distance_score = max(0, 1 - (distance / max_distance))
        
        # Size similarity score
        size_score = get_bbox_size_similarity(current_bbox, detection_bbox)
        
        # History consistency score
        history_score = 0
        if bbox_history:
            history_distances = []
            for hist_bbox in bbox_history[-5:]:
                hist_distance = get_bbox_distance(hist_bbox, detection_bbox)
                history_distances.append(hist_distance)
            
            avg_hist_distance = np.mean(history_distances)
            history_score = max(0, 1 - (avg_hist_distance / max_distance))
        
        # Combined score
        total_score = (distance_score * 0.4 + size_score * 0.3 + 
                      history_score * 0.2 + confidence * 0.1)
        
        if total_score > best_score:
            best_score = total_score
            best_detection = detection
    
    return best_detection

def get_best_detection(boxes, min_area, target_class_id=None):
    """Get the best detection from YOLO results"""
    best_box = None
    best_score = 0
    best_class_id = None
    
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
        if target_class_id is not None and class_id != target_class_id:
            continue
        
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        
        if area > min_area and confidence > best_score:
            best_score = confidence
            best_box = box
            best_class_id = class_id
    
    return best_box, best_score, best_class_id

def get_class_name(class_id):
    """Get class name from class ID"""
    for name, id_val in YOLO_CLASSES.items():
        if id_val == class_id:
            return name
    return f"class_{class_id}"

def list_available_objects():
    """List all available object types"""
    print("\nAvailable preset objects:")
    for name, config in OBJECT_PRESETS.items():
        mode = config.get('tracking_mode', 'normal')
        print(f"  '{name}' -> {mode} mode")
    
    print(f"\nAll YOLO classes ({len(YOLO_CLASSES)} total):")
    for i, (name, class_id) in enumerate(YOLO_CLASSES.items()):
        if i % 4 == 0:
            print()
        print(f"  {name:15} (ID:{class_id:2d})", end="")
    print("\n")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle list request
    if args.list:
        list_available_objects()
        return
    
    # Configure target object (command line overrides manual config)
    target_class_name = args.object if args.object else TARGET_CLASS_NAME
    video_path = args.video if args.video else VIDEO_PATH
    confidence_threshold = args.confidence if args.confidence else YOLO_CONFIDENCE_THRESHOLD
    
    # Get class ID and object config
    if target_class_name.lower() in YOLO_CLASSES:
        target_class_id = YOLO_CLASSES[target_class_name.lower()]
    else:
        print(f"Error: Unknown object '{target_class_name}'")
        list_available_objects()
        return
    
    # Get object-specific settings
    object_config = OBJECT_PRESETS.get(target_class_name.lower(), {
        "min_area": 3000, "redetect_min_area": 1500, "tracking_mode": "normal"
    })
    
    print(f"Configuration:")
    print(f"  Target: {target_class_name} (class_id: {target_class_id})")
    print(f"  Video: {video_path}")
    print(f"  Confidence: {confidence_threshold}")
    print(f"  Min Area: {object_config['min_area']}")
    print(f"  Tracking Mode: {object_config['tracking_mode']}")
    
    # --- INITIALIZATION ---
    print("\nLoading YOLO model...")
    try:
        model = YOLO(MODEL_NAME)
        model.overrides['verbose'] = False
        model.overrides['device'] = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        print(f"Using device: {model.overrides['device']}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    frame_time = 1.0 / fps
    print(f"Video: {total_frames} frames at {fps} FPS")
    
    # Initialize modes (command line can enable them)
    auto_redetect = args.ad if hasattr(args, 'ad') else False
    history_redetection = args.hd if hasattr(args, 'hd') else False
    redetect_counter = 0
    history_counter = 0
    bbox_history = []
    
    # Read first frame
    success, frame = cap.read()
    if not success:
        print("Error reading first frame.")
        cap.release()
        return

    # --- INITIAL DETECTION ---
    print(f"\nDetecting initial {target_class_name}...")
    results = model(frame, verbose=False, classes=[target_class_id], conf=confidence_threshold)
    
    if len(results[0].boxes) == 0:
        print(f"Could not find '{target_class_name}' in the first frame.")
        print("Try:")
        print(f"  python {sys.argv[0]} --confidence 0.3")
        print(f"  python {sys.argv[0]} --list")
        cap.release()
        return
    
    # Get the best detection
    best_box, confidence, detected_class_id = get_best_detection(
        results[0].boxes, object_config['min_area'], target_class_id
    )
    
    if best_box is None:
        print(f"No suitable {target_class_name} detection found.")
        print(f"Detected objects might be smaller than minimum area: {object_config['min_area']}")
        
        # Show what was detected
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            area = width * height
            class_name = get_class_name(class_id)
            print(f"  Found: {class_name} (area: {area:.0f}, conf: {conf:.3f})")
        
        cap.release()
        return
    
    # Extract bounding box
    xyxy = best_box.xyxy[0].cpu().numpy()
    initial_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
    print(f"Initial detection: {initial_bbox} (confidence: {confidence:.3f})")
    
    # Initialize bbox history
    bbox_history.append(initial_bbox)

    # --- INITIALIZE TRACKER ---
    tracker = CustomTracker()
    
    # Set tracking mode from preset or command line
    tracking_mode = args.mode if args.mode else object_config['tracking_mode']
    tracker.set_tracking_mode(tracking_mode)
    print(f"Using {tracking_mode} tracking mode")
    
    if not tracker.init(frame, initial_bbox):
        print("Failed to initialize tracker")
        cap.release()
        return

    print(f"\nControls:")
    print("  'q' - quit")
    print("  'p' - pause/unpause")
    print("  'f' - toggle auto re-detection")
    print("  'e' - toggle history re-detection (every 7 frames)")
    print("  'r' - manual re-detection")
    print("  's/n/h' - smooth/normal/high_motion tracking modes")
    print("  SPACE - step frame when paused")
    print("  'l' - toggle FPS limiting (30 FPS / unlimited)")
    
    if auto_redetect:
        print("Auto re-detection: ENABLED (from command line)")
    if history_redetection:
        print("History re-detection: ENABLED (from command line)")
    
    # --- TRACKING LOOP ---
    frame_count = 0
    redetection_attempts = 0
    paused = False
    last_frame_time = time.time()
    fps_limited = False  # New variable to control FPS limiting
    actual_fps = 0  # Variable to track actual FPS
    fps_start_time = time.time()
    fps_frame_count = 0
    
    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                print("End of video or read error")
                break
            
            frame_count += 1
        
        # --- UPDATE TRACKER ---
        if not paused or frame_count == 1:
            tracking_success, box = tracker.update(frame)
            
            # Update bbox history if tracking is successful
            if tracking_success and box is not None:
                bbox_history.append(box)
                if len(bbox_history) > 20:
                    bbox_history.pop(0)
            
            # History-based re-detection mode
            if history_redetection and box is not None:
                history_counter += 1
                if history_counter >= 7:
                    history_counter = 0
                    print("History-based re-detection...")
                    
                    results = model(frame, verbose=False, classes=[target_class_id], 
                                  conf=confidence_threshold * 0.8)
                    
                    if len(results[0].boxes) > 0:
                        best_match = find_best_matching_detection(box, results[0].boxes, bbox_history)
                        
                        if best_match is not None:
                            xyxy = best_match.xyxy[0].cpu().numpy()
                            confidence = best_match.conf[0].cpu().numpy()
                            new_bbox = (int(xyxy[0]), int(xyxy[1]), 
                                      int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                            
                            distance = get_bbox_distance(box, new_bbox)
                            if distance > 50:
                                if tracker.init(frame, new_bbox):
                                    print(f"History re-detection: {new_bbox} (conf: {confidence:.3f}, dist: {distance:.1f})")
                                    bbox_history.append(new_bbox)
                                    tracking_success, box = True, new_bbox
                            else:
                                print(f"History re-detection: staying with current (dist: {distance:.1f})")
            
            # Auto re-detection mode
            if auto_redetect:
                redetect_counter += 1
                
                if tracker.tracking_mode == "smooth":
                    if redetect_counter >= int(fps):
                        redetect_counter = 0
                        print("Smooth mode auto re-detection...")
                        results = model(frame, verbose=False, classes=[target_class_id], 
                                      conf=confidence_threshold * 0.8)
                        if len(results[0].boxes) > 0:
                            best_box, conf, _ = get_best_detection(
                                results[0].boxes, object_config['redetect_min_area'], target_class_id
                            )
                            
                            if best_box is not None:
                                xyxy = best_box.xyxy[0].cpu().numpy()
                                new_bbox = (int(xyxy[0]), int(xyxy[1]), 
                                          int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                                
                                if tracker.init(frame, new_bbox):
                                    print(f"Smooth auto re-detection successful: {new_bbox} (conf: {conf:.3f})")
                                    tracking_success, box = True, new_bbox
                else:
                    if redetect_counter >= 15:
                        redetect_counter = 0
                        if len(tracker.tracks) < 20:
                            print("Auto re-detection...")
                            results = model(frame, verbose=False, classes=[target_class_id], 
                                          conf=confidence_threshold * 0.8)
                            if len(results[0].boxes) > 0:
                                best_box, conf, _ = get_best_detection(
                                    results[0].boxes, object_config['redetect_min_area'], target_class_id
                                )
                                
                                if best_box is not None:
                                    xyxy = best_box.xyxy[0].cpu().numpy()
                                    new_bbox = (int(xyxy[0]), int(xyxy[1]),
                                              int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                                    
                                    if tracker.init(frame, new_bbox):
                                        print(f"Auto re-detection successful: {new_bbox} (conf: {conf:.3f})")
            
            # Handle tracking failure
            if not tracking_success:
                print(f"Tracking lost at frame {frame_count}")
                if tracker.tracking_mode == "smooth" and frame_count % 20 == 0 and redetection_attempts < 3:
                    print("Attempting automatic re-detection...")
                    results = model(frame, verbose=False, classes=[target_class_id], 
                                  conf=confidence_threshold)
                    
                    if len(results[0].boxes) > 0:
                        best_box, conf, _ = get_best_detection(
                            results[0].boxes, object_config['redetect_min_area'] * 1.5, target_class_id
                        )
                        
                        if best_box is not None:
                            xyxy = best_box.xyxy[0].cpu().numpy()
                            new_bbox = (int(xyxy[0]), int(xyxy[1]),
                                      int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                            
                            if tracker.init(frame, new_bbox):
                                print(f"Re-detection successful: {new_bbox} (conf: {conf:.3f})")
                                redetection_attempts += 1
                                tracking_success, box = True, new_bbox
                else:
                    print(f"Waiting for {target_class_name} to return to frame...")

        # --- VISUALIZATION ---
        display_frame = frame.copy()
        
        # Show tracking status
        if tracking_success and box is not None:
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display_frame, f"Tracking: {len(tracker.tracks)} features", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"TRACKING LOST - Frame {frame_count}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Information overlay
        info_y = 30
        cv2.putText(display_frame, f"Target: {target_class_name}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 20
        cv2.putText(display_frame, f"Mode: {tracker.tracking_mode.upper()}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        info_y += 20
        fps_color = (0, 255, 255) if fps_limited else (255, 255, 255)
        fps_text = f"FPS: {actual_fps:.1f}" + (" (Limited)" if fps_limited else " (Unlimited)")
        cv2.putText(display_frame, fps_text, (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1) 
        if auto_redetect:
            info_y += 20
            cv2.putText(display_frame, f"Auto-redetect: ON", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        if history_redetection:
            info_y += 20
            cv2.putText(display_frame, f"History-redetect: ON ({7-history_counter})", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        if paused:
            cv2.putText(display_frame, "PAUSED - Press 'p' to continue", (10, display_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(f"Multi-Object Tracking - {target_class_name}", display_frame)

        # --- FPS CONTROL ---
        if not paused:
            current_time = time.time()
            
            # Calculate actual FPS
            fps_frame_count += 1
            if current_time - fps_start_time >= 1.0:  # Update every second
                actual_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
            
            # Apply FPS limiting only if enabled
            if fps_limited:
                elapsed = current_time - last_frame_time
                wait_time = max(0, frame_time - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
            
            last_frame_time = current_time

        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
            if not paused:
                last_frame_time = time.time()
        elif key == ord('f'):
            auto_redetect = not auto_redetect
            redetect_counter = 0
            print(f"Auto re-detect: {'ON' if auto_redetect else 'OFF'}")
        elif key == ord('e'):
            history_redetection = not history_redetection
            history_counter = 0
            print(f"History re-detection: {'ON' if history_redetection else 'OFF'}")
        
        elif key == ord('l'):
            fps_limited = not fps_limited
            print(f"FPS limiting: {'ON (30 FPS)' if fps_limited else 'OFF (Unlimited)'}")    
    
        elif key == ord(' ') and paused:
            success, frame = cap.read()
            if success:
                frame_count += 1
            else:
                print("End of video")
                break
        elif key == ord('r'):  # Manual re-detection
            print("Manual re-detection...")
            results = model(frame, verbose=False, classes=[target_class_id], 
                          conf=confidence_threshold * 0.7)
            if len(results[0].boxes) > 0:
                best_box, conf, detected_class = get_best_detection(
                    results[0].boxes, object_config['redetect_min_area'], target_class_id
                )
                
                if best_box is not None:
                    xyxy = best_box.xyxy[0].cpu().numpy()
                    new_bbox = (int(xyxy[0]), int(xyxy[1]),
                               int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                    
                    if tracker.init(frame, new_bbox):
                        detected_name = get_class_name(detected_class)
                        print(f"Manual re-detection successful: {detected_name} at {new_bbox} (conf: {conf:.3f})")
                        bbox_history.append(new_bbox)
                    else:
                        print("Failed to re-initialize tracker")
                else:
                    print(f"No suitable {target_class_name} detection found")
            else:
                print("No detections found")
        elif key == ord('s'):
            tracker.set_tracking_mode("smooth")
            print("Switched to SMOOTH mode")
        elif key == ord('h'):
            tracker.set_tracking_mode("high_motion")
            print("Switched to HIGH MOTION mode")
        elif key == ord('n'):
            tracker.set_tracking_mode("normal")
            print("Switched to NORMAL mode")

        # Check if window was closed
        try:
            window_name = f"Multi-Object Tracking - {target_class_name}"
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    # --- CLEANUP ---
    print(f"Processed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()