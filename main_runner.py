import cv2
import numpy as np
from ultralytics import YOLO
from custom_tracker import CustomTracker
import sys
import time
import argparse

VIDEO_PATH = "person4.mp4"
TARGET_CLASS_NAME = "person"
MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.6 # A balanced confidence
INITIAL_SEARCH_FRAMES = 20
Res = 600
MAX_REENTRY_DISTANCE = 250
EDGE_MARGINS = {
    "top": 30,
    "bottom": 30,
    "left": 30,
    "right": 30
}

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
OBJECT_PRESETS = {
    "person": {"min_area": 500, "redetect_min_area": 200, "tracking_mode": "normal"},
    "car": {"min_area": 100, "redetect_min_area": 50, "tracking_mode": "high_motion"},
    "dog": {"min_area": 3000, "redetect_min_area": 1500, "tracking_mode": "high_motion"},
    "cat": {"min_area": 2000, "redetect_min_area": 1000, "tracking_mode": "smooth"},
    "bicycle": {"min_area": 6000, "redetect_min_area": 2500, "tracking_mode": "normal"},
    "motorcycle": {"min_area": 7000, "redetect_min_area": 3000, "tracking_mode": "high_motion"},
    "bottle": {"min_area": 1000, "redetect_min_area": 500, "tracking_mode": "smooth"},
    "laptop": {"min_area": 4000, "redetect_min_area": 2000, "tracking_mode": "smooth"}
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Multi-Object Tracking System with Edge Detection')
    parser.add_argument('--object', '-o', type=str, help='Object to track (overrides manual config)')
    parser.add_argument('--video', '-v', type=str, help='Video path (overrides manual config)')
    parser.add_argument('--confidence', '-c', type=float, help='YOLO confidence threshold (0.1-1.0)')
    parser.add_argument('--mode', '-m', type=str, choices=['smooth', 'normal', 'high_motion'], help='Tracking mode')
    parser.add_argument('--list', '-l', action='store_true', help='List available objects and exit')
    parser.add_argument('--margin-top', type=int, help='Top edge margin in pixels')
    parser.add_argument('--margin-bottom', type=int, help='Bottom edge margin in pixels')
    parser.add_argument('--margin-left', type=int, help='Left edge margin in pixels')
    parser.add_argument('--margin-right', type=int, help='Right edge margin in pixels')
    parser.add_argument('--margin-all', type=int, help='Set all margins to same value')
    parser.add_argument('--reentry-distance', type=int, help='Maximum pixels for re-entry acceptance (default: 100)')
    
    return parser.parse_args()

def is_bbox_in_margin(bbox, frame_shape, margins):
    """
    Check if a smaller, centered 'inner box' (half the size) 
    has touched the edge margins.
    """
    if bbox is None:
        return False
    
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]
    inner_w = w // 2
    inner_h = h // 2
    inner_x = x + (w - inner_w) // 2
    inner_y = y + (h - inner_h) // 2
    in_top_margin = inner_y < margins["top"]
    in_bottom_margin = (inner_y + inner_h) > (frame_height - margins["bottom"])
    in_left_margin = inner_x < margins["left"]
    in_right_margin = (inner_x + inner_w) > (frame_width - margins["right"])
    
    return in_top_margin or in_bottom_margin or in_left_margin or in_right_margin

def calculate_bbox_features(frame, bbox):
    """Calculate features for bbox verification"""
    if bbox is None:
        return None
    
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    
    if roi.size == 0:
        return None
    
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi
    
    features = {
        'size': (w, h),
        'aspect_ratio': w / h if h > 0 else 1.0,
        'histogram': cv2.calcHist([roi_gray], [0], None, [32], [0, 256]),
        'mean_intensity': np.mean(roi_gray),
        'center': (x + w//2, y + h//2)
    } 
    return features
def compare_bbox_features(features1, features2, threshold=0.7):
    """Compare two sets of bbox features"""
    if features1 is None or features2 is None:
        return False
    
    w1, h1 = features1['size']
    w2, h2 = features2['size']
    size_sim = min(w1*h1, w2*h2) / max(w1*h1, w2*h2) if max(w1*h1, w2*h2) > 0 else 0
    
    ar_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
    ar_sim = max(0, 1 - ar_diff)
    
    hist_corr = cv2.compareHist(features1['histogram'], features2['histogram'], cv2.HISTCMP_CORREL)
    
    intensity_diff = abs(features1['mean_intensity'] - features2['mean_intensity']) / 255.0
    intensity_sim = max(0, 1 - intensity_diff)
    
    combined_score = (size_sim * 0.3 + ar_sim * 0.2 + hist_corr * 0.3 + intensity_sim * 0.2)
    
    return combined_score >= threshold

def get_bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w/2, y + h/2)

def get_bbox_distance(bbox1, bbox2):
    center1 = get_bbox_center(bbox1)
    center2 = get_bbox_center(bbox2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def get_bbox_size_similarity(bbox1, bbox2):
    _, _, w1, h1 = bbox1
    _, _, w2, h2 = bbox2 
    area1 = w1 * h1
    area2 = w2 * h2
    if area1 == 0 or area2 == 0:
        return 0
    ratio = min(area1, area2) / max(area1, area2)
    return ratio

def find_best_matching_detection(current_box, detections, history):

    best_score = -1
    best_match = None
    ref_box = history[-1] if len(history) > 0 else current_box
    for detection in detections:
        xyxy = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf[0].cpu().numpy()
        new_bbox = (int(xyxy[0]), int(xyxy[1]), 
                    int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))

        distance = get_bbox_distance(ref_box, new_bbox)
        size_sim = get_bbox_size_similarity(ref_box, new_bbox)
        max_dist = 300
        dist_score = max(0, 1 - (distance / max_dist))
        combined_score = (dist_score * 0.4) + (size_sim * 0.4) + (confidence * 0.2)

        if combined_score > best_score:
            best_score = combined_score
            best_match = detection
            
    if best_score > 0.5:
        return best_match
        
    return None

def get_best_detection(boxes, min_area, target_class_id=None):
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
    for name, id_val in YOLO_CLASSES.items():
        if id_val == class_id:
            return name
    return f"class_{class_id}"

def list_available_objects():
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

def draw_margin_lines(frame, margins):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.line(overlay, (0, margins["top"]), (width, margins["top"]), (255, 255, 0), 2)
    cv2.line(overlay, (0, height - margins["bottom"]), (width, height - margins["bottom"]), (255, 255, 0), 2)
    cv2.line(overlay, (margins["left"], 0), (margins["left"], height), (255, 255, 0), 2)
    cv2.line(overlay, (width - margins["right"], 0), (width - margins["right"], height), (255, 255, 0), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def main():
    args = parse_arguments()
    
    if args.list:
        list_available_objects()
        return
    target_class_name = args.object if args.object else TARGET_CLASS_NAME
    video_path = args.video if args.video else VIDEO_PATH
    confidence_threshold = args.confidence if args.confidence else YOLO_CONFIDENCE_THRESHOLD
    
    margins = EDGE_MARGINS.copy()
    if args.margin_all:
        margins = {k: args.margin_all for k in margins.keys()}
    else:
        if args.margin_top: margins["top"] = args.margin_top
        if args.margin_bottom: margins["bottom"] = args.margin_bottom
        if args.margin_left: margins["left"] = args.margin_left
        if args.margin_right: margins["right"] = args.margin_right
    
    max_reentry_distance = args.reentry_distance if args.reentry_distance else MAX_REENTRY_DISTANCE

    if target_class_name.lower() in YOLO_CLASSES:
        target_class_id = YOLO_CLASSES[target_class_name.lower()]
    else:
        print(f"Error: Unknown object '{target_class_name}'")
        list_available_objects()
        return
    
    object_config = OBJECT_PRESETS.get(target_class_name.lower(), {
        "min_area": 3000, "redetect_min_area": 1500, "tracking_mode": "normal"
    })
    
    print(f"Configuration:")
    print(f"  Target: {target_class_name} (class_id: {target_class_id})")
    print(f"  Video: {video_path}")
    print(f"  Confidence: {confidence_threshold}")
    print(f"  Tracking Mode: {object_config['tracking_mode']}")
    print(f"  Edge Margins: {margins}")
    print(f"  Max Re-entry Distance: {max_reentry_distance}px")
    
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
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    frame_time = 1.0 / fps
    
    initial_bbox = None
    initial_frame = None
    confidence = 0.0
    
    print(f"\nSearching for '{target_class_name}' in the first {INITIAL_SEARCH_FRAMES} frames...")

    for frame_num in range(INITIAL_SEARCH_FRAMES):
        success, frame = cap.read()
        if not success:
            print(f"  -> End of video or read error at frame {frame_num + 1}.")
            cap.release()
            return
            
        print(f"  Analyzing frame {frame_num + 1}/{INITIAL_SEARCH_FRAMES}...")
        results = model(frame, imgsz=Res, verbose=False, classes=[target_class_id], conf=confidence_threshold)
        
        best_box, conf, detected_class_id = get_best_detection(
            results[0].boxes, object_config['min_area'], target_class_id
        )
        
        if best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            initial_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
            initial_frame = frame
            confidence = conf
            print(f"Initial detection successful in frame {frame_num + 1}: {initial_bbox} (confidence: {confidence:.3f})")
            break

    if initial_bbox is None:
        print(f"\nCould not find a suitable '{target_class_name}' in the first {INITIAL_SEARCH_FRAMES} frames.")
        cap.release()
        return

    frame = initial_frame
    
    tracker = CustomTracker()
    tracking_mode = args.mode if args.mode else object_config['tracking_mode']
    tracker.set_tracking_mode(tracking_mode)
    
    if not tracker.init(frame, initial_bbox):
        print("Failed to initialize main tracker")
        cap.release()
        return
    aux_tracker = None
    tracker_types = ['MIL']
    for tracker_type in tracker_types:
        try:
            if hasattr(cv2, f'Tracker{tracker_type}_create'):
                aux_tracker = getattr(cv2, f'Tracker{tracker_type}_create')()
                aux_tracker.init(frame, initial_bbox)
                print(f"Initialized auxiliary {tracker_type} tracker")
                break
            elif hasattr(cv2.legacy, f'Tracker{tracker_type}_create'):
                aux_tracker = getattr(cv2.legacy, f'Tracker{tracker_type}_create')()
                aux_tracker.init(frame, initial_bbox)
                print(f"Initialized auxiliary {tracker_type} tracker (legacy)")
                break
        except Exception as e:
            print(f"Failed to initialize {tracker_type} tracker: {e}")
            continue
    
    aux_tracker_available = aux_tracker is not None
    reference_features = calculate_bbox_features(frame, initial_bbox)
    bbox_history = [initial_bbox]
    aux_features_history = [reference_features]
    
    print(f"\nControls:")
    print("  'q' - quit, 'p' - pause")
    print("  'r' - correct current tracker")
    print("  'd' - hard reset detection to best new object")
    print("  's/n/h' - tracking modes, 'f' - toggle FPS limit")
    
    frame_count = 0
    paused = False
    show_margins = False
    last_frame_time = time.time()
    fps_limited = False
    actual_fps = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    auto_redetect = True
    redetect_counter = 0
    object_in_margin = False
    last_valid_bbox = initial_bbox
    pending_verification = False
    verification_counter = 0
    pending_bbox = None
    aux_update_counter = 0
    background_detection_counter = 0
    exit_position = None
    box = initial_bbox
    tracking_success = True

    while True:
        if not paused:
            if frame_count > 0:
                success, frame = cap.read()
                if not success:
                    print("End of video or read error")
                    break
            frame_count += 1
            
        if object_in_margin:
            background_detection_counter += 1
            if background_detection_counter >= 5:
                background_detection_counter = 0
                results = model(frame, imgsz=Res, verbose=False, classes=[target_class_id], conf=confidence_threshold * 0.7)
                
                if len(results[0].boxes) > 0:
                    best_match = None
                    best_score = 0
                    
                    for box_det in results[0].boxes:
                        xyxy = box_det.xyxy[0].cpu().numpy()
                        det_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                        
                        if not is_bbox_in_margin(det_bbox, frame.shape, margins):
                            det_center = get_bbox_center(det_bbox)
                            distance_from_exit = float('inf')
                            if exit_position is not None:
                                distance_from_exit = np.sqrt((det_center[0] - exit_position[0])**2 + (det_center[1] - exit_position[1])**2)
                            if distance_from_exit <= max_reentry_distance:
                                det_features = calculate_bbox_features(frame, det_bbox)
                                match_score = 0
                                if aux_tracker_available and len(aux_features_history) > 0:
                                    for aux_feat in aux_features_history[-3:]:
                                        if compare_bbox_features(det_features, aux_feat, 0.5):
                                            match_score += 1
                                else:
                                    distance = get_bbox_distance(last_valid_bbox, det_bbox)
                                    size_sim = get_bbox_size_similarity(last_valid_bbox, det_bbox)
                                    if distance < 150 and size_sim > 0.5:
                                        match_score = 2
                                
                                if match_score > best_score:
                                    best_score = match_score
                                    best_match = box_det
                    
                    if best_match is not None and best_score >= 1:
                        xyxy = best_match.xyxy[0].cpu().numpy()
                        new_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))

                        pending_verification = True
                        verification_counter = 1
                        pending_bbox = new_bbox
                        object_in_margin = False
                        
                        print(f"Re-entry candidate found. Starting 3-frame verification...")

        elif pending_verification:
            results = model(frame, imgsz=Res, verbose=False, classes=[target_class_id], conf=confidence_threshold * 0.7)
            best_match = None
            if len(results[0].boxes) > 0:
                best_match = find_best_matching_detection(pending_bbox, results[0].boxes, [pending_bbox])

            if best_match is not None:
                verification_counter += 1
                xyxy = best_match.xyxy[0].cpu().numpy()
                pending_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                print(f"Verification success: {verification_counter}/3")

                if verification_counter >= 3:
                    print("Verification successful! Resuming normal tracking.")
                    if tracker.init(frame, pending_bbox):
                        if aux_tracker_available:
                            try:
                                aux_tracker.init(frame, pending_bbox)
                            except Exception as e:
                                print(f"Failed to reinitialize auxiliary tracker: {e}")
                        last_valid_bbox = pending_bbox
                        box = pending_bbox
                        tracking_success = True
                    
                    pending_verification = False
                    verification_counter = 0
                    pending_bbox = None
            else:
                print("Verification failed. Returning to search mode.")
                pending_verification = False
                verification_counter = 0
                pending_bbox = None
                object_in_margin = True

        else: # Normal Tracking
            if not paused:
                tracking_success, box = tracker.update(frame)

            current_in_margin = is_bbox_in_margin(box, frame.shape, margins)

            if tracking_success and box is not None and not current_in_margin and aux_tracker_available:
                aux_update_counter += 1
                if aux_update_counter >= 5:
                    aux_update_counter = 0
                    try:
                        aux_success, aux_box = aux_tracker.update(frame)
                        if aux_success:
                            aux_features = calculate_bbox_features(frame, aux_box)
                            if aux_features:
                                aux_features_history.append(aux_features)
                                if len(aux_features_history) > 10:
                                    aux_features_history.pop(0)
                    except Exception as e:
                        aux_tracker_available = False
                last_valid_bbox = box

            if current_in_margin and not object_in_margin:
                if box is not None:
                    exit_position = get_bbox_center(box)
                object_in_margin = True
                background_detection_counter = 0

            if auto_redetect and not object_in_margin and tracking_success and box is not None:
                bbox_history.append(box)
                if len(bbox_history) > 20:
                    bbox_history.pop(0)
                redetect_counter += 1
                if redetect_counter >= 8:
                    redetect_counter = 0

                    results = model(frame, imgsz=Res, verbose=False, classes=[target_class_id], conf=confidence_threshold * 0.78)
                    
                    if len(results[0].boxes) > 0:
                        best_match = find_best_matching_detection(box, results[0].boxes, bbox_history)
                        
                        if best_match is not None:
                            xyxy = best_match.xyxy[0].cpu().numpy()
                            confidence = best_match.conf[0].cpu().numpy()
                            new_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                            
                            distance = get_bbox_distance(box, new_bbox)
                            size_sim = get_bbox_size_similarity(box, new_bbox)
                            
                            if distance > 40 or size_sim < 0.75:
                                if tracker.init(frame, new_bbox):
                                    print(f"Auto-Correction successful: Dist: {distance:.1f}px, SizeSim: {size_sim:.2f}")
                                    box = new_bbox

        display_frame = frame.copy()
        if show_margins:
            draw_margin_lines(display_frame, margins)
        
        if pending_verification:
            status_text = f"Verifying Re-entry... ({verification_counter}/3)"
            cv2.putText(display_frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif tracking_success and box is not None and not object_in_margin:
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display_frame, f"Tracking: {len(tracker.tracks)} features", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif object_in_margin:
            cv2.putText(display_frame, "Object in margin - Searching...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"TRACKING LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        info_y = 30
        cv2.putText(display_frame, f"Target: {target_class_name}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        fps_color = (0, 255, 255) if fps_limited else (255, 255, 255)
        fps_text = f"FPS: {actual_fps:.1f}" + (" (Limited)" if fps_limited else "")
        cv2.putText(display_frame, fps_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1)

        if paused:
            cv2.putText(display_frame, "PAUSED", (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(f"Enhanced Tracking - {target_class_name}", display_frame)

        if not paused:
            current_time = time.time()
            
            fps_frame_count += 1
            if current_time - fps_start_time >= 1.0:
                actual_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
            
            if fps_limited:
                elapsed = current_time - last_frame_time
                wait_time = max(0, frame_time - elapsed)
                if wait_time > 0:
                    time.sleep(wait_time)
            
            last_frame_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('m'):
            show_margins = not show_margins
        elif key == ord('f'):
            fps_limited = not fps_limited
            print(f"FPS Limiter: {'ON' if fps_limited else 'OFF'}")
        elif key == ord(' ') and paused:
            success, frame = cap.read()
            if success:
                frame_count += 1
            else:
                break
        elif key == ord('r'):
            print("Manual re-detection (correction)...")
            results = model(frame, imgsz=Res, verbose=False, classes=[target_class_id], conf=confidence_threshold * 0.7)
                          
            if len(results[0].boxes) > 0 and box is not None:
                best_match = find_best_matching_detection(box, results[0].boxes, bbox_history)
                
                if best_match is not None:
                    xyxy = best_match.xyxy[0].cpu().numpy()
                    conf = best_match.conf[0].cpu().numpy()
                    detected_class = int(best_match.cls[0].cpu().numpy())
                    new_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                    if tracker.init(frame, new_bbox):
                        if aux_tracker_available:
                            try:
                                aux_tracker.init(frame, new_bbox)
                            except Exception as e:
                                print(f"Failed to reinitialize auxiliary tracker: {e}")
                        
                        detected_name = get_class_name(detected_class)
                        print(f"Correction successful: {detected_name} at {new_bbox} (conf: {conf:.3f})")
                        object_in_margin = False
                        last_valid_bbox = new_bbox
                        box = new_bbox
                else:
                    print(f"No suitable matching {target_class_name} found.")
            else:
                print("No detections found for manual override.")

        elif key == ord('d'):
            print("Hard re-detection initiated: Searching for best new object...")
            results = model(frame, imgsz=Res, verbose=False, classes=[target_class_id], conf=confidence_threshold)

            best_box, conf, detected_class_id = get_best_detection(
                results[0].boxes,
                object_config['min_area'],
                target_class_id
            )

            if best_box is not None:
                xyxy = best_box.xyxy[0].cpu().numpy()
                new_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                
                if tracker.init(frame, new_bbox):
                    if aux_tracker_available:
                        try:
                            aux_tracker.init(frame, new_bbox)
                        except Exception as e:
                            print(f"Failed to reinitialize auxiliary tracker: {e}")
                    
                    detected_name = get_class_name(detected_class_id)
                    print(f"Hard reset successful: New target is {detected_name} at {new_bbox} (conf: {conf:.3f})")
                    object_in_margin = False
                    last_valid_bbox = new_bbox
                    box = new_bbox
                    bbox_history = [new_bbox] 
                    aux_features_history = [calculate_bbox_features(frame, new_bbox)] 
                else:
                    print("Failed to initialize tracker on new object.")
            else:
                print(f"No suitable '{target_class_name}' found on screen for hard reset.")

        elif key == ord('s'):
            tracker.set_tracking_mode("smooth")
        elif key == ord('h'):
            tracker.set_tracking_mode("high_motion")
        elif key == ord('n'):
            tracker.set_tracking_mode("normal")

        try:
            if cv2.getWindowProperty(f"Enhanced Tracking - {target_class_name}", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
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