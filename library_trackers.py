import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. MASTER CONFIGURATION ---
ENABLE_HYBRID_MODE = True
# --- NEW: Set to False to disable specific object identification ---
ENABLE_RE_IDENTIFICATION = True

VIDEO_PATH = "person1.mp4"
TARGET_CLASS_NAME = "person"
MODEL_NAME = "yolov8n.pt"
TRACKER_TYPE = "MOSSE"

# --- 2. TUNING PARAMETERS (for Hybrid Mode) ---
HEALTH_CHECK_INTERVAL = 45
REDETECTION_INTERVAL = 15
SCALE_SMOOTHING_FACTOR = 0.2
YOLO_CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
# Set this to a lower value like 0.45 if Re-ID is too strict
REID_SIMILARITY_THRESHOLD = 0.45 
FEATURE_UPDATE_RATE = 0.05

def create_tracker(tracker_type):
    if tracker_type == 'CSRT': tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF': tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE': tracker = cv2.legacy.TrackerMOSSE_create()
    else: raise ValueError("Invalid tracker type specified.")
    return tracker

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea, boxBArea = boxA[2] * boxA[3], boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou

def get_color_histogram(frame, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if w <= 0 or h <= 0: return None
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"Error opening video {VIDEO_PATH}"); return
    success, frame = cap.read()
    if not success: print("Error reading first frame."); return

    target_class_id = list(model.names.keys())[list(model.names.values()).index(TARGET_CLASS_NAME.lower())]
    results = model(frame, verbose=False, classes=[target_class_id])
    
    initial_bbox, tracked_features = None, None
    if len(results[0].boxes) > 0:
        xyxy = results[0].boxes[0].xyxy[0].cpu().numpy()
        initial_bbox = tuple(map(int, [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]))
        if ENABLE_RE_IDENTIFICATION:
            tracked_features = get_color_histogram(frame, initial_bbox)
            if tracked_features is None: print("Initial object has invalid size."); return
            print(f"Found '{TARGET_CLASS_NAME}' and extracted its feature signature.")

    if not initial_bbox: print("Target not found in first frame."); return

    tracker = create_tracker(TRACKER_TYPE)
    tracker.init(frame, initial_bbox)

    if ENABLE_HYBRID_MODE:
        run_hybrid_mode(cap, tracker, model, initial_bbox, tracked_features)
    else:
        run_pure_tracking_mode(cap, tracker)

    cap.release()
    cv2.destroyAllWindows()

def run_pure_tracking_mode(cap, tracker):
    while True:
        success, frame = cap.read()
        if not success: break
        timer = cv2.getTickCount()
        tracking_success, box = tracker.update(frame)
        if tracking_success:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {int(fps)} ({TRACKER_TYPE})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Pure Tracking Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def run_hybrid_mode(cap, tracker, model, initial_bbox, tracked_features):
    tracking_active, frames_since_lost, frame_count = True, 0, 0
    current_size, target_size = (initial_bbox[2], initial_bbox[3]), (initial_bbox[2], initial_bbox[3])
    target_class_id = list(model.names.keys())[list(model.names.values()).index(TARGET_CLASS_NAME.lower())]

    while True:
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        timer = cv2.getTickCount()
        final_box_to_draw = None

        if tracking_active:
            tracking_success, box = tracker.update(frame)
            is_healthy = True
            if tracking_success and frame_count % HEALTH_CHECK_INTERVAL == 0:
                det_results = model(frame, verbose=False, imgsz=320, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[target_class_id])
                if len(det_results[0].boxes) > 0:
                    det_box = tuple(map(int, det_results[0].boxes[0].xyxy[0].cpu().numpy()))
                    det_box = (det_box[0], det_box[1], det_box[2] - det_box[0], det_box[3] - det_box[1])
                    if calculate_iou(box, det_box) < IOU_THRESHOLD: is_healthy = False
                    else:
                        target_size = (det_box[2], det_box[3])
                        if ENABLE_RE_IDENTIFICATION:
                            current_healthy_features = get_color_histogram(frame, det_box)
                            if current_healthy_features is not None:
                                cv2.addWeighted(current_healthy_features, FEATURE_UPDATE_RATE, tracked_features, 1 - FEATURE_UPDATE_RATE, 0, tracked_features)
                else: is_healthy = False
            if tracking_success and is_healthy:
                w, h = int(current_size[0] * (1 - SCALE_SMOOTHING_FACTOR) + target_size[0] * SCALE_SMOOTHING_FACTOR), int(current_size[1] * (1 - SCALE_SMOOTHING_FACTOR) + target_size[1] * SCALE_SMOOTHING_FACTOR)
                current_size = (w, h)
                center_x, center_y = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)
                final_box_to_draw = (center_x - w // 2, center_y - h // 2, w, h)
            else:
                tracking_active, frames_since_lost = False, 0

        if not tracking_active:
            cv2.putText(frame, "Object lost! Searching...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if frame_count % REDETECTION_INTERVAL == 0:
                det_results = model(frame, verbose=False, imgsz=416, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[target_class_id])
                
                reacquired_box = None
                if ENABLE_RE_IDENTIFICATION:
                    # --- Re-ID Logic: Find the best visual match ---
                    best_match_score, best_match_box = -1, None
                    for box_det in det_results[0].boxes:
                        candidate_box = tuple(map(int, box_det.xyxy[0].cpu().numpy())); candidate_box = (candidate_box[0], candidate_box[1], candidate_box[2] - candidate_box[0], candidate_box[3] - candidate_box[1])
                        candidate_features = get_color_histogram(frame, candidate_box)
                        if candidate_features is None: continue
                        similarity = cv2.compareHist(tracked_features, candidate_features, cv2.HISTCMP_CORREL)
                        if similarity > best_match_score: best_match_score, best_match_box = similarity, candidate_box
                    if best_match_box and best_match_score > REID_SIMILARITY_THRESHOLD:
                        reacquired_box = best_match_box
                        print(f"Re-acquired original target with similarity: {best_match_score:.2f}")
                else:
                    # --- Simple Logic: Grab the first available target ---
                    if len(det_results[0].boxes) > 0:
                        box_det = det_results[0].boxes[0]
                        reacquired_box = tuple(map(int, box_det.xyxy[0].cpu().numpy()))
                        reacquired_box = (reacquired_box[0], reacquired_box[1], reacquired_box[2] - reacquired_box[0], reacquired_box[3] - reacquired_box[1])
                        print("Re-acquired first available target (Re-ID disabled).")

                if reacquired_box:
                    tracker, tracking_active = create_tracker(TRACKER_TYPE), True
                    tracker.init(frame, reacquired_box)
                    current_size, target_size = (reacquired_box[2], reacquired_box[3]), (reacquired_box[2], reacquired_box[3])
                    final_box_to_draw = reacquired_box

        if final_box_to_draw:
            p1, p2 = (final_box_to_draw[0], final_box_to_draw[1]), (final_box_to_draw[0] + final_box_to_draw[2], final_box_to_draw[1] + final_box_to_draw[3])
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {int(fps)} ({TRACKER_TYPE})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Hybrid Tracking Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":
    main()