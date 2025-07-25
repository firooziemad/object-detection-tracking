import cv2
import numpy as np
from ultralytics import YOLO
from custom_tracker import CustomTracker
import argparse
import time

VIDEO_PATH = "person4.mp4"
TARGET_CLASS_NAME = "person"
MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.6
YOLO_CLASSES = { "person": 0, "car": 2, "dog": 16, "cat": 15, "bicycle": 1, "motorcycle": 3, "bottle": 39 }

PRESETS = {
    "person": {"min": 5000, "rmin": 2000, "mode": "normal"},
    "car": {"min": 8000, "rmin": 3000, "mode": "high_motion"},
    "dog": {"min": 3000, "rmin": 1500, "mode": "high_motion"},
    "cat": {"min": 2000, "rmin": 1000, "mode": "smooth"},
    "bicycle": {"min": 6000, "rmin": 2500, "mode": "normal"},
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Object Tracking System')
    parser.add_argument('--object', '-o', type=str, help='Object to track')
    parser.add_argument('--video', '-v', type=str, help='Video path')
    parser.add_argument('--ad', action='store_true', help='Enable auto re-detection mode')
    parser.add_argument('--hd', action='store_true', help='Enable history-based re-detection')
    return parser.parse_args()

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
    if area1 == 0 or area2 == 0: return 0
    return min(area1, area2) / max(area1, area2)

def get_best_detection(boxes, min_area, target_class_id):
    best_box, best_score, best_class_id = None, 0, None
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        if class_id != target_class_id: continue
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if area > min_area and confidence > best_score:
            best_score, best_box, best_class_id = confidence, box, class_id
    return best_box, best_score, best_class_id

def get_best_match(current_bbox, detections, history_bboxes):
    if not detections:
        return None

    best_match_box = None
    highest_score = -1

    for det in detections:
        xyxy = det.xyxy[0].cpu().numpy()
        conf = det.conf[0].cpu().numpy()
        det_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))

        # Score based on distance, size similarity, history, and confidence
        dist_score = max(0, 1 - (get_bbox_distance(current_bbox, det_bbox) / 250)) # Normalize distance score
        size_score = get_bbox_size_similarity(current_bbox, det_bbox)

        hist_score = 0
        if history_bboxes:
            hist_dists = [get_bbox_distance(hist_bbox, det_bbox) for hist_bbox in history_bboxes[-5:]]
            hist_score = max(0, 1 - (np.mean(hist_dists) / 250))

        # Weighted average of scores
        total_score = (dist_score * 0.4 + size_score * 0.3 + hist_score * 0.2 + conf * 0.1)

        if total_score > highest_score:
            highest_score = total_score
            best_match_box = det

    return best_match_box

def main():
    args = parse_arguments()
    target_class_name = args.object if args.object else TARGET_CLASS_NAME
    video_path = args.video if args.video else VIDEO_PATH
    target_class_id = YOLO_CLASSES.get(target_class_name.lower())
    auto_redetect = args.ad
    history_redetect = args.hd

    if target_class_id is None:
        print(f"Error: Object '{target_class_name}' not in YOLO_CLASSES list.")
        return

    config = PRESETS.get(target_class_name.lower(), {"min": 3000, "rmin": 1500, "mode": "normal"})
    print(f"Loaded config for '{target_class_name}': {config}")

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30

    success, frame = cap.read()
    if not success: return

    results = model(frame, verbose=False, classes=[target_class_id], conf=YOLO_CONFIDENCE_THRESHOLD)
    best_box, _, _ = get_best_detection(results[0].boxes, config['min'], target_class_id)
    if best_box is None: 
        print(f"Initial detection failed. No object found with minimum area {config['min']}.")
        return

    xyxy = best_box.xyxy[0].cpu().numpy()
    initial_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))

    tracker = CustomTracker()
    tracker.set_tracking_mode(config['mode'])
    tracker.init(frame, initial_bbox)

    bbox_history = [initial_bbox]
    paused = False
    redetect_counter = 0
    while True:
        if not paused:
            success, frame = cap.read()
            if not success: break

            tracking_success, box = tracker.update(frame)
            if tracking_success and box:
                bbox_history.append(box)
                if len(bbox_history) > 20:
                    bbox_history.pop(0)

            if auto_redetect and not tracking_success:
                redetect_counter += 1
                if redetect_counter >= int(fps / 2):
                    redetect_counter = 0
                    print("Auto re-detection...")
                    results = model(frame, verbose=False, classes=[target_class_id], conf=YOLO_CONFIDENCE_THRESHOLD * 0.8)
                    best_box, conf, _ = get_best_detection(results[0].boxes, config['rmin'], target_class_id)
                    if best_box is not None:
                        xyxy = best_box.xyxy[0].cpu().numpy()
                        new_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                        if tracker.init(frame, new_bbox):
                            tracking_success, box = True, new_bbox
                            bbox_history.append(new_bbox)

        display_frame = frame.copy()
        if tracking_success and box is not None:
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(display_frame, "TRACKING LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        info_y = 30
        cv2.putText(display_frame, f"Target: {target_class_name}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(display_frame, f"Mode: {tracker.tracking_mode.upper()}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        info_y += 20
        if auto_redetect:
            cv2.putText(display_frame, "Auto-redetect: ON", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            info_y += 20
        if history_redetect:
            cv2.putText(display_frame, "History-redetect: ON", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        cv2.imshow(f"Tracking - {target_class_name}", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('f'): auto_redetect = not auto_redetect
        elif key == ord('e'): history_redetect = not history_redetect
        elif key == ord('s'): tracker.set_tracking_mode("smooth")
        elif key == ord('h'): tracker.set_tracking_mode("high_motion")
        elif key == ord('n'): tracker.set_tracking_mode("normal")
        elif key == ord('r'):
            results = model(frame, verbose=False, classes=[target_class_id], conf=YOLO_CONFIDENCE_THRESHOLD * 0.8)
            best_box, conf, _ = get_best_detection(results[0].boxes, config['rmin'], target_class_id)
            if best_box is not None:
                xyxy = best_box.xyxy[0].cpu().numpy()
                new_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                tracker.init(frame, new_bbox)
                bbox_history.append(new_bbox)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()