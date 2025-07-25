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
YOLO_CLASSES = { "person": 0, "car": 2, "dog": 16, "cat": 15 }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Object Tracking System')
    parser.add_argument('--object', '-o', type=str, help='Object to track')
    parser.add_argument('--video', '-v', type=str, help='Video path')
    return parser.parse_args()

def get_best_detection(boxes, min_area, target_class_id):
    best_box, best_score = None, 0
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        if class_id != target_class_id: continue
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if area > min_area and confidence > best_score:
            best_score, best_box = confidence, box
    return best_box, best_score

def main():
    args = parse_arguments()
    target_class_name = args.object if args.object else TARGET_CLASS_NAME
    video_path = args.video if args.video else VIDEO_PATH
    target_class_id = YOLO_CLASSES.get(target_class_name.lower())

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    success, frame = cap.read()
    results = model(frame, verbose=False, classes=[target_class_id], conf=YOLO_CONFIDENCE_THRESHOLD)
    best_box, _ = get_best_detection(results[0].boxes, 5000, target_class_id)
    if best_box is None:
        print("Detection failed."); return
        
    xyxy = best_box.xyxy[0].cpu().numpy()
    initial_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
    
    tracker = CustomTracker()
    tracker.init(frame, initial_bbox)
    
    frame_count = 0
    paused = False

    while True:
        if not paused:
            success, frame = cap.read()
            if not success: break
            frame_count += 1
            tracking_success, box = tracker.update(frame)
        
        display_frame = frame.copy()
        if tracking_success and box is not None:
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display_frame, f"Tracking: {len(tracker.tracks)} features", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "TRACKING LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        info_y = 30
        cv2.putText(display_frame, f"Target: {target_class_name}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 20
        cv2.putText(display_frame, f"Mode: {tracker.tracking_mode.upper()}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        if paused:
            cv2.putText(display_frame, "PAUSED", (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(f"Tracking - {target_class_name}", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('s'):
            tracker.set_tracking_mode("smooth")
            print("Switched to SMOOTH mode")
        elif key == ord('h'):
            tracker.set_tracking_mode("high_motion")
            print("Switched to HIGH MOTION mode")
        elif key == ord('n'):
            tracker.set_tracking_mode("normal")
            print("Switched to NORMAL mode")
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()