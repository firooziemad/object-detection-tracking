import cv2
from ultralytics import YOLO
from custom_tracker import CustomTracker

VIDEO_PATH = "person4.mp4"
TARGET_CLASS_NAME = "person"
MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.6

YOLO_CLASSES = {"person": 0, "car": 2, "dog": 16, "cat": 15}

def get_best_detection(boxes, min_area, target_class_id):
    best_box = None
    best_score = 0
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        if class_id != target_class_id: continue
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if area > min_area and confidence > best_score:
            best_score = confidence
            best_box = box
    return best_box, best_score

def main():
    target_class_id = YOLO_CLASSES.get(TARGET_CLASS_NAME.lower())
    if target_class_id is None:
        print(f"Error: Unknown object '{TARGET_CLASS_NAME}'")
        return

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, frame = cap.read()
    if not success: return

    results = model(frame, verbose=False, classes=[target_class_id], conf=YOLO_CONFIDENCE_THRESHOLD)
    best_box, _ = get_best_detection(results[0].boxes, 5000, target_class_id)
    if best_box is None:
        print(f"No suitable {TARGET_CLASS_NAME} found.")
        return

    xyxy = best_box.xyxy[0].cpu().numpy()
    initial_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
    
    tracker = CustomTracker()
    tracker.init(frame, initial_bbox)

    while True:
        success, frame = cap.read()
        if not success: break

        tracking_success, box = tracker.update(frame)

        if tracking_success:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "TRACKING LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow(f"Tracking - {TARGET_CLASS_NAME}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()