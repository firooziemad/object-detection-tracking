import cv2
import numpy as np
from ultralytics import YOLO
from custom_tracker import Tracker
import sys
import time
import argparse

VIDEO = "input1.mp4"
OBJ = "car"
MODEL = "yolov8n.pt"
CONF = 0.4
MIN_AREA = 50

CLASSES = {
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

class Tracker:
    def __init__(self, mode='normal'):
        self.trackers = {}
        self.next_id = 0
        self.mode = mode
        self.colors = {}

    def color(self, id):
        if id not in self.colors:
            np.random.seed(id)
            self.colors[id] = (np.random.randint(50, 255), 
                               np.random.randint(50, 255), 
                               np.random.randint(50, 255))
        return self.colors[id]

    def init_all(self, frame, bboxes):
        for box in bboxes:
            t = CustomTracker()
            t.set_tracking_mode(self.mode)
            if t.init(frame, box):
                self.trackers[self.next_id] = t
                print(f"Initialized tracker ID: {self.next_id} at {box}")
                self.next_id += 1
        print(f"\nSuccessfully initialized {len(self.trackers)} trackers.")

    def update(self, frame):
        boxes = {}
        lost = []
        for id, t in self.trackers.items():
            ok, box = t.update(frame)
            if ok:
                boxes[id] = box
            else:
                lost.append(id)
        for id in lost:
            print(f"Tracker {id} lost.")
            del self.trackers[id]
        return boxes

    def draw(self, frame):
        for id, t in self.trackers.items():
            if t.bbox is not None:
                x, y, w, h = t.bbox
                color = self.color(id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                label = f"ID: {id}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def detections(boxes, min_area, class_id):
    out = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cid = int(box.cls[0].cpu().numpy())
        if cid == class_id:
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            if area > min_area:
                bbox = (int(xyxy[0]), int(xyxy[1]), int(w), int(h))
                out.append(bbox)
    return out

def main():
    parser = argparse.ArgumentParser(description='Multi-Object Tracking using YOLO and CustomTracker')
    parser.add_argument('--object', '-o', type=str, default=OBJ, help='Object to track')
    parser.add_argument('--video', '-v', type=str, default=VIDEO, help='Path to video file')
    parser.add_argument('--confidence', '-c', type=float, default=CONF, help='YOLO confidence threshold')
    parser.add_argument('--mode', '-m', type=str, default='high_motion', choices=['smooth', 'normal', 'high_motion'], help='Tracking mode')
    args = parser.parse_args()

    obj = args.object
    if obj not in CLASSES:
        print(f"Error: Object '{obj}' not in CLASSES.")
        return
    cid = CLASSES[obj]

    print("Configuration:")
    print(f"  - Video: {args.video}")
    print(f"  - Target: {obj} (ID: {cid})")
    print(f"  - Confidence: {args.confidence}")
    print(f"  - Tracking Mode: {args.mode}")
    print(f"  - Min Initial Area: {MIN_AREA}")

    print("\nLoading YOLO model...")
    model = YOLO(MODEL)
    model.overrides['verbose'] = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        return

    ok, frame = cap.read()
    if not ok:
        print("Error reading the first frame.")
        cap.release()
        return

    print(f"\nDetecting all '{obj}' objects in the first frame...")
    results = model(frame, verbose=False, classes=[cid], conf=args.confidence)
    bboxes = detections(results[0].boxes, MIN_AREA, cid)
    if not bboxes:
        print(f"Could not find any '{obj}' objects meeting the criteria in the first frame.")
        print("Try lowering the --confidence or checking the video content.")
        cap.release()
        return
    print(f"Found {len(bboxes)} initial objects to track.")

    tracker = Tracker(mode=args.mode)
    tracker.init_all(frame, bboxes)

    count = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("\nEnd of video.")
                break
            count += 1
        if not paused:
            tracker.update(frame)
        show = frame.copy()
        show = tracker.draw(show)
        info = f"Frame: {count} | Tracking {len(tracker.trackers)} Objects | Mode: {args.mode.upper()}"
        cv2.putText(show, info, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if paused:
            cv2.putText(show, "PAUSED", (10, show.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(f"Multi-Object Tracking - {obj}", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' ') and paused:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            tracker.update(frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cv2.destroyAllWindows()