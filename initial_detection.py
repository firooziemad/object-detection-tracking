import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. CONFIGURATION ---
# The path to your video file.
VIDEO_PATH = "person4.mp4" 
# The name of the object you want to detect (e.g., "person", "car", "cat").
# Make sure this class is in the COCO dataset, which YOLO was trained on.
TARGET_CLASS_NAME = "person" 
# The YOLO model to use. "yolov8n.pt" is small and fast.
# The model will be downloaded automatically the first time you run this.
MODEL_NAME = "yolov8n.pt" 

def main():
    # --- 2. LOAD THE MODEL ---
    print(f"Loading model {MODEL_NAME}...")
    # This line loads the pre-trained YOLO model.
    model = YOLO(MODEL_NAME)
    print("Model loaded successfully.")

    # --- 3. OPEN THE VIDEO AND GET THE FIRST FRAME ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    # Check if the video opened correctly.
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    # Read the first frame from the video.
    success, first_frame = cap.read()
    if not success:
        print("Error: Could not read the first frame from the video.")
        cap.release()
        return
        
    print("First frame read successfully.")

    # --- 4. PERFORM DETECTION ON THE FIRST FRAME ---
    # The model() call runs the detection.
    results = model(first_frame, verbose=False) 

    # The 'results' object contains all detections. We work with the first (and only) result.
    result = results[0]
    initial_bbox = None # This will store our target's bounding box.

    print(f"Searching for '{TARGET_CLASS_NAME}' in the first frame...")
    # Loop through all the detected boxes in the frame.
    for box in result.boxes:
        # Get the class ID and the class name.
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Check if the detected object is our target.
        if class_name.lower() == TARGET_CLASS_NAME.lower():
            print(f"Found '{TARGET_CLASS_NAME}'!")
            # Get the bounding box coordinates in [x1, y1, x2, y2] format.
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Convert to OpenCV's desired [x, y, width, height] format.
            x = int(xyxy[0])
            y = int(xyxy[1])
            width = int(xyxy[2] - xyxy[0])
            height = int(xyxy[3] - xyxy[1])
            initial_bbox = (x, y, width, height)
            
            # We found our target, so we can stop searching.
            break 
    
    # --- 5. VISUALIZE THE RESULT AND CLEAN UP ---
    if initial_bbox is not None:
        # Draw the bounding box on the frame.
        p1 = (initial_bbox[0], initial_bbox[1])
        p2 = (initial_bbox[0] + initial_bbox[2], initial_bbox[1] + initial_bbox[3])
        cv2.rectangle(first_frame, p1, p2, (0, 255, 0), 2)

        # Add a text label.
        label = f"Tracking: {TARGET_CLASS_NAME}"
        cv2.putText(first_frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Initial bounding box for tracking: {initial_bbox}")
        
        # Display the result in a window.
        cv2.imshow("Initial Detection", first_frame)
        print("Press any key to close the window.")
        cv2.waitKey(0) # Wait until a key is pressed.

    else:
        print(f"Could not find '{TARGET_CLASS_NAME}' in the first frame.")

    # Release resources.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()